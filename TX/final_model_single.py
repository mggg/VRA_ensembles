# -*- coding: utf-8 -*-
"""
Created on Mon Jul 13 13:29:26 2020

@author: darac
"""
import random
import csv
import os
import shutil
from functools import partial
import json
import math
import numpy as np
import geopandas as gpd
import matplotlib
#matplotlib.use('Agg')
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd
from gerrychain import (
    Election,
    Graph,
    MarkovChain,
    Partition,
    accept,
    constraints,
    updaters,
)
from gerrychain.metrics import efficiency_gap, mean_median
from gerrychain.proposals import recom
from gerrychain.updaters import cut_edges
from gerrychain.updaters import *
from gerrychain.tree import recursive_tree_part
from gerrychain.updaters import Tally
from gerrychain import GeographicPartition
from scipy.spatial import ConvexHull
from gerrychain.proposals import recom, propose_random_flip
from gerrychain.tree import recursive_tree_part
from gerrychain.accept import always_accept
from gerrychain.constraints import single_flip_contiguous, Validator
import collections
from enum import Enum
import re
import statsmodels.formula.api as smf
import statsmodels.api as sm
import scipy
from scipy import stats
import intervals as I
import time
import heapq
import operator
from operator import itemgetter
from ER_functions import f, norm_dist_params, ER_run, preferred_cand, accrue_points
from ast import literal_eval

DIR = ''
#user inputs
map_num_test = 0
display_dist = 28#0 index
display_elec = '16G_President'
run_name = 'local_test_prob'
tot_pop = 'TOTPOP_x'
num_districts = 36
cand_drop_thresh = 0
model_mode = 'district' #or district, statewide
plot_path = 'tx-results-cvap-adjoined/tx-results-cvap-adjoined.shp'  #for shapefile
assign_test = "CD" #map to assess (by column title in shapefile)
#assign_test = "Map{}".format(map_num_test)

#read files
elec_data = pd.read_csv("TX_elections.csv")
elections = list(elec_data["Election"]) #elections we care about

elec_type = elec_data["Type"]
election_returns = pd.read_csv("TX_statewide_election_returns.csv")
dropped_elecs = pd.read_csv("dropped_elecs.csv")["Dropped Elections"]
elec_cand_list = list(election_returns.columns)[2:] #all candidates in all elections
recency_weights = pd.read_csv("recency_weights.csv")
min_cand_weights = pd.read_csv("min_pref_weight_binary.csv")
cand_race_table = pd.read_csv("CandidateRace.csv")
EI_statewide = pd.read_csv("EI_statewide_data.csv")

#elections data structures
elecs_bool = ~elec_data.Election.isin(list(dropped_elecs))
elec_data_trunc = elec_data[elecs_bool].reset_index(drop = True)
elec_sets = list(set(elec_data_trunc["Election Set"]))
elections = list(elec_data_trunc["Election"])
general_elecs = list(elec_data_trunc[elec_data_trunc["Type"] == 'General'].Election)
primary_elecs = list(elec_data_trunc[elec_data_trunc["Type"] == 'Primary'].Election)
runoff_elecs = list(elec_data_trunc[elec_data_trunc["Type"] == 'Runoff'].Election)
elec_set_dict = {}
for elec_set in elec_sets:
    elec_set_df = elec_data_trunc[elec_data_trunc["Election Set"] == elec_set]
    elec_set_dict[elec_set] = dict(zip(elec_set_df.Type, elec_set_df.Election))
elec_match_dict = dict(zip(elec_data_trunc["Election"], elec_data_trunc["Election Set"]))
#for effectiveness on different maps:
#stored_plans = pd.read_csv("store_plans_TX_chain_free_NEW.csv")

state_gdf = gpd.read_file(plot_path)

state_gdf["CD"] = [int(i) for i in state_gdf["CD"]]
#state_gdf["Map{}".format(map_num_test)] = state_gdf.index.map(dict(zip(stored_plans["Index"], stored_plans["Map{}".format(map_num_test)])))
#to edit cut off shape file columns
election_return_cols = list(election_returns.columns)
cand1_index = election_return_cols.index('RomneyR_12G_President') #first
cand2_index = election_return_cols.index('ObamaD_12P_President') #last
elec_results_trunc = election_return_cols[cand1_index:cand2_index+1]
state_gdf_cols = list(state_gdf.columns)
cand1_index = state_gdf_cols.index('RomneyR_12')
cand2_index = state_gdf_cols.index('ObamaD_12P')
state_gdf_cols[cand1_index:cand2_index+1] = elec_results_trunc
state_gdf.columns = state_gdf_cols

#make graph from gdf
graph = Graph.from_geodataframe(state_gdf)
graph.add_data(state_gdf)

#reformat elections return - get summary stats
state_df = pd.DataFrame(state_gdf)
state_df = state_df.drop(['geometry'], axis = 1)
state_df.columns = state_df.columns.str.replace("-", "_")

cvap_types = ['CVAP', 'WCVAP', 'BCVAP', 'HCVAP']
cvap_codes = ['1', '7', '5', '13'] 
cvap_key = dict(zip(cvap_types,cvap_codes ))
cvap_years = [2012, 2014, 2016, 2018]
cvap_columns = {year:  {t: cvap_key[t] + '_' + str(year) for t in cvap_types} for year in cvap_years}

#drop CVAP = 0 precints
#state_df = state_df[state_df["CVAP"] > 0] #DOUBLE CHECK JUST FOR REGRESSIONS IN CHAIN MODEL!!
#make candidate dictionary (key is election and value is candidates)
candidates = {}
for elec in elections: #only elections we care about
    #get rid of republican candidates in primaries or runoffs (primary runoffs)
    cands = [y for y in elec_cand_list if elec in y and "R_" not in re.sub(elec, '', y) ] if \
    "R_" in elec or "P_" in elec else [y for y in elec_cand_list if elec in y] 
    
    elec_year = elec_data_trunc.loc[elec_data_trunc["Election"] == elec, 'Year'].values[0]          
    #in general elections, only include 2 major party candidates
    #in all other elections, only include candidates whose vote share is above cand_drop_thresh
    if elec in general_elecs:
        cands = cands[:2]
    if elec not in general_elecs:
       pattern = '|'.join(cands)
       elec_df = state_df.copy().loc[:, state_df.columns.str.contains(pattern)]
       elec_df["Total"] = elec_df.sum(axis=1)
       if elec == '18P_Governor':
           elec_df.to_csv(DIR + "outputs/elec test.csv")
       for cand in cands:
           if sum(elec_df["{}".format(cand)])/sum(elec_df["Total"]) < cand_drop_thresh:
               cands = [i for i in cands if i != cand]   
               print("removed!", cand)

    for cand in cands:
        state_df["{}%CVAP".format(cand)] = state_df["{}".format(cand)]/state_df[cvap_columns[elec_year]['CVAP']]    
        state_df["{}%CVAP".format(cand)] = [min(x,1) for x in list(state_df["{}%CVAP".format(cand)])]
    candidates[elec] = dict(zip(list(range(len(cands))), cands))

#state_df["WCVAP%"] = state_df["WCVAP"]/state_df["CVAP"]
#state_df["HCVAP%"] = state_df["HCVAP"]/state_df["CVAP"]
#state_df["BCVAP%"] = state_df["BCVAP"]/state_df["CVAP"]
            
state_df.to_csv(DIR + "outputs/state_df.csv")

my_updaters = {
        "population": updaters.Tally(tot_pop, alias = "population")
      }

election_functions = [Election(j, candidates[j]) for j in elections]
election_updaters = {election.name: election for election in election_functions}
my_updaters.update(election_updaters)

partition = GeographicPartition(graph = graph, assignment = assign_test, updaters = my_updaters)
num_districts = len(partition)

def winner(partition, election, elec_cands):
    order = [x for x in partition.parts]
    perc_for_cand = {}
    for j in range(len(elec_cands)):
        perc_for_cand[j] = dict(zip(order, partition[election].percents(j)))
    winners = {}
    for i in range(len(partition)):
        dist_percents = [perc_for_cand[z][i] for z in range(len(elec_cands))]
        winner_index = dist_percents.index(max(dist_percents))
        winners[i] = elec_cands[winner_index]
    return winners

#key: election, value: dictionary with keys = dists and values are dicts of % for each cand
#for particular elec and dist can access all cand results by: dist_elec_results[elec][dist]
dist_elec_results = {}
order = [x for x in partition.parts]
for elec in elections:
    cands = candidates[elec]
    dist_elec_results[elec] = {}
    outcome_list = [dict(zip(order, partition[elec].percents(cand))) for cand in cands.keys()]      
    dist_elec_results[elec] = {d: {cands[i]: outcome_list[i][d] for i in cands.keys()} for d in range(num_districts)}
        

#prepare dataframes for results all for single map
#df 1B and 1H: rows = elecs, columns = districtss, entry is black or hisp pref cand (map-specific)
#df 2: winners: rows = elec, columns = districts, entry is winner (note this df is for single map)
#df 3W1: rows = elec, columns = districts, entry is recency weight (not district or race specific)
#df 3W2: rows = elec, columns = districts, entries are min-preferred min weights (district specific, not race-specific)
#df 3BW3, 3HW3: rows = elec, columns = districts, entries are preferred-candidate-confidence weights (district and race specific)
    #(these are combined into one 3W3 df, by averaging the pairwise values in 3BW3 and 3HW3)

#df 1B and 1H: black_pref_cands_df, hisp_pref_cands_df
#make df with preferred candidate in every district for every election (only for enacted map)
        #also in this section- 2aii and 2bii dfs, or those whose entries are confidence in first choice pref. candidate pick
black_pref_cands_df = pd.DataFrame(columns = range(num_districts))
black_pref_cands_df["Election Set"] = elec_sets
hisp_pref_cands_df = pd.DataFrame(columns = range(num_districts))
hisp_pref_cands_df["Election Set"] = elec_sets

#store runoff preferences for instances where min-pref candidate needs to switch btwn prim and runoff
black_pref_cands_runoffs = pd.DataFrame(columns = range(num_districts))
black_pref_cands_runoffs["Election Set"] = elec_sets
hisp_pref_cands_runoffs = pd.DataFrame(columns = range(num_districts))
hisp_pref_cands_runoffs["Election Set"] = elec_sets

black_conf_W3 = pd.DataFrame(columns = range(num_districts))
black_conf_W3["Election Set"] = elec_sets
hisp_conf_W3 = pd.DataFrame(columns = range(num_districts))
hisp_conf_W3["Election Set"] = elec_sets    

start_time = time.time()

#compute district winners
#df 3 winners (this example just for enacted map)
dist_winners = {} #adding district winners for each election to that election's df
map_winners = pd.DataFrame(columns = range(num_districts))

for j in elections:
    dist_winners[j] = winner(partition, j, candidates[j])
    keys = list(dist_winners[j].keys())
    values = list(dist_winners[j].values())
    map_winners.loc[len(map_winners)] = [value for _,value in sorted(zip(keys,values))]
    
map_winners = map_winners.reset_index(drop = True)
map_winners["Election"] = elections
map_winners["Election Set"] = elec_data_trunc["Election Set"]
map_winners["Election Type"] = elec_data_trunc["Type"]
map_winners.to_csv(DIR + "outputs/winnersElec.csv")  
  
for district in range(num_districts): #get vector of precinct values for each district       
    dist_df = state_df[state_df[assign_test] == district]   
    
    for elec in primary_elecs + runoff_elecs:
        elec_year = elec_data_trunc.loc[elec_data_trunc["Election"] == elec, 'Year'].values[0]

        cvap = cvap_columns[elec_year]['CVAP']
        white_cvap = cvap_columns[elec_year]['WCVAP']
        black_cvap = cvap_columns[elec_year]['BCVAP']
        hisp_cvap = cvap_columns[elec_year]['HCVAP']
        
        dist_df = dist_df[dist_df[cvap] > 0] #drop rows with that year's cvap = 0
        black_share = list(dist_df[black_cvap]/dist_df[cvap])
        hisp_share = list(dist_df[hisp_cvap]/dist_df[cvap])
        white_share = list(dist_df[white_cvap]/dist_df[cvap])  
               
        #run ER regressions for black and Latino voters
        #determine black and Latino preferred candidates and confidence preferred-cand is correct
        pop_weights = list(dist_df.loc[:,cvap].apply(lambda x: x/sum(dist_df[cvap])))           
        
        if model_mode == 'statewide':
            black_norm_params = literal_eval(EI_statewide.loc[EI_statewide["Election"] == elec, 'Black Preferred'].values[0])
            hisp_norm_params = literal_eval(EI_statewide.loc[EI_statewide["Election"] == elec, 'Latino Preferred'].values[0])
        else:
            black_norm_params = {}
            hisp_norm_params = {}
            for cand in candidates[elec].values():
                cand_cvap_share = list(dist_df["{}%CVAP".format(cand)])
                          
            #regrss cand share of total vote on demo-share-CVAP, black and latino voters                                                                                  
                mean, std = ER_run(cand,elec, district, black_share, cand_cvap_share,\
                       pop_weights, black_norm_params, display_dist, display_elec)
                black_norm_params[cand] = [mean, std]
       
                mean, std = ER_run(cand,elec, district, hisp_share, cand_cvap_share,\
                       pop_weights, hisp_norm_params, display_dist, display_elec)
                hisp_norm_params[cand] = [mean, std]

        #optimizations for confidence! (W3)
        #populate black pref candidate and confidence in candidate (df 1a and 2aii)
        #if after dropping candidates under cand_drop_thresh, only one left, that is preferred candidate                          
        
        black_pref_cand, black_er_conf = preferred_cand(district, elec, black_norm_params, display_dist, display_elec)
        hisp_pref_cand, hisp_er_conf = preferred_cand(district, elec, hisp_norm_params, display_dist, display_elec)
        if elec in primary_elecs:
            black_pref_cands_df.at[black_pref_cands_df["Election Set"] == elec_match_dict[elec], district] = black_pref_cand
            black_conf_W3.at[black_conf_W3["Election Set"] == elec_match_dict[elec], district] = black_er_conf
            hisp_pref_cands_df.at[hisp_pref_cands_df["Election Set"] == elec_match_dict[elec], district] = hisp_pref_cand
            hisp_conf_W3.at[hisp_conf_W3["Election Set"] == elec_match_dict[elec], district] = hisp_er_conf         
        else:
            black_pref_cands_runoffs.at[black_pref_cands_runoffs["Election Set"] == elec_match_dict[elec], district] = black_pref_cand
            hisp_pref_cands_runoffs.at[hisp_pref_cands_runoffs["Election Set"] == elec_match_dict[elec], district] = hisp_pref_cand
        
neither_conf_W3 = black_conf_W3.drop(["Election Set"], axis =1)* hisp_conf_W3.drop(["Election Set"], axis =1)
neither_conf_W3["Election Set"] = elec_sets
        
#########################################################################################
#get election weights 1 and 2 and combine for final
black_weight_df = pd.DataFrame(columns = range(num_districts))
hisp_weight_df = pd.DataFrame(columns = range(num_districts))

#get weights W1 and W2 for weighting elections.   
recency_W1 = pd.DataFrame(columns = range(num_districts))
recency_W1["Election Set"] = elec_sets

min_cand_black_W2 = pd.DataFrame(columns = range(num_districts))
min_cand_black_W2["Election Set"] = elec_sets

min_cand_hisp_W2 = pd.DataFrame(columns = range(num_districts))
min_cand_hisp_W2["Election Set"] = elec_sets

min_cand_neither_W2 = pd.DataFrame(columns = range(num_districts))
min_cand_neither_W2["Election Set"] = elec_sets
        
for elec_set in elec_sets:
    elec_year = elec_data_trunc.loc[elec_data_trunc["Election Set"] == elec_set, 'Year'].values[0].astype(str)
    for dist in range(num_districts):      
        recency_W1.at[recency_W1["Election Set"] == elec_set, dist] = recency_weights[elec_year][0]

        black_pref = black_pref_cands_df.loc[black_pref_cands_df["Election Set"] == elec_set, dist].values[0]
        black_pref_race = cand_race_table.loc[cand_race_table["Candidates"] == black_pref, "Race"].values[0]
        black_pref_black = True if 'Black' in black_pref_race else False        
        black_cand_weight_type = 'Relevant Minority' if black_pref_black else 'Other'
        min_cand_black_W2.at[min_cand_black_W2["Election Set"] == elec_set, dist] = min_cand_weights[black_cand_weight_type][0] 
        
        hisp_pref = hisp_pref_cands_df.loc[hisp_pref_cands_df["Election Set"] == elec_set, dist].values[0]
        hisp_pref_race = cand_race_table.loc[cand_race_table["Candidates"] == hisp_pref, "Race"].values[0]
        hisp_pref_hisp = True if 'Hispanic' in hisp_pref_race else False             
        hisp_cand_weight_type = 'Relevant Minority' if hisp_pref_hisp else 'Other'
        min_cand_hisp_W2.at[min_cand_hisp_W2["Election Set"] == elec_set, dist] = min_cand_weights[hisp_cand_weight_type][0] 
        
        neither_cand_weight_type = 'Relevant Minority' if (hisp_pref_hisp & black_pref_black) else \
                'Other' if (not hisp_pref_hisp and not black_pref_black) else 'Partial Minority'
        min_cand_neither_W2.at[min_cand_neither_W2["Election Set"] == elec_set, dist] = min_cand_weights[neither_cand_weight_type][0] 


#final 2a and 2b election probativity scores
black_weight_df = recency_W1.drop(["Election Set"], axis=1)*min_cand_black_W2.drop(["Election Set"], axis=1)*black_conf_W3.drop(["Election Set"], axis=1)
black_weight_df["Election Set"] = elec_sets
hisp_weight_df = recency_W1.drop(["Election Set"], axis=1)*min_cand_hisp_W2.drop(["Election Set"], axis=1)*hisp_conf_W3.drop(["Election Set"], axis=1)    
hisp_weight_df["Election Set"] = elec_sets
neither_weight_df = recency_W1.drop(["Election Set"], axis=1)*min_cand_neither_W2.drop(["Election Set"], axis=1)*neither_conf_W3.drop(["Election Set"], axis=1)    
neither_weight_df["Election Set"] = elec_sets


if model_mode == 'equal':
    for col in black_weight_df.columns[:len(black_weight_df.columns)-1]:
        black_weight_df[col].values[:] = 1
    for col in hisp_weight_df.columns[:len(hisp_weight_df.columns)-1]:
        hisp_weight_df[col].values[:] = 1
    
##############################################################################
#accrue points for black and hispanic voters if cand-of-choice wins
general_winners = map_winners[map_winners["Election Type"] == 'General'].reset_index(drop = True)
primary_winners = map_winners[map_winners["Election Type"] == 'Primary'].reset_index(drop = True)
runoff_winners = map_winners[map_winners["Election Type"] == 'Runoff'].reset_index(drop = True)

#determine if election set accrues points by district for black and Latino voters
black_pref_wins = pd.DataFrame(columns = range(num_districts))
black_pref_wins["Election Set"] = elec_sets
hisp_pref_wins = pd.DataFrame(columns = range(num_districts))
hisp_pref_wins["Election Set"] = elec_sets

for i in range(num_districts):
    for elec_set in elec_sets:
        black_pref_cand = black_pref_cands_df.loc[black_pref_cands_df["Election Set"] == elec_set, i].values[0]
        hisp_pref_cand = hisp_pref_cands_df.loc[hisp_pref_cands_df["Election Set"] == elec_set, i].values[0]       
        
        primary_race = elec_set_dict[elec_set]["Primary"]
        runoff_race = None if 'Runoff' not in elec_set_dict[elec_set].keys() else elec_set_dict[elec_set]["Runoff"]
        general_race = elec_set_dict[elec_set]["General"]
        
        primary_winner = primary_winners.loc[primary_winners["Election Set"] == elec_set, i].values[0]
        general_winner = general_winners.loc[general_winners["Election Set"] == elec_set, i].values[0]
        runoff_winner = "N/A" if elec_set not in list(runoff_winners["Election Set"]) \
        else runoff_winners.loc[runoff_winners["Election Set"] == elec_set, i].values[0]
        
        primary_race_shares = dist_elec_results[primary_race][i]
        primary_ranking = {key: rank for rank, key in enumerate(sorted(primary_race_shares, key=primary_race_shares.get, reverse=True), 1)}        
        black_pref_prim_share = primary_race_shares[black_pref_cand]
        hisp_pref_prim_share = primary_race_shares[hisp_pref_cand]
        
        black_pref_prim_rank = primary_ranking[black_pref_cand]
        hisp_pref_prim_rank = primary_ranking[hisp_pref_cand]
        
        party_general_winner = cand_race_table.loc[cand_race_table["Candidates"] == general_winner, "Party"].values[0] 
         
        #need to compute new minority-preferred-candidate for runoff?
        #yes if min-pref wins district primary, but not on runoff ballot
        runoff_black_pref = "N/A" if runoff_winner == "N/A" else \
                    black_pref_cands_runoffs.loc[black_pref_cands_runoffs["Election Set"] == elec_set, i].values[0]
        
        runoff_hisp_pref = "N/A" if runoff_winner == 'N/A' else \
                    hisp_pref_cands_runoffs.loc[hisp_pref_cands_runoffs["Election Set"] == elec_set, i].values[0]
                         
        #winning conditions (conditions to accrue points for election set/minority group):
        black_accrue = accrue_points(primary_winner, black_pref_cand, party_general_winner, black_pref_prim_rank, \
                  runoff_winner, runoff_black_pref, candidates, runoff_race)
        black_pref_wins.at[black_pref_wins["Election Set"] == elec_set, i] = black_accrue 
        
        hisp_accrue = accrue_points(primary_winner, hisp_pref_cand, party_general_winner, hisp_pref_prim_rank, 
                  runoff_winner, runoff_hisp_pref, candidates, runoff_race)
        hisp_pref_wins.at[hisp_pref_wins["Election Set"] == elec_set, i] = hisp_accrue 
        

neither_pref_wins = (1-black_pref_wins.drop(['Election Set'], axis = 1))*(1-hisp_pref_wins.drop(['Election Set'], axis = 1))
neither_pref_wins["Election Set"] = elec_sets

neither_points_accrued = neither_weight_df.drop(['Election Set'], axis = 1)*neither_pref_wins.drop(['Election Set'], axis = 1)  
neither_points_accrued["Election Set"] = elec_sets
black_points_accrued = black_weight_df.drop(['Election Set'], axis = 1)*black_pref_wins.drop(['Election Set'], axis = 1)  
black_points_accrued["Election Set"] = elec_sets
hisp_points_accrued = hisp_weight_df.drop(['Election Set'], axis = 1)*hisp_pref_wins.drop(['Election Set'], axis = 1)      
hisp_points_accrued["Election Set"] = elec_sets

###################################################################    
#Compute district probabilities: black, Latino, neither and overlap 
black_vra_prob = [0 if sum(black_weight_df[i]) == 0 else sum(black_points_accrued[i])/sum(black_weight_df[i]) for i in range(num_districts)]
hisp_vra_prob = [0 if sum(hisp_weight_df[i])  == 0 else sum(hisp_points_accrued[i])/sum(hisp_weight_df[i]) for i in range(num_districts)]   
neither_vra_prob = [0 if sum(neither_weight_df[i])  == 0 else sum(neither_points_accrued[i])/sum(neither_weight_df[i]) for i in range(num_districts)]   

min_neither = [0 if (black_vra_prob[i] + hisp_vra_prob[i]) > 1 else 1 -(black_vra_prob[i] + hisp_vra_prob[i]) for i in range(num_districts)]
max_neither = [1 - max(black_vra_prob[i], hisp_vra_prob[i]) for i in range(num_districts)]

final_neither = [min_neither[i] + neither_vra_prob[i]*(max_neither[i]-min_neither[i]) for i in range(num_districts)]
final_overlap = [final_neither[i] + black_vra_prob[i] + hisp_vra_prob[i] - 1 for i in range(num_districts)]
final_black_prob = [black_vra_prob[i] - final_overlap[i] for i in range(num_districts)]
final_hisp_prob = [hisp_vra_prob[i] - final_overlap[i] for i in range(num_districts)]
################################################## 
#District by district probability breakdowns
dist_perc_df = pd.DataFrame(columns = ["District"])
dist_perc_df["District"] = list(range(1, num_districts+1))
dist_perc_df["Original Latino Perc"] = hisp_vra_prob
dist_perc_df["Original Black Perc"] = black_vra_prob
dist_perc_df["Final Latino"] = final_hisp_prob
dist_perc_df["Final Black"] = final_black_prob
dist_perc_df["Final Neither"] = final_neither
dist_perc_df["Final Overlap"] = final_overlap
dist_perc_df.to_csv(DIR + "outputs/Dist_perc df, {} mode.csv".format(model_mode), index = False)

#district deep dive
district_df = pd.DataFrame(columns = ["Election Set"])
district_entry = 29
district = district_entry-1
district_df["Election Set"] = elec_sets
district_df["Primary Winner"] = district_df["Election Set"].map(dict(zip(primary_winners["Election Set"], primary_winners[district])))
district_df["Runoff Winner"] = district_df["Election Set"].map(dict(zip(runoff_winners["Election Set"], runoff_winners[district])))
district_df["General Winner"] = district_df["Election Set"].map(dict(zip(general_winners["Election Set"], general_winners[district])))
district_df["Black pref cand (primary)"] = district_df["Election Set"].map(dict(zip(black_pref_cands_df["Election Set"], black_pref_cands_df[district])))
district_df["Hisp pref cand (primary)"] = district_df["Election Set"].map(dict(zip(hisp_pref_cands_df["Election Set"], hisp_pref_cands_df[district])))
district_df["Black pref cand (runoff)"] = district_df["Election Set"].map(dict(zip(black_pref_cands_runoffs["Election Set"], black_pref_cands_runoffs[district])))
district_df["Hisp pref cand (runoff)"] = district_df["Election Set"].map(dict(zip(hisp_pref_cands_runoffs["Election Set"], hisp_pref_cands_runoffs[district])))
district_df["Black primary cand wins"] = district_df["Black pref cand (primary)"] == district_df["Primary Winner"]
district_df["Hisp primary cand wins"] = district_df["Hisp pref cand (primary)"] == district_df["Primary Winner"]
district_df["Black runoff cand wins"] = ["N/A" if pd.isna(district_df["Runoff Winner"][i]) else \
           district_df["Black pref cand (runoff)"][i] == district_df["Runoff Winner"][i] for i in range(len(district_df))]
district_df["Hisp runoff cand wins"] = ["N/A" if pd.isna(district_df["Runoff Winner"][i]) else \
           district_df["Hisp pref cand (runoff)"][i] == district_df["Runoff Winner"][i] for i in range(len(district_df))]
district_df["Black accrue points"] = district_df["Election Set"].map(dict(zip(black_pref_wins["Election Set"], black_pref_wins[district])))
district_df["Hisp accrue points"] = district_df["Election Set"].map(dict(zip(hisp_pref_wins["Election Set"], hisp_pref_wins[district])))
district_df["Recency W1"] = district_df["Election Set"].map(dict(zip(recency_W1["Election Set"], recency_W1[district])))
district_df["Min Pref Min Black W2"] = district_df["Election Set"].map(dict(zip(min_cand_black_W2["Election Set"], min_cand_black_W2[district]))) 
district_df["Min Pref Min Hisp W2"] = district_df["Election Set"].map(dict(zip(min_cand_hisp_W2["Election Set"], min_cand_hisp_W2[district])))
district_df["Min Pref Min Neither W2"] = district_df["Election Set"].map(dict(zip(min_cand_neither_W2["Election Set"], min_cand_neither_W2[district])))
district_df["Black Conf W3"] =district_df["Election Set"].map(dict(zip(black_conf_W3["Election Set"], black_conf_W3[district])))   
district_df["Hisp Conf W3"] = district_df["Election Set"].map(dict(zip(hisp_conf_W3["Election Set"], hisp_conf_W3[district])))
district_df["Neither Conf W3"] = district_df["Election Set"].map(dict(zip(neither_conf_W3["Election Set"], neither_conf_W3[district])))
district_df["Black elec weight"] = district_df["Election Set"].map(dict(zip(black_weight_df["Election Set"], black_weight_df[district])))
district_df["Hisp elec weight"] = district_df["Election Set"].map(dict(zip(hisp_weight_df["Election Set"], hisp_weight_df[district])))
district_df["Neither elec weight"] = district_df["Election Set"].map(dict(zip(neither_weight_df["Election Set"], neither_weight_df[district])))
district_df["Black points accrued"] = district_df["Election Set"].map(dict(zip(black_points_accrued["Election Set"], black_points_accrued[district])))
district_df["Hisp points accrued"] =  district_df["Election Set"].map(dict(zip(hisp_points_accrued["Election Set"], hisp_points_accrued[district])))
district_df["Neither points accrued"] =  district_df["Election Set"].map(dict(zip(neither_points_accrued["Election Set"], neither_points_accrued[district])))

ratio_df = pd.DataFrame(columns = list(dist_perc_df.columns))
ratio_df.loc[0] =list(dist_perc_df.iloc[district]) 

writer = pd.ExcelWriter(DIR + 'outputs/District {} TX model, {} mode.xlsx'.format(district+1, model_mode), engine = 'xlsxwriter')
district_df.to_excel(writer, sheet_name = "All")
ratio_df.to_excel(writer, sheet_name = 'Ratios')

writer.save()


