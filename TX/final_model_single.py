# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 12:19:57 2020

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
import operator
import time
import heapq
import statsmodels.formula.api as smf
import statsmodels.api as sm
import scipy
from scipy import stats
import sys
from functools import partial
from run_functions import f, norm_dist_params, ER_run, preferred_cand, compute_final_dist, compute_W2
from ast import literal_eval
#############################################################################################################
#DATA PREP AND INPUTS:
#columns from GeoDF for processing
tot_pop = 'TOTPOP_x'
white_pop = 'NH_WHITE'
CVAP = "1_2018"
WCVAP = "7_2018"
HCVAP = "13_2018"
BCVAP = "5_2018" #with new CVAP codes!
geo_id = 'CNTYVTD'
county_split_id = "CNTY_x"
PRES16_DEM = 'ClintonD_16G_President'
PRES16_REP = 'TrumpR_16G_President'
PRES12_DEM = 'ObamaD_12G_President' 
PRES12_REP = 'RomneyR_12G_President'
SEN18_DEM = "ORourkeD_18G_U.S. Sen"
SEN18_REP = 'CruzR_18G_U.S. Sen'
C_X = "C_X"
C_Y = "C_Y"

#run parameters
start_time_total = time.time()
pop_tol = .005 #U.S. Cong
assignment1= 'sldl358' #CD, sldl358, sldu172, sldl309

#fixed parameters
num_districts = 150 #150 state house, 31 senate, 36 Cong
cand_drop_thresh = 0

plot_path = 'tx-results-cvap-sl-adjoined/tx-results-cvap-sl-adjoined.shp'  #for shapefile
DIR = ''

#read files
elec_data = pd.read_csv("TX_elections.csv")
elections = list(elec_data["Election"]) 
elec_type = elec_data["Type"]
election_returns = pd.read_csv("TX_statewide_election_returns.csv")
dropped_elecs = pd.read_csv("dropped_elecs.csv")["Dropped Elections"]
elec_cand_list = list(election_returns.columns)[2:] 
recency_weights = pd.read_csv("recency_weights.csv")
min_cand_weights = pd.read_csv("min_pref_weight_binary.csv")
cand_race_table = pd.read_csv("CandidateRace.csv")
EI_statewide = pd.read_csv("EI_statewide_data.csv")

#set up elections data structures
elecs_bool = ~elec_data.Election.isin(list(dropped_elecs))
elec_data_trunc = elec_data[elecs_bool].reset_index(drop = True)
elec_sets = list(set(elec_data_trunc["Election Set"]))
elections = list(elec_data_trunc["Election"])
general_elecs = list(elec_data_trunc[elec_data_trunc["Type"] == 'General'].Election)
primary_elecs = list(elec_data_trunc[elec_data_trunc["Type"] == 'Primary'].Election)
runoff_elecs = list(elec_data_trunc[elec_data_trunc["Type"] == 'Runoff'].Election)
#this dictionary matches a specific election with the set it belongs to
elec_set_dict = {}
for elec_set in elec_sets:
    elec_set_df = elec_data_trunc[elec_data_trunc["Election Set"] == elec_set]
    elec_set_dict[elec_set] = dict(zip(elec_set_df.Type, elec_set_df.Election))
elec_match_dict = dict(zip(elec_data_trunc["Election"], elec_data_trunc["Election Set"]))

#initialize state_gdf
#reformat/re-index enacted map plans
state_gdf = gpd.read_file(plot_path)
state_gdf["CD"] = state_gdf["CD"].astype('int')
state_gdf["sldu172"] = state_gdf["sldu172"] - 1
state_gdf["sldl358"] = state_gdf["sldl358"] - 1
state_gdf["sldl309"] = state_gdf["sldl309"] - 1
state_gdf.columns = state_gdf.columns.str.replace("-", "_")


#replace cut-off candidate names from shapefile with full names
election_return_cols = list(election_returns.columns)
cand1_index = election_return_cols.index('RomneyR_12G_President') #first
cand2_index = election_return_cols.index('ObamaD_12P_President') #last
elec_results_trunc = election_return_cols[cand1_index:cand2_index+1]
state_gdf_cols = list(state_gdf.columns)
cand1_index = state_gdf_cols.index('RomneyR_12')
cand2_index = state_gdf_cols.index('ObamaD_12P')
state_gdf_cols[cand1_index:cand2_index+1] = elec_results_trunc
state_gdf.columns = state_gdf_cols

state_df = pd.DataFrame(state_gdf)
state_df = state_df.drop(['geometry'], axis = 1)

##build graph from geo_dataframe
#graph = Graph.from_geodataframe(state_gdf)
#graph.add_data(state_gdf)
#centroids = state_gdf.centroid
#c_x = centroids.x
#c_y = centroids.y
#for node in graph.nodes():
#    graph.nodes[node]["C_X"] = c_x[node]
#    graph.nodes[node]["C_Y"] = c_y[node]
#
#CVAP in ER regressions will correspond to year
#this dictionary matches year and CVAP type to relevant data column 
cvap_types = ['CVAP', 'WCVAP', 'BCVAP', 'HCVAP']
cvap_codes = ['1', '7', '5', '13'] 
cvap_key = dict(zip(cvap_types,cvap_codes ))
cvap_years = [2012, 2014, 2016, 2018]
cvap_columns = {year:  {t: cvap_key[t] + '_' + str(year) for t in cvap_types} for year in cvap_years}

#make dictionary that maps an election to its candidates
#only include 2 major party candidates in generals
#only include candidates with > cand_drop_thresh of statewide vote share
candidates = {}
for elec in elections:
    #get rid of republican candidates in primaries or runoffs (primary runoffs)
    cands = [y for y in elec_cand_list if elec in y and "R_" not in re.sub(elec, '', y) ] if \
    "R_" in elec or "P_" in elec else [y for y in elec_cand_list if elec in y] 
    
    elec_year = elec_data_trunc.loc[elec_data_trunc["Election"] == elec, 'Year'].values[0]          
    if elec in general_elecs:
        #assumes D and R are always first two candidates
        cands = cands[:2]
    if elec not in general_elecs:
       pattern = '|'.join(cands)
       elec_df = state_df.copy().loc[:, state_df.columns.str.contains(pattern)]
       elec_df["Total"] = elec_df.sum(axis=1)
       remove_list = []
       for cand in cands:
           if sum(elec_df["{}".format(cand)])/sum(elec_df["Total"]) < cand_drop_thresh:
               remove_list.append(cand)  
       cands = [i for i in cands if i not in remove_list]

    #to prep for ER regressions, add cand-share-of-precinct-CVAP (cap at 1 if data error)
    for cand in cands:
        state_df["{}%CVAP".format(cand)] = state_df["{}".format(cand)]/state_df[cvap_columns[elec_year]['CVAP']]    
        state_df["{}%CVAP".format(cand)] = np.minimum(1,state_df["{}%CVAP".format(cand)])
    candidates[elec] = dict(zip(list(range(len(cands))), cands))

cand_race_dict = cand_race_table.set_index("Candidates").to_dict()["Race"]
min_cand_weights_dict = {key:min_cand_weights.to_dict()[key][0] for key in  min_cand_weights.to_dict().keys()}     

#precompute election recency weights and statewide EI for statewide/district mode
#map data storage: set up all dataframes to be filled   
black_pref_cands_prim_state = pd.DataFrame(columns = range(num_districts))
black_pref_cands_prim_state["Election Set"] = elec_sets
hisp_pref_cands_prim_state = pd.DataFrame(columns = range(num_districts))
hisp_pref_cands_prim_state["Election Set"] = elec_sets
#store runoff preferences for instances where min-pref candidate needs to switch btwn prim and runoff
black_pref_cands_runoffs_state = pd.DataFrame(columns = range(num_districts))
black_pref_cands_runoffs_state["Election Set"] = elec_sets
hisp_pref_cands_runoffs_state = pd.DataFrame(columns = range(num_districts))
hisp_pref_cands_runoffs_state["Election Set"] = elec_sets 
recency_W1 = pd.DataFrame(columns = range(num_districts))
recency_W1["Election Set"] = elec_sets
black_conf_W3_state = pd.DataFrame(columns = range(num_districts))
black_conf_W3_state["Election Set"] = elec_sets
hisp_conf_W3_state= pd.DataFrame(columns = range(num_districts))
hisp_conf_W3_state["Election Set"] = elec_sets 

#pre-compute recency_W1 df for all model modes, and W3, W2 dfs for statewide/equal modes    
for elec_set in elec_sets:
        elec_year = elec_data_trunc.loc[elec_data_trunc["Election Set"] == elec_set, 'Year'].values[0].astype(str)
        for dist in range(num_districts):
            recency_W1.at[recency_W1["Election Set"] == elec_set, dist] = recency_weights[elec_year][0]
   
# pref cands needed for statewide and equal, weights only needed for state 
for elec in primary_elecs + runoff_elecs:
    for district in range(num_districts):
        if elec in primary_elecs:
            black_pref_cand = EI_statewide.loc[EI_statewide["Election"] == elec, "Black Pref Cand"].values[0]
            hisp_pref_cand = EI_statewide.loc[EI_statewide["Election"] == elec, "Latino Pref Cand"].values[0]
            black_ei_conf = EI_statewide.loc[EI_statewide["Election"] == elec, "Black Confidence"].values[0]
            hisp_ei_conf = EI_statewide.loc[EI_statewide["Election"] == elec, "Latino Confidence"].values[0]               
            
            black_pref_cands_prim_state.at[black_pref_cands_prim_state["Election Set"] == elec_match_dict[elec], district] = black_pref_cand
            black_conf_W3_state.at[black_conf_W3_state["Election Set"] == elec_match_dict[elec], district] = black_ei_conf
            hisp_pref_cands_prim_state.at[hisp_pref_cands_prim_state["Election Set"] == elec_match_dict[elec], district] = hisp_pref_cand
            hisp_conf_W3_state.at[hisp_conf_W3_state["Election Set"] == elec_match_dict[elec], district] = hisp_ei_conf                                             
        else:
            black_pref_cand = EI_statewide.loc[EI_statewide["Election"] == elec, "Black Pref Cand"].values[0]
            hisp_pref_cand = EI_statewide.loc[EI_statewide["Election"] == elec, "Latino Pref Cand"].values[0]
                        
            black_pref_cands_runoffs_state.at[black_pref_cands_runoffs_state["Election Set"] == elec_match_dict[elec], district] = black_pref_cand       
            hisp_pref_cands_runoffs_state.at[hisp_pref_cands_runoffs_state["Election Set"] == elec_match_dict[elec], district] = hisp_pref_cand
           
#neither computation is based on results of black and latino computations
neither_conf_W3_state = black_conf_W3_state.drop(["Election Set"], axis = 1)*hisp_conf_W3_state.drop(["Election Set"], axis =1)
neither_conf_W3_state["Election Set"] = elec_sets
   
min_cand_black_W2_state, min_cand_hisp_W2_state, min_cand_neither_W2_state = compute_W2(elec_sets, \
              range(num_districts), min_cand_weights_dict, black_pref_cands_prim_state, hisp_pref_cands_prim_state, cand_race_dict)

#compute final election weights by taking product of weights 1,2, and 3 for each election set and district
#Note: because these are statewide weights, and election set will have the same weight across districts
black_weight_state = recency_W1.drop(["Election Set"], axis=1)*min_cand_black_W2_state.drop(["Election Set"], axis=1)*black_conf_W3_state.drop(["Election Set"], axis=1)
black_weight_state["Election Set"] = elec_sets
hisp_weight_state = recency_W1.drop(["Election Set"], axis=1)*min_cand_hisp_W2_state.drop(["Election Set"], axis=1)*hisp_conf_W3_state.drop(["Election Set"], axis=1)    
hisp_weight_state["Election Set"] = elec_sets
neither_weight_state = recency_W1.drop(["Election Set"], axis=1)*min_cand_neither_W2_state.drop(["Election Set"], axis=1)*neither_conf_W3_state.drop(["Election Set"], axis=1)    
neither_weight_state["Election Set"] = elec_sets


black_weight_equal = pd.DataFrame(columns = range(num_districts))
black_weight_equal[0] = [1]*len(elec_sets)
hisp_weight_equal = pd.DataFrame(columns = range(num_districts))
hisp_weight_equal[0] = [1]*len(elec_sets)
neither_weight_equal = pd.DataFrame(columns = range(num_districts))
neither_weight_equal[0] = [1]*len(elec_sets)
for i in range(1, num_districts):
    black_weight_equal[i] = 1
    hisp_weight_equal[i] = 1
    neither_weight_equal[i] = 1
black_weight_equal["Election Set"] = elec_sets
hisp_weight_equal["Election Set"] = elec_sets
neither_weight_equal["Election Set"] = elec_sets

my_updaters = {
        "population": updaters.Tally(tot_pop, alias = "population")
      }

election_functions = [Election(j, candidates[j]) for j in elections]
election_updaters = {election.name: election for election in election_functions}
my_updaters.update(election_updaters)

partition = GeographicPartition(graph = graph, assignment = assignment1, updaters = my_updaters)
#key: election, value: dictionary with keys = dists and values are dicts of % for each cand
#for particular elec and dist can access all cand results by: dist_elec_results[elec][dist]
dist_elec_results = {}
order = [x for x in partition.parts]
for elec in elections:    
    cands = candidates[elec]
    dist_elec_results[elec] = {}
    outcome_list = [dict(zip(order, partition[elec].percents(cand))) for cand in cands.keys()]      
    dist_elec_results[elec] = {d: {cands[i]: outcome_list[i][d] for i in cands.keys()} for d in range(num_districts)}
       
#compute winners of each election in each district and store
    #winners df:
map_winners = pd.DataFrame(columns = range(num_districts))
map_winners["Election"] = elections
map_winners["Election Set"] = elec_data_trunc["Election Set"]
map_winners["Election Type"] = elec_data_trunc["Type"]
for i in range(num_districts):
    map_winners[i] = [max(dist_elec_results[elec][i].items(), key=operator.itemgetter(1))[0] for elec in elections]

black_pref_cands_prim_dist = pd.DataFrame(columns = range(num_districts))
black_pref_cands_prim_dist["Election Set"] = elec_sets
hisp_pref_cands_prim_dist = pd.DataFrame(columns = range(num_districts))
hisp_pref_cands_prim_dist["Election Set"] = elec_sets
#store runoff preferences for instances where min-pref candidate needs to switch btwn prim and runoff
black_pref_cands_runoffs_dist = pd.DataFrame(columns = range(num_districts))
black_pref_cands_runoffs_dist["Election Set"] = elec_sets
hisp_pref_cands_runoffs_dist = pd.DataFrame(columns = range(num_districts))
hisp_pref_cands_runoffs_dist["Election Set"] = elec_sets 
black_conf_W3_dist = pd.DataFrame(columns = range(num_districts))
black_conf_W3_dist["Election Set"] = elec_sets
hisp_conf_W3_dist = pd.DataFrame(columns = range(num_districts))
hisp_conf_W3_dist["Election Set"] = elec_sets  
    ##########################################################################################
#pre-compute recency_W1 df for all model modes, and W3, W2 dfs for statewide/equal modes
#to compute district weights, preferred candidate and confidence is computed
    #for each district at every ReCom step
for district in range(num_districts): #get vector of precinct values for each district                  
    #only need preferred candidates and condidence in primary and runoffs
    #(in Generals we only care if the Democrat wins)
    for elec in primary_elecs + runoff_elecs: 
        elec_year = elec_data_trunc.loc[elec_data_trunc["Election"] == elec, 'Year'].values[0]
        
        #demographic data for regression pulled from that election's year
        cvap = cvap_columns[elec_year]['CVAP']
        black_cvap = cvap_columns[elec_year]['BCVAP']
        hisp_cvap = cvap_columns[elec_year]['HCVAP']
        
        cols_of_interest = [cvap, black_cvap, hisp_cvap] + ["{}%CVAP".format(i) for i in candidates[elec].values()]
        dist_df = state_df[state_df.index.isin(partition.parts[district])][cols_of_interest]
        dist_df = dist_df[dist_df[cvap] > 0] #drop rows with that year's cvap = 0
        
        dist_df['black share'] = list(dist_df[black_cvap]/dist_df[cvap])
        dist_df['hisp share'] = list(dist_df[hisp_cvap]/dist_df[cvap])
               
        #run ER regressions for black and Latino voters
        #determine black and Latino preferred candidates and confidence preferred-cand is correct
        #we run Weighted Linear Regression, weighted by precinct CVAP
        pop_weights = list(dist_df[cvap]/sum(dist_df[cvap]))          
        if elec == '16P_President' and district == 49:
            dist_df2 = dist_df.copy()
            dist_df2["pop weights"] = pop_weights
            dist_df2.to_csv("dist_df_test.csv")
        #double equation method means we run race share of CVAP on x-axis and 
        #candidate vote share of CVAP on y-axis
        black_norm_params = {}
        hisp_norm_params = {}
        for cand in candidates[elec].values():
            cand_cvap_share = list(dist_df["{}%CVAP".format(cand)])
                      
        #regrss cand share of total vote on demo-share-CVAP, black and latino voters  
            point_est, std = ER_run(cand,elec, district, dist_df[['black share', 'hisp share']], \
                            cand_cvap_share, pop_weights)
                                                                                
            black_norm_params[cand] = [point_est[1], std]
            hisp_norm_params[cand] = [point_est[0], std]

        #computing preferred candidate and confidence in that choice gives is weight 3
        black_pref_cand, black_er_conf = preferred_cand(district, elec, black_norm_params, 49, '16P_President', "black", True )
        hisp_pref_cand, hisp_er_conf = preferred_cand(district, elec, hisp_norm_params, 49, '16P_President', "hisp", True)
        if elec in primary_elecs:
            black_pref_cands_prim_dist.at[black_pref_cands_prim_dist["Election Set"] == elec_match_dict[elec], district] = black_pref_cand
            black_conf_W3_dist.at[black_conf_W3_dist["Election Set"] == elec_match_dict[elec], district] = black_er_conf
            hisp_pref_cands_prim_dist.at[hisp_pref_cands_prim_dist["Election Set"] == elec_match_dict[elec], district] = hisp_pref_cand
            hisp_conf_W3_dist.at[hisp_conf_W3_dist["Election Set"] == elec_match_dict[elec], district] = hisp_er_conf         
        else:
            black_pref_cands_runoffs_dist.at[black_pref_cands_runoffs_dist["Election Set"] == elec_match_dict[elec], district] = black_pref_cand
            hisp_pref_cands_runoffs_dist.at[hisp_pref_cands_runoffs_dist["Election Set"] == elec_match_dict[elec], district] = hisp_pref_cand
              
################################################################################
#get election weight 2 (minority preferred minority_ and combine for final            
min_cand_black_W2_dist, min_cand_hisp_W2_dist, min_cand_neither_W2_dist = compute_W2(elec_sets, \
      range(num_districts), min_cand_weights_dict, black_pref_cands_prim_dist, hisp_pref_cands_prim_dist, cand_race_dict)
#combine for final black and Latino election weights (now district-dependent)
neither_conf_W3_dist = black_conf_W3_dist.drop(["Election Set"], axis =1)*hisp_conf_W3_dist.drop(["Election Set"], axis =1)
neither_conf_W3_dist["Election Set"] = elec_sets

black_weight_dist = recency_W1.drop(["Election Set"], axis=1)*min_cand_black_W2_dist.drop(["Election Set"], axis=1)*black_conf_W3_dist.drop(["Election Set"], axis=1)
black_weight_dist["Election Set"] = elec_sets
hisp_weight_dist = recency_W1.drop(["Election Set"], axis=1)*min_cand_hisp_W2_dist.drop(["Election Set"], axis=1)*hisp_conf_W3_dist.drop(["Election Set"], axis=1)    
hisp_weight_dist["Election Set"] = elec_sets
neither_weight_dist = recency_W1.drop(["Election Set"], axis=1)*min_cand_neither_W2_dist.drop(["Election Set"], axis=1)*neither_conf_W3_dist.drop(["Election Set"], axis=1)    
neither_weight_dist["Election Set"] = elec_sets
  ###################################################################    
#district probability distribution: state
final_state_prob_dict, black_pref_wins_state, hisp_pref_wins_state, neither_pref_wins_state,\
             black_points_accrued_state, hisp_points_accrued_state, \
             neither_points_accrued_state, primary_second_df_state \
             = compute_final_dist(map_winners, black_pref_cands_prim_state, black_pref_cands_runoffs_state,\
             hisp_pref_cands_prim_state, hisp_pref_cands_runoffs_state, neither_weight_state, \
             black_weight_state, hisp_weight_state, dist_elec_results, range(num_districts),
             cand_race_table, num_districts, candidates, elec_sets, elec_set_dict, True)

#district probability distribution: equal
final_equal_prob_dict, black_pref_wins_equal, hisp_pref_wins_equal, neither_pref_wins_equal,\
         black_points_accrued_equal, hisp_points_accrued_equal, \
         neither_points_accrued_equal, primary_second_df_equal \
         = compute_final_dist(map_winners, black_pref_cands_prim_state, black_pref_cands_runoffs_state,\
         hisp_pref_cands_prim_state, hisp_pref_cands_runoffs_state, neither_weight_equal, \
         black_weight_equal, hisp_weight_equal, dist_elec_results, range(num_districts),
         cand_race_table, num_districts, candidates, elec_sets, elec_set_dict, True)

#district probability distribution: district
final_dist_prob_dict, black_pref_wins_dist, hisp_pref_wins_dist, neither_pref_wins_dist,\
         black_points_accrued_dist, hisp_points_accrued_dist, \
         neither_points_accrued_dist, primary_second_df_dist\
         = compute_final_dist(map_winners, black_pref_cands_prim_dist, black_pref_cands_runoffs_dist,\
         hisp_pref_cands_prim_dist, hisp_pref_cands_runoffs_dist, neither_weight_dist, \
         black_weight_dist, hisp_weight_dist, dist_elec_results, range(num_districts),
         cand_race_table, num_districts, candidates, elec_sets, elec_set_dict, True)
################################################## 

#district deep dives
dist_tests = [50, 51] #reg iindex
model_modes = ['district', 'statewide', 'equal']

for dist in dist_tests:
    writer = pd.ExcelWriter(DIR + 'outputs/District {}, map {} analysis.xlsx'.format(dist, assignment1), engine = 'xlsxwriter')
    for model_mode in model_modes:
        primary_winners = map_winners[map_winners["Election Type"] == 'Primary'].reset_index(drop = True)
        runoff_winners = map_winners[map_winners["Election Type"] == 'Runoff'].reset_index(drop = True)
        general_winners = map_winners[map_winners["Election Type"] == 'General'].reset_index(drop = True)
        
        primary_races = [elec_set_dict[elec_set]["Primary"] for elec_set in elec_sets]
        runoff_races = [None if 'Runoff' not in elec_set_dict[elec_set].keys() else elec_set_dict[elec_set]["Runoff"] for elec_set in elec_sets]
        cand_party_dict = cand_race_table.set_index("Candidates").to_dict()["Party"]
        black_pref_cands_prim = black_pref_cands_prim_dist if model_mode == 'district' else black_pref_cands_prim_state
        hisp_pref_cands_prim = hisp_pref_cands_prim_dist if model_mode == 'district' else hisp_pref_cands_prim_state
        black_pref_cands_runoffs = black_pref_cands_runoffs_dist if model_mode == 'district' else black_pref_cands_runoffs_state
        hisp_pref_cands_runoffs = hisp_pref_cands_runoffs_dist if model_mode == 'district' else hisp_pref_cands_runoffs_state
        
        min_cand_black_W2 = min_cand_black_W2_dist if model_mode == 'district' else min_cand_black_W2_state
        min_cand_hisp_W2 =  min_cand_hisp_W2_dist if model_mode == 'district' else min_cand_hisp_W2_state
        min_cand_neither_W2 =  min_cand_neither_W2_dist if model_mode == 'district' else min_cand_neither_W2_state
        
        black_conf_W3 = black_conf_W3_dist if model_mode == 'district' else black_conf_W3_state
        hisp_conf_W3 = hisp_conf_W3_dist if model_mode == 'district' else hisp_conf_W3_state
        neither_conf_W3 = neither_conf_W3_dist if model_mode == 'district' else neither_conf_W3_state
        
        black_weight_df = black_weight_dist if model_mode == 'district' else black_weight_state if model_mode == 'statewide' else black_weight_equal
        hisp_weight_df = hisp_weight_dist if model_mode == 'district' else hisp_weight_state if model_mode == 'statewide' else hisp_weight_equal
        neither_weight_df = neither_weight_dist if model_mode == 'district' else neither_weight_state if model_mode == 'statewide' else neither_weight_equal
        
        black_pref_wins = black_pref_wins_dist if model_mode == 'district' else black_pref_wins_state if model_mode == 'statewide' else black_pref_wins_equal
        hisp_pref_wins = hisp_pref_wins_dist if model_mode == 'district' else hisp_pref_wins_state if model_mode == 'statewide' else hisp_pref_wins_equal
        neither_pref_wins = neither_pref_wins_dist if model_mode == 'district' else neither_pref_wins_state if model_mode == 'statewide' else neither_pref_wins_equal
        
        primary_second_df = primary_second_df_dist if model_mode == 'district' else primary_second_df_state if model_mode == 'statewide' \
                            else primary_second_df_equal
                            
        black_points_accrued = black_points_accrued_dist if model_mode == 'district' else black_points_accrued_state if model_mode == 'statewide' \
                            else black_points_accrued_equal
        hisp_points_accrued = hisp_points_accrued_dist if model_mode == 'district' else hisp_points_accrued_state if model_mode == 'statewide' \
                            else hisp_points_accrued_equal
        neither_points_accrued = neither_points_accrued_dist if model_mode == 'district' else neither_points_accrued_state if model_mode == 'statewide' \
                            else neither_points_accrued_equal
        
        final_prob = final_dist_prob_dict if model_mode == 'district' else final_state_prob_dict if model_mode == 'statewide' \
                    else final_equal_prob_dict
                            
        district_df = pd.DataFrame(columns = ["Election Set"])
        district = dist-1
        district_df["Election Set"] = elec_sets
        district_df["Primary Winner"] = district_df["Election Set"].map(dict(zip(primary_winners["Election Set"], primary_winners[district])))
        district_df["Primary Second Place"] = district_df["Election Set"].map(dict(zip(primary_second_df["Election Set"], primary_second_df[district])))
        district_df["Runoff Winner"] = district_df["Election Set"].map(dict(zip(runoff_winners["Election Set"], runoff_winners[district])))
        district_df["General Winner"] = district_df["Election Set"].map(dict(zip(general_winners["Election Set"], general_winners[district])))
        district_df["Black pref cand (primary)"] = district_df["Election Set"].map(dict(zip(black_pref_cands_prim["Election Set"], black_pref_cands_prim[district])))
        district_df["Hisp pref cand (primary)"] = district_df["Election Set"].map(dict(zip(hisp_pref_cands_prim["Election Set"], hisp_pref_cands_prim[district])))
        district_df["Black pref cand (runoff)"] = district_df["Election Set"].map(dict(zip(black_pref_cands_runoffs["Election Set"], black_pref_cands_runoffs[district])))
        district_df["Hisp pref cand (runoff)"] = district_df["Election Set"].map(dict(zip(hisp_pref_cands_runoffs["Election Set"], hisp_pref_cands_runoffs[district])))
        district_df["Black primary cand top 2"] = [(district_df["Black pref cand (primary)"][i] == district_df["Primary Winner"][i])\
                                            or  (district_df["Black pref cand (primary)"][i] == district_df["Primary Second Place"][i]) for i in range(len(district_df))]
        district_df["Hisp primary cand top 2"] = [(district_df["Hisp pref cand (primary)"][i] == district_df["Primary Winner"][i])\
                                             or  (district_df["Hisp pref cand (primary)"][i] == district_df["Primary Second Place"][i]) for i in range(len(district_df))]
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
        district_df["Final Distribution"] = [final_prob[district]] + [None]*(len(district_df) - 1)
        district_df.to_excel(writer, sheet_name = "District {}, {} Model".format(dist, model_mode), index = False)

    writer.save()



