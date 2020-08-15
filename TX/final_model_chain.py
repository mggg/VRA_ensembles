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
total_steps = 0
pop_tol = .005 #U.S. Cong
assignment1= 'CD' #CD, sldl358, sldu172, sldl309
run_name = 'sldl358' #sys.argv[1]
run_type = 'free' #sys.argv[3] #(free, hill_climb, sim_anneal)
min_group = 'hisp' #sys.argv[4] # (black, hisp, both)
model_mode = 'equal' #sys.argv[2] #'district', equal, statewide
start_map = 'enacted' #sys.argv[5] #'enacted' or random
#additional parameters for opimization runs:
#need if running hillclimb with bound
bound = .1 #float(sys.argv[6]) 
#need if simAnneal run (with cycles)
cycle_length = 2000 #float(sys.argv[7])
start_cool = 500 #float(sys.argv[8])
stop_cool = 1500 #float(sys.argv[9])
effective_thresh = .5 #float(sys.argv[10])

#fixed parameters
num_districts = 36 #150 state house, 31 senate, 36 Cong
degrandy_hisp = 11 #10.39 rounded up
degrandy_black = 5 #4.69 rounded up
cand_drop_thresh = 0

plot_path = 'tx-results-cvap-sl-adjoined/tx-results-cvap-sl-adjoined.shp'  #for shapefile
store_interval = 5 #how many steps until storage
stuck_length = 1000 #steps at same score until break
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
hisp_conf_W3_state = pd.DataFrame(columns = range(num_districts))
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


############################################################################################################       
#FUNCTIONS FOR CHAIN
#elections model function. Takes in partition and returns effectiveness distribution per district
    #and total black effective and Latino-effective districts (>50% effective)
def final_elec_model(partition):  
    #only need to run model on two ReCom districts that have changed
    if partition.parent is not None:
        dict1 = dict(partition.parent.assignment)
        dict2 = dict(partition.assignment)
        differences = set([dict1[k] for k in dict1.keys() if dict1[k] != dict2[k]]).union(set([dict2[k] for k in dict2.keys() if dict1[k] != dict2[k]]))
        
    dist_changes = range(num_districts) if partition.parent is None else differences
   
    #The output of the elections model is a probability distribution for each district:
    #% Latino, Black, Neither or Overlap effetive
    #To compute these, each election set is first weighted (different for Black and Latino)
    #by multiplying a recency, minority-preferred and confidence weight.
    #If the black (Latino) preferred candidate wins the election (set) a number of points equal to
    #the set's weight is accrued. The ratio of black-accrued points to total possible points
    #is the raw black (latino) effectiviness for the district. The FINAL black, latino, overlap
    #and neither are computed from there.
    #We need to track several entities, which will be dataframes, whose columns are districts and
    #rows are election sets (or sometimes individual elections)
    #These dataframes each store one of the following: black (latino) preferred candidates (in the
    #election set's primary), black (latino) preferred candidate in runoff, winners of primary,
    #runoff and general elections, recency weights (weight 1), minority-preferred-minority weihgt (weight 2),
    #confidence weight (weight 3) and final election weight for both black and latino voters
      
    #dictionary to store district level vote share results in each election for each candidate
    #key: election, value: dictionary whose keys are districtss and values are 
    #another dictionary of % vote share for each cand
    #for particular elec and dist can access all cand results by: dist_elec_results[elec][dist]
    dist_elec_results = {}
    order = [x for x in partition.parts]
    for elec in elections:
        cands = candidates[elec]
        dist_elec_results[elec] = {}
        outcome_list = [dict(zip(order, partition[elec].percents(cand))) for cand in cands.keys()]      
        dist_elec_results[elec] = {d: {cands[i]: outcome_list[i][d] for i in cands.keys()} for d in range(num_districts)}
    ##########################################################################################   
    #compute winners of each election in each district and store
    #winners df:
    map_winners = pd.DataFrame(columns = dist_changes)
    map_winners["Election"] = elections
    map_winners["Election Set"] = elec_data_trunc["Election Set"]
    map_winners["Election Type"] = elec_data_trunc["Type"]
    for i in dist_changes:
        map_winners[i] = [max(dist_elec_results[elec][i].items(), key=operator.itemgetter(1))[0] for elec in elections]

    black_pref_cands_prim_dist = pd.DataFrame(columns = dist_changes)
    black_pref_cands_prim_dist["Election Set"] = elec_sets
    hisp_pref_cands_prim_dist = pd.DataFrame(columns = dist_changes)
    hisp_pref_cands_prim_dist["Election Set"] = elec_sets
    #store runoff preferences for instances where min-pref candidate needs to switch btwn prim and runoff
    black_pref_cands_runoffs_dist = pd.DataFrame(columns = dist_changes)
    black_pref_cands_runoffs_dist["Election Set"] = elec_sets
    hisp_pref_cands_runoffs_dist = pd.DataFrame(columns = dist_changes)
    hisp_pref_cands_runoffs_dist["Election Set"] = elec_sets 
    black_conf_W3_dist = pd.DataFrame(columns = dist_changes)
    black_conf_W3_dist["Election Set"] = elec_sets
    hisp_conf_W3_dist = pd.DataFrame(columns = dist_changes)
    hisp_conf_W3_dist["Election Set"] = elec_sets  
    
    #to compute district weights, preferred candidate and confidence is computed
    #for each district at every ReCom step
    for district in dist_changes: #get vector of precinct values for each district                  
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
            
            black_share = list(dist_df[black_cvap]/dist_df[cvap])
            hisp_share = list(dist_df[hisp_cvap]/dist_df[cvap])
                   
            #run ER regressions for black and Latino voters
            #determine black and Latino preferred candidates and confidence preferred-cand is correct
            #we run Weighted Linear Regression, weighted by precinct CVAP
            pop_weights = list(dist_df[cvap]/sum(dist_df[cvap]))          
            
            #double equation method means we run race share of CVAP on x-axis and 
            #candidate vote share of CVAP on y-axis
            black_norm_params = {}
            hisp_norm_params = {}
            for cand in candidates[elec].values():
                cand_cvap_share = list(dist_df["{}%CVAP".format(cand)])
                          
            #regrss cand share of total vote on demo-share-CVAP, black and latino voters                                                                                  
                mean, std = ER_run(cand,elec, district, black_share, cand_cvap_share,\
                       pop_weights, black_norm_params)
                black_norm_params[cand] = [mean, std]
       
                mean, std = ER_run(cand,elec, district, hisp_share, cand_cvap_share,\
                       pop_weights, hisp_norm_params)
                hisp_norm_params[cand] = [mean, std]

            #computing preferred candidate and confidence in that choice gives is weight 3
            black_pref_cand, black_er_conf = preferred_cand(district, elec, black_norm_params)
            hisp_pref_cand, hisp_er_conf = preferred_cand(district, elec, hisp_norm_params)
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
          dist_changes, min_cand_weights_dict, black_pref_cands_prim_dist, hisp_pref_cands_prim_dist, cand_race_dict)
    #combine for final black and Latino election weights (now district-dependent)
    neither_conf_W3_dist = black_conf_W3_dist.drop(["Election Set"], axis =1)*hisp_conf_W3_dist.drop(["Election Set"], axis =1)
    neither_conf_W3_dist["Election Set"] = elec_sets
    
    black_weight_dist = recency_W1.drop(["Election Set"], axis=1)*min_cand_black_W2_dist.drop(["Election Set"], axis=1)*black_conf_W3_dist.drop(["Election Set"], axis=1)
    black_weight_dist["Election Set"] = elec_sets
    hisp_weight_dist = recency_W1.drop(["Election Set"], axis=1)*min_cand_hisp_W2_dist.drop(["Election Set"], axis=1)*hisp_conf_W3_dist.drop(["Election Set"], axis=1)    
    hisp_weight_dist["Election Set"] = elec_sets
    neither_weight_dist = recency_W1.drop(["Election Set"], axis=1)*min_cand_neither_W2_dist.drop(["Election Set"], axis=1)*neither_conf_W3_dist.drop(["Election Set"], axis=1)    
    neither_weight_dist["Election Set"] = elec_sets
                                  
    #################################################################################  
    #district probability distribution: state
    final_state_prob_dict = compute_final_dist(map_winners, black_pref_cands_prim_state, black_pref_cands_runoffs_state,\
                 hisp_pref_cands_prim_state, hisp_pref_cands_runoffs_state, neither_weight_state, \
                 black_weight_state, hisp_weight_state, dist_elec_results, dist_changes,
                 cand_race_table, num_districts, candidates, elec_sets, elec_set_dict)
    
    #district probability distribution: equal
    final_equal_prob_dict = compute_final_dist(map_winners, black_pref_cands_prim_state, black_pref_cands_runoffs_state,\
             hisp_pref_cands_prim_state, hisp_pref_cands_runoffs_state, neither_weight_equal, \
             black_weight_equal, hisp_weight_equal, dist_elec_results, dist_changes,
             cand_race_table, num_districts, candidates, elec_sets, elec_set_dict)
    
    #district probability distribution: district
    final_dist_prob_dict = compute_final_dist(map_winners, black_pref_cands_prim_dist, black_pref_cands_runoffs_dist,\
             hisp_pref_cands_prim_dist, hisp_pref_cands_runoffs_dist, neither_weight_dist, \
             black_weight_dist, hisp_weight_dist, dist_elec_results, dist_changes,
             cand_race_table, num_districts, candidates, elec_sets, elec_set_dict)

    #new vector of probability distributions-by-district is the same as last ReCom step, except in 2 districts 
    if step_Num == 0:
         final_state_prob = {key:final_state_prob_dict[key] for key in sorted(final_state_prob_dict)}
         final_equal_prob = {key:final_equal_prob_dict[key] for key in sorted(final_equal_prob_dict)}
         final_dist_prob = {key:final_dist_prob_dict[key] for key in sorted(final_dist_prob_dict)}
    
     
    elif step_Num % store_interval == 0 and step_Num > 0 and run_type != 'free':
        final_state_prob = dict(final_state_prob_df.loc[store_interval-1])
        final_equal_prob = dict(final_equal_prob_df.loc[store_interval-1])
        final_dist_prob = dict(final_dist_prob_df.loc[store_interval-1])
        
        for i in final_state_prob_dict.keys():
             final_state_prob[i] = final_state_prob_dict[i]
             final_equal_prob[i] = final_equal_prob_dict[i]
             final_dist_prob[i] = final_dist_prob_dict[i]
    else:
        final_state_prob = dict(final_state_prob_df.loc[(step_Num % store_interval)-1])
        final_equal_prob = dict(final_equal_prob_df.loc[(step_Num % store_interval)-1])
        final_dist_prob = dict(final_dist_prob_df.loc[(step_Num % store_interval)-1])

        for i in final_state_prob_dict.keys():
             final_state_prob[i] = final_state_prob_dict[i]
             final_equal_prob[i] = final_equal_prob_dict[i]
             final_dist_prob[i] = final_dist_prob_dict[i]
    
    optimize_dict = final_state_prob if model_mode == 'statewide' else final_equal_prob\
                    if model_mode == 'equal' else final_dist_prob
                        

    hisp_effective = [i+l for i,j,k,l in optimize_dict.values()]
    black_effective = [j+l for i,j,k,l in optimize_dict.values()]
    
    total_hisp_final = len([z for z in hisp_effective if z > effective_thresh])
    total_black_final = len([z for z in black_effective if z > effective_thresh])
     
    return final_state_prob_dict, final_equal_prob_dict, final_dist_prob_dict, \
            total_hisp_final, total_black_final
            
def num_cut_edges(partition):
    return len(partition["cut_edges"])

def demo_percents(partition): 
    hisp_pct = {k: partition["HCVAP"][k]/partition["CVAP"][k] for k in partition["HCVAP"].keys()}
    black_pct = {k: partition["BCVAP"][k]/partition["CVAP"][k] for k in partition["BCVAP"].keys()}
    white_pct = {k: partition["WCVAP"][k]/partition["CVAP"][k] for k in partition["WCVAP"].keys()}
    return hisp_pct, black_pct, white_pct

def centroids(partition):
    CXs = {k: partition["Sum_CX"][k]/len(partition.parts[k]) for k in list(partition.parts.keys())}
    CYs = {k: partition["Sum_CY"][k]/len(partition.parts[k]) for k in list(partition.parts.keys())}
    centroids = {k: (CXs[k], CYs[k]) for k in list(partition.parts.keys())}
    return centroids

def num_splits(partition, df = state_gdf):
    df["current"] = df.index.map(partition.assignment.to_dict())
    return sum(df.groupby(county_split_id)['current'].nunique() > 1)

def vra_score(partition):
    hisp_vra_dists = partition["final_elec_model"][3] 
    black_vra_dists = partition["final_elec_model"][4]    
    if min_group == 'black':
        return black_vra_dists
    if min_group == 'hisp':
        return hisp_vra_dists
    if min_group == 'both':
        return black_vra_dists + hisp_vra_dists

my_updaters = {
    "population": updaters.Tally(tot_pop, alias = "population"),
    "CVAP": updaters.Tally(CVAP, alias = "CVAP"),
    "WCVAP": updaters.Tally(WCVAP, alias = "WCVAP"),
    "HCVAP": updaters.Tally(HCVAP, alias = "HCVAP"),
    "BCVAP": updaters.Tally(BCVAP, alias = "BCVAP"),
    "PRES16_DEM": updaters.Tally(PRES16_DEM, alias = "PRES16_DEM"),
    "PRES16_REP": updaters.Tally(PRES16_REP, alias = "PRES16_REP"),
    "PRES12_DEM": updaters.Tally(PRES12_DEM, alias = "PRES12_DEM"),
    "PRES12_REP": updaters.Tally(PRES12_REP, alias = "PRES12_REP"),
    "SEN18_DEM": updaters.Tally(SEN18_DEM, alias = "SEN18_DEM"),
    "SEN18_REP": updaters.Tally(SEN18_REP, alias = "SEN18_REP"),
    "Sum_CX": updaters.Tally(C_X, alias = "Sum_CX"),
    "Sum_CY": updaters.Tally(C_Y, alias = "Sum_CY"),
    "cut_edges": cut_edges,
    "num_cut_edges": num_cut_edges,  
    "num_splits": num_splits,
    "demo_percents": demo_percents,
    "final_elec_model": final_elec_model,
    "vra_score": vra_score,
    "centroids": centroids
}


#updater functions
elections_track = [
    Election("PRES16", {"Democratic": 'ClintonD_16G_President' , "Republican": 'TrumpR_16G_President'}, alias = "PRES16"),
    Election("PRES12", {"Democratic": 'ObamaD_12G_President' , "Republican": 'RomneyR_12G_President'}, alias = "PRES12"),
    Election("SEN18", {"Democratic": "ORourkeD_18G_U.S. Sen" , "Republican": 'CruzR_18G_U.S. Sen'}, alias = "SEN18"),   
]

election_updaters = {election.name: election for election in elections_track}
my_updaters.update(election_updaters)

election_functions = [Election(j, candidates[j]) for j in elections]
election_updaters = {election.name: election for election in election_functions}
my_updaters.update(election_updaters)

#order = [x for x in initial_partition.parts]
#df_elec = pd.DataFrame(columns = ["district", "elec_results_Dem", "elec_results_Rep", 'dem div dem rep', 'rep div dem rep', 'updater dem', 'updater rep'])
#df_elec["district"] = list(range(36))
#df_elec["elec_results_Dem"] = [dist_elec_results['16G_President'][i]['ClintonD_16G_President'] for i in list(range(36))]
#df_elec["elec_results_Rep"] = [dist_elec_results['16G_President'][i]['TrumpR_16G_President'] for i in list(range(36))]
#df_elec["dem div dem rep"] = df_elec["district"].map({j:initial_partition["PRES16_DEM"][j]/(initial_partition["PRES16_DEM"][j]+initial_partition["PRES16_REP"][j]) for j in [x for x in initial_partition.parts]})
#df_elec["rep div dem rep"] = df_elec["district"].map({j:initial_partition["PRES16_REP"][j]/(initial_partition["PRES16_DEM"][j]+initial_partition["PRES16_REP"][j]) for j in [x for x in initial_partition.parts]})
#df_elec["updater dem"] = df_elec["district"].map(dict(zip(order, initial_partition['PRES16'].percents("Democratic"))))
#df_elec["updater rep"] = df_elec["district"].map(dict(zip(order, initial_partition['PRES16'].percents("Republican"))))

#initial partition
total_population = state_gdf[tot_pop].sum()
ideal_population = total_population/num_districts
random_assign = recursive_tree_part(graph, range(num_districts), ideal_population, tot_pop, pop_tol, node_repeats = 5)
assignment = assignment1 if start_map == 'enacted' else random_assign
initial_partition = GeographicPartition(graph = graph, assignment = assignment, updaters = my_updaters)

proposal = partial(
    recom, pop_col=tot_pop, pop_target=ideal_population, epsilon= pop_tol, node_repeats=3
)

#acceptance functions
accept = accept.always_accept

def hill_accept_bound(partition):
    if not partition.parent:
        return True
    proposal_vra = partition["vra_score"]
    parent_vra = partition.parent["vra_score"]
    if proposal_vra > parent_vra:
        return True
    else:
        draw = random.random()
        return draw < bound

def sim_anneal_accept(partition):
    if not partition.parent:
        return True
    proposal_vra = partition["vra_score"]
    parent_vra = partition.parent["vra_score"] 
    if step_Num % cycle_length < start_cool:
         return True

    elif step_Num % cycle_length > stop_cool:
        return proposal_vra > parent_vra

    else: 
        if proposal_vra > parent_vra:
            return True
        else:
            draw = random.random()      
            return draw < (stop_cool - (step_Num % cycle_length))/(stop_cool - start_cool)
       
#define chain
chain = MarkovChain(
    proposal = proposal,
    constraints=[
        constraints.within_percent_of_ideal_population(initial_partition, pop_tol), #% from ideal
    ],
            accept = accept if run_type == 'free' else \
            hill_accept_bound if run_type == 'hill_climb' \
            else sim_anneal_accept,
    initial_state = initial_partition,
    total_steps = total_steps
)

#prep storage for plans
store_plans = pd.DataFrame(columns = ["Index", "GEOID" ])
store_plans["Index"] = list(initial_partition.assignment.keys())
state_gdf_geoid = state_gdf[[geo_id]]
store_plans["GEOID"] = [state_gdf_geoid.iloc[i][0] for i in store_plans["Index"]]

#prep district-by-district storage (each metric in its own df)
final_state_prob_df = pd.DataFrame(columns = range(num_districts), index = list(range(store_interval)))
final_equal_prob_df = pd.DataFrame(columns = range(num_districts), index = list(range(store_interval)))
final_dist_prob_df = pd.DataFrame(columns = range(num_districts), index = list(range(store_interval)))

#demographic data storage (use 2018 CVAP for this!)
hisp_prop_df = pd.DataFrame(columns = range(num_districts), index = list(range(store_interval)))
black_prop_df = pd.DataFrame(columns = range(num_districts), index = list(range(store_interval)))
white_prop_df = pd.DataFrame(columns = range(num_districts), index = list(range(store_interval)))

#partisan data "input"
pres16_df = pd.DataFrame(columns = range(num_districts), index = list(range(store_interval)))
pres12_df = pd.DataFrame(columns = range(num_districts), index = list(range(store_interval)))
sen18_df = pd.DataFrame(columns = range(num_districts), index = list(range(store_interval)))
centroids_df = pd.DataFrame(columns = range(num_districts), index = list(range(store_interval)))

count_moves = 0
temp_score = 0
stuck_step = 0
step_Num = 0
best_score = 0
#run chain and collect data
for step in chain:
    #saving at intervals
    if step_Num % store_interval == 0 and step_Num > 0:
        store_plans.to_csv(DIR + "outputs/store_plans_{}.csv".format(run_name), index= False)

        if step_Num == store_interval:
            pres16_df.to_csv(DIR + "outputs/pres16_df_{}.csv".format(run_name), index = False)
            pres16_df = pd.DataFrame(columns = range(num_districts), index = list(range(store_interval)))
            pres12_df.to_csv(DIR + "outputs/pres12_df_{}.csv".format(run_name), index = False)
            pres12_df = pd.DataFrame(columns = range(num_districts), index = list(range(store_interval)))
            sen18_df.to_csv(DIR + "outputs/sen18_df_{}.csv".format(run_name), index = False)
            sen18_df = pd.DataFrame(columns = range(num_districts), index = list(range(store_interval)))
            centroids_df.to_csv(DIR + "outputs/centroids_df_{}.csv".format(run_name), index = False)
            centroids_df = pd.DataFrame(columns = range(num_districts), index = list(range(store_interval)))
            
            hisp_prop_df.to_csv(DIR + "outputs/hisp_prop_df_{}.csv".format(run_name), index = False)
            hisp_prop_df = pd.DataFrame(columns = range(num_districts), index = list(range(store_interval)))
            black_prop_df.to_csv(DIR + "outputs/black_prop_df_{}.csv".format(run_name), index = False)
            black_prop_df = pd.DataFrame(columns = range(num_districts), index = list(range(store_interval)))
            white_prop_df.to_csv(DIR + "outputs/white_prop_df_{}.csv".format(run_name), index = False)
            white_prop_df = pd.DataFrame(columns = range(num_districts), index = list(range(store_interval)))
            
            final_state_prob_df.to_csv(DIR + "outputs/final_state_prob_df_{}.csv".format(run_name), index= False)
            last_state_prob_dict = dict(zip(final_state_prob_df.columns, final_state_prob_df.loc[step_Num - 1]))
            final_state_prob_df = pd.DataFrame(columns = range(num_districts), index = [-1] + list(range(store_interval)))
            final_state_prob_df.loc[-1] = list(last_state_prob_dict.values())
            
            final_equal_prob_df.to_csv(DIR + "outputs/final_equal_prob_df_{}.csv".format(run_name), index= False)
            last_equal_prob_dict = dict(zip(final_equal_prob_df.columns, final_equal_prob_df.loc[step_Num - 1]))
            final_equal_prob_df = pd.DataFrame(columns = range(num_districts), index = [-1] + list(range(store_interval)))
            final_equal_prob_df.loc[-1] = list(last_equal_prob_dict.values())
            
            final_dist_prob_df.to_csv(DIR + "outputs/final_dist_prob_df_{}.csv".format(run_name), index= False)
            last_dist_prob_dict = dict(zip(final_dist_prob_df.columns, final_dist_prob_df.loc[step_Num - 1]))
            final_dist_prob_df = pd.DataFrame(columns = range(num_districts), index = [-1] + list(range(store_interval)))
            final_dist_prob_df.loc[-1] = list(last_dist_prob_dict.values())
            
          
        else:
            pres16_df.to_csv(DIR + "outputs/pres16_df_{}.csv".format(run_name), mode = 'a', header = False, index = False)
            pres16_df = pd.DataFrame(columns = range(num_districts), index = list(range(store_interval)))
            
            pres12_df.to_csv(DIR + "outputs/pres12_df_{}.csv".format(run_name), mode = 'a', header = False, index = False)
            pres12_df = pd.DataFrame(columns = range(num_districts), index = list(range(store_interval)))
            sen18_df.to_csv(DIR + "outputs/sen18_df_{}.csv".format(run_name), mode = 'a', header = False, index = False)
            sen18_df = pd.DataFrame(columns = range(num_districts), index = list(range(store_interval)))
            centroids_df.to_csv(DIR + "outputs/centroids_df_{}.csv".format(run_name), mode = 'a', header = False, index = False)
            centroids_df = pd.DataFrame(columns = range(num_districts), index = list(range(store_interval)))
            
            hisp_prop_df.to_csv(DIR + "outputs/hisp_prop_df_{}.csv".format(run_name), mode = 'a', header = False, index = False)
            hisp_prop_df = pd.DataFrame(columns = range(num_districts), index = list(range(store_interval)))
            black_prop_df.to_csv(DIR + "outputs/black_prop_df_{}.csv".format(run_name), mode = 'a', header = False, index = False)
            black_prop_df = pd.DataFrame(columns = range(num_districts), index = list(range(store_interval)))
            white_prop_df.to_csv(DIR + "outputs/white_prop_df_{}.csv".format(run_name), mode = 'a', header = False, index = False)
            white_prop_df = pd.DataFrame(columns = range(num_districts), index = list(range(store_interval)))
            
            last_state_prob_dict = dict(zip(final_state_prob_df.columns, final_state_prob_df.loc[(step_Num - 1) % store_interval]))
            final_state_prob_df.drop(-1).to_csv(DIR + "outputs/final_state_prob_df_{}.csv".format(run_name), mode = 'a', header = False, index= False)            
            final_state_prob_df = pd.DataFrame(columns = range(num_districts), index = [-1] + list(range(store_interval)))
            final_state_prob_df.loc[-1] = list(last_state_prob_dict.values())
            
            last_equal_prob_dict = dict(zip(final_equal_prob_df.columns, final_equal_prob_df.loc[(step_Num -1) % store_interval]))
            final_equal_prob_df.drop(-1).to_csv(DIR + "outputs/final_equal_prob_df_{}.csv".format(run_name), mode = 'a', header = False, index= False)            
            final_equal_prob_df = pd.DataFrame(columns = range(num_districts), index = [-1] + list(range(store_interval)))
            final_equal_prob_df.loc[-1] = list(last_equal_prob_dict.values())
            
            last_dist_prob_dict = dict(zip(final_dist_prob_df.columns, final_dist_prob_df.loc[(step_Num - 1) % store_interval]))
            final_dist_prob_df.drop(-1).to_csv(DIR + "outputs/final_dist_prob_df_{}.csv".format(run_name), mode = 'a', header = False, index= False)            
            final_dist_prob_df = pd.DataFrame(columns = range(num_districts), index = [-1] + list(range(store_interval)))
            final_dist_prob_df.loc[-1] = list(last_dist_prob_dict.values())
            
    if step.parent is not None:
        if step.assignment != step.parent.assignment:
            count_moves += 1
            
    #district-by-district storage
    centroids_data = step["centroids"]
    keys = list(centroids_data.keys())
    values = list(centroids_data.values())
    centroids_df.loc[step_Num % store_interval] = [value for _,value in sorted(zip(keys,values))]
    
    hisp_prop_data = step["demo_percents"][0]
    keys = list(hisp_prop_data.keys())
    values = list(hisp_prop_data.values())
    hisp_prop_df.loc[step_Num % store_interval] = [value for _,value in sorted(zip(keys,values))]    
    
    black_prop_data = step["demo_percents"][1]
    keys = list(black_prop_data.keys())
    values = list(black_prop_data.values())
    black_prop_df.loc[step_Num % store_interval] = [value for _,value in sorted(zip(keys,values))]
    
    white_prop_data = step["demo_percents"][2]
    keys = list(white_prop_data.keys())
    values = list(white_prop_data.values())
    white_prop_df.loc[step_Num % store_interval] = [value for _,value in sorted(zip(keys,values))]
    
    order = [x for x in step.parts]
    percents = {}
    for elec in elections_track:
        percents[elec.name] = dict(zip(order, step[elec.name].percents("Democratic")))
    
    keys = list(percents["PRES16"].keys())
    values = list(percents["PRES16"].values())
    pres16_df.loc[step_Num % store_interval] = [value for _,value in sorted(zip(keys,values))]
    
    keys = list(percents["PRES12"].keys())
    values = list(percents["PRES12"].values())
    pres12_df.loc[step_Num % store_interval] = [value for _,value in sorted(zip(keys,values))]
    
    keys = list(percents["SEN18"].keys())
    values = list(percents["SEN18"].values())
    sen18_df.loc[step_Num % store_interval] = [value for _,value in sorted(zip(keys,values))]
    
    final_state_prob_dict, final_equal_prob_dict, final_dist_prob_dict, \
    total_hisp_final, total_black_final = step["final_elec_model"]
    
    if step_Num == 0:
        keys = list(final_state_prob_dict.keys())
        values = list(final_state_prob_dict.values())
        final_state_prob_df.loc[step_Num] = [value for _,value in sorted(zip(keys,values))]                
   
        keys = list(final_equal_prob_dict.keys())
        values = list(final_equal_prob_dict.values())
        final_equal_prob_df.loc[step_Num] = [value for _,value in sorted(zip(keys,values))]                
   
        keys = list(final_dist_prob_dict.keys())
        values = list(final_dist_prob_dict.values())
        final_dist_prob_df.loc[step_Num] = [value for _,value in sorted(zip(keys,values))]                
    
    else:
        final_state_prob_df.loc[step_Num % store_interval] = final_state_prob_df.loc[(step_Num % store_interval) -1]
        for i in final_state_prob_dict.keys():
            final_state_prob_df.at[step_Num % store_interval, i] = final_state_prob_dict[i]
        
        final_equal_prob_df.loc[step_Num % store_interval] = final_equal_prob_df.loc[(step_Num % store_interval) -1]
        for i in final_equal_prob_dict.keys():
            final_equal_prob_df.at[step_Num % store_interval, i] = final_equal_prob_dict[i]

        final_dist_prob_df.loc[step_Num % store_interval] = final_dist_prob_df.loc[(step_Num % store_interval) -1]
        for i in final_dist_prob_dict.keys():
            final_dist_prob_df.at[step_Num % store_interval, i] = final_dist_prob_dict[i]

     #store plans        
    if (step["vra_score"] > best_score and run_type != 'free') or step_Num == 0:
        store_plans["{}".format(step_Num)] = store_plans["Index"].map(dict(step.assignment))
        best_score = step["vra_score"]
    if step["vra_score"] == temp_score:
        stuck_step += 1
    else:
        stuck_step = 0
    if stuck_step == stuck_length:
        print("Run stuck!")
        break
    
    temp_score = step["vra_score"]
    step_Num += 1

#output data
store_plans.to_csv(DIR + "outputs/store_plans_{}.csv".format(run_name), index= False)

#store district-by-district data
#demo data
hisp_prop_df.to_csv(DIR + "outputs/hisp_prop_df_{}.csv".format(run_name), mode = 'a', header = False, index= False)
black_prop_df.to_csv(DIR + "outputs/black_prop_df_{}.csv".format(run_name), mode = 'a', header = False, index= False)
white_prop_df.to_csv(DIR + "outputs/white_prop_df_{}.csv".format(run_name), mode = 'a', header = False, index= False)
#partisan data
pres16_df.to_csv(DIR + "outputs/pres16_df_{}.csv".format(run_name), mode = 'a', header = False, index = False)
pres12_df.to_csv(DIR + "outputs/pres12_df_{}.csv".format(run_name), mode = 'a', header = False, index = False)
sen18_df.to_csv(DIR + "outputs/sen18_df_{}.csv".format(run_name), mode = 'a', header = False, index = False)
centroids_df.to_csv(DIR + "centroids_df_{}.csv".format(run_name), mode = 'a', header = False, index = False)
#vra data
if total_steps < store_interval:
    final_state_prob_df.to_csv(DIR + "outputs/final_state_prob_df_{}.csv".format(run_name), index= False)
    final_equal_prob_df.to_csv(DIR + "outputs/final_equal_prob_df_{}.csv".format(run_name),  index= False)
    final_dist_prob_df.to_csv(DIR + "outputs/final_dist_prob_df_{}.csv".format(run_name), index= False)
else:  
    final_state_prob_df.drop(-1).to_csv(DIR + "outputs/final_state_prob_df_{}.csv".format(run_name), mode = 'a', header = False, index= False)
    final_equal_prob_df.drop(-1).to_csv(DIR + "outputs/final_equal_prob_df_{}.csv".format(run_name), mode = 'a', header = False, index= False)
    final_dist_prob_df.drop(-1).to_csv(DIR + "outputs/final_dist_prob_df_{}.csv".format(run_name), mode = 'a', header = False, index= False)
############# final print outs
print("--- %s TOTAL seconds ---" % (time.time() - start_time_total))
print("total moves", count_moves)
print("run name:", run_name)
print("num steps", total_steps)
print("current step", step_Num)


