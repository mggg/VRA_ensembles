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
from run_functions import compute_final_dist, compute_W2, prob_conf_conversion, cand_pref_outcome_sum, \
cand_pref_all_draws_outcomes, cand_pref_all, cand_pref_all_alt_qv
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
C_X = "C_X"
C_Y = "C_Y"

#user input parameters######################################
total_steps = 5
pop_tol = .01 #U.S. Cong
assignment1= 'CD' #CD, sldl358, sldu172, sldl309
run_name = 'Texas_run' #sys.argv[1]
run_type = 'free' 
model_mode = 'statewide' #'district', equal, statewide
start_map = 'Seed1' #'enacted' or a seed
ensemble_inclusion = False
ensemble_inclusion_demo = False
#if vra tilted tun
bound_type = 'bound2' #for vra tilted run
beta = 7 #parameter in vra_opt_accept function

#fixed parameters#################################################
num_districts = 36 #150 state house, 31 senate, 36 Cong
enacted_black = 4
enacted_hisp = 8
enacted_distinct = 11
plot_path = 'tx-results-cvap-sl-adjoined/tx-results-cvap-sl-adjoined.shp'  #for shapefile
store_interval = 20  #how many steps until storage

DIR = ''

#read files###################################################################
elec_data = pd.read_csv("TX_elections.csv")
TX_columns = list(pd.read_csv("TX_columns.csv")["Columns"])
dropped_elecs = pd.read_csv("dropped_elecs.csv")["Dropped Elections"]
recency_weights = pd.read_csv("recency_weights.csv")
min_cand_weights = pd.read_csv("min_pref_weight_binary.csv")
cand_race_table = pd.read_csv("Candidate_Race_Party.csv")
EI_statewide = pd.read_csv("statewide_rxc_EI_preferences.csv")
model_cutoffs = pd.read_csv("cutoffs.csv")
prec_ei_df = pd.read_csv("prec_count_quants.csv", dtype = {'CNTYVTD':'str'})
mean_prec_counts = pd.read_csv("mean_prec_vote_counts.csv", dtype = {'CNTYVTD':'str'})
logit_params = pd.read_csv("TX_logit_params.csv")
equal_seed_plans = pd.read_csv("seed_plans_equal.csv") 
state_seed_plans = pd.read_csv("seed_plans_state.csv") 
dist_seed_plans = pd.read_csv("seed_plans_dist.csv") 
demo_seed_plans = pd.read_csv("seed_plans_demo.csv")

#set up elections data structures
elections = list(elec_data["Election"]) 
elec_type = elec_data["Type"]
elec_cand_list = TX_columns

elecs_bool = ~elec_data.Election.isin(list(dropped_elecs))
elec_data_trunc = elec_data[elecs_bool].reset_index(drop = True)
elec_sets = list(set(elec_data_trunc["Election Set"]))
elections = list(elec_data_trunc["Election"])
general_elecs = list(elec_data_trunc[elec_data_trunc["Type"] == 'General'].Election)
primary_elecs = list(elec_data_trunc[elec_data_trunc["Type"] == 'Primary'].Election)
runoff_elecs = list(elec_data_trunc[elec_data_trunc["Type"] == 'Runoff'].Election)
#this dictionary matches a specific election with the election set it belongs to
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
state_gdf_cols = list(state_gdf.columns)
cand1_index = state_gdf_cols.index('RomneyR_12')
cand2_index = state_gdf_cols.index('ObamaD_12P')
state_gdf_cols[cand1_index:cand2_index+1] = TX_columns
state_gdf.columns = state_gdf_cols

state_df = pd.DataFrame(state_gdf)
state_df = state_df.drop(['geometry'], axis = 1)

##build graph from geo_dataframe
graph = Graph.from_geodataframe(state_gdf)
graph.add_data(state_gdf)
centroids = state_gdf.centroid
c_x = centroids.x
c_y = centroids.y
for node in graph.nodes():
    graph.nodes[node]["C_X"] = c_x[node]
    graph.nodes[node]["C_Y"] = c_y[node]


#make dictionary that maps an election to its candidates
#only include 2 major party candidates in generals (assumes here major party candidates are first in candidate list)
candidates = {}
for elec in elections:
    #get rid of republican candidates in primaries or runoffs (primary runoffs)
    cands = [y for y in elec_cand_list if elec in y and "R_" not in y.split('1')[0]] if \
    "R_" in elec[:4] or "P_" in elec[:4] else [y for y in elec_cand_list if elec in y] 
    
    elec_year = elec_data_trunc.loc[elec_data_trunc["Election"] == elec, 'Year'].values[0]          
    if elec in general_elecs:
        #assumes D and R are always first two candidates
        cands = cands[:2]
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
neither_conf_W3_state = pd.DataFrame(columns = range(num_districts))
neither_conf_W3_state["Election Set"] = elec_sets

#pre-compute recency_W1 df for all model modes, and W3, W2 dfs for statewide/equal modes    
for elec_set in elec_sets:
        elec_year = elec_data_trunc.loc[elec_data_trunc["Election Set"] == elec_set, 'Year'].values[0].astype(str)
        for dist in range(num_districts):
            recency_W1.at[recency_W1["Election Set"] == elec_set, dist] = recency_weights[elec_year][0]
   
# pref cands needed for statewide and equal, weights only needed for state 
for elec in primary_elecs + runoff_elecs:
    black_pref_cand = EI_statewide.loc[((EI_statewide["Election"] == elec) & (EI_statewide["Demog"] == 'BCVAP')), "Candidate"].values[0]
    hisp_pref_cand = EI_statewide.loc[((EI_statewide["Election"] == elec) & (EI_statewide["Demog"] == 'HCVAP')), "Candidate"].values[0]
    black_ei_prob = EI_statewide.loc[((EI_statewide["Election"] == elec) & (EI_statewide["Demog"] == 'BCVAP')), "prob"].values[0]
    hisp_ei_prob = EI_statewide.loc[((EI_statewide["Election"] == elec) & (EI_statewide["Demog"] == 'HCVAP')), "prob"].values[0]   
  
    for district in range(num_districts):
        if elec in primary_elecs:           
            black_pref_cands_prim_state.at[black_pref_cands_prim_state["Election Set"] == elec_match_dict[elec], district] = black_pref_cand
            black_conf_W3_state.at[black_conf_W3_state["Election Set"] == elec_match_dict[elec], district] = prob_conf_conversion(black_ei_prob)
            hisp_pref_cands_prim_state.at[hisp_pref_cands_prim_state["Election Set"] == elec_match_dict[elec], district] = hisp_pref_cand
            hisp_conf_W3_state.at[hisp_conf_W3_state["Election Set"] == elec_match_dict[elec], district] = prob_conf_conversion(hisp_ei_prob)                                             
            neither_conf_W3_state.at[neither_conf_W3_state["Election Set"] == elec_match_dict[elec], district] = prob_conf_conversion(hisp_ei_prob*black_ei_prob)                                             
            
        else:                        
            black_pref_cands_runoffs_state.at[black_pref_cands_runoffs_state["Election Set"] == elec_match_dict[elec], district] = black_pref_cand       
            hisp_pref_cands_runoffs_state.at[hisp_pref_cands_runoffs_state["Election Set"] == elec_match_dict[elec], district] = hisp_pref_cand
              
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

#equal weights are all 
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

#precompute set up for district mode (need precinct level EI set up)
#need to precompute all the column bases and dictionary for all (demog, election) pairs
demogs = ['BCVAP','HCVAP']
bases = {col.split('.')[0]+'.'+col.split('.')[1] for col in prec_ei_df.columns if col[:5] in demogs and 'abstain' not in col and \
          not any(x in col for x in general_elecs)}
base_dict = {b:(b.split('.')[0].split('_')[0],'_'.join(b.split('.')[1].split('_')[1:-1])) for b in bases}
outcomes = {val:[] for val in base_dict.values()}
for b in bases:
    outcomes[base_dict[b]].append(b) 
    
#AB add
precs = list(state_gdf['CNTYVTD'])
prec_draws_outcomes = cand_pref_all_draws_outcomes(prec_ei_df, precs, bases, outcomes)

############################################################################################################       
#UPDATERS FOR CHAIN
#elections model function. Takes in partition and returns effectiveness distribution per district
    #and total black effective and Latino-effective districts (>60% effective)
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
    #for particular elec and district can access all cand results by: dist_elec_results[elec][dist]
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
    neither_conf_W3_dist = pd.DataFrame(columns = dist_changes)
    neither_conf_W3_dist["Election Set"] = elec_sets
    
    black_align_prim_dist = pd.DataFrame(columns = dist_changes)
    black_align_prim_dist["Election Set"] = elec_sets
    hisp_align_prim_dist = pd.DataFrame(columns = dist_changes)
    hisp_align_prim_dist["Election Set"] = elec_sets
    
    #need to compute alignment score district by district - even in STATEWIDE modes!! (may have diff candidate of choice etc.)
    black_align_prim_state = pd.DataFrame(columns = range(num_districts))
    black_align_prim_state["Election Set"] = elec_sets
    hisp_align_prim_state = pd.DataFrame(columns = range(num_districts))
    hisp_align_prim_state["Election Set"] = elec_sets
    
    #to compute district weights, preferred candidate and confidence is computed
    #for each district at every ReCom step
    for district in dist_changes: #get vector of precinct values for each district                  
        #only need preferred candidates and condidence in primary and runoffs
        #(in Generals we only care if the Democrat wins)
        state_gdf["New Map"] = state_gdf.index.map(dict(partition.assignment))
        dist_prec_list =  list(state_gdf[state_gdf["New Map"] == district][geo_id])
        #district_support_all = cand_pref_all_alt_qv(prec_ei_df, dist_prec_list, bases, outcomes, sample_size = 1000)        
        dist_prec_indices = state_gdf.index[state_gdf['CNTYVTD'].isin(dist_prec_list)].tolist()
        district_support_all = cand_pref_outcome_sum(prec_draws_outcomes, dist_prec_indices, bases, outcomes)
        
        cand_counts_dist = mean_prec_counts[mean_prec_counts[geo_id].isin(dist_prec_list)]
        for elec in primary_elecs + runoff_elecs:                    
            HCVAP_support_elec = district_support_all[('HCVAP', elec)]
            hisp_pref_cand_dist = max(HCVAP_support_elec.items(), key=operator.itemgetter(1))[0]
            hisp_pref_prob_dist = HCVAP_support_elec[hisp_pref_cand_dist]
                        
            BCVAP_support_elec = district_support_all[('BCVAP', elec)]
            black_pref_cand_dist = max(BCVAP_support_elec.items(), key=operator.itemgetter(1))[0]
            black_pref_prob_dist = BCVAP_support_elec[black_pref_cand_dist]
            
            black_pref_cand_state = black_pref_cands_prim_state.loc[black_pref_cands_prim_state["Election Set"] == elec_match_dict[elec], district].values[0]
            hisp_pref_cand_state = hisp_pref_cands_prim_state.loc[hisp_pref_cands_prim_state["Election Set"] == elec_match_dict[elec], district].values[0]
            #computing preferred candidate and confidence in that choice gives is weight 3        
            if elec in primary_elecs:
                black_pref_cands_prim_dist.at[black_pref_cands_prim_dist["Election Set"] == elec_match_dict[elec], district] = black_pref_cand_dist
                black_conf_W3_dist.at[black_conf_W3_dist["Election Set"] == elec_match_dict[elec], district] = prob_conf_conversion(black_pref_prob_dist)
                hisp_pref_cands_prim_dist.at[hisp_pref_cands_prim_dist["Election Set"] == elec_match_dict[elec], district] = hisp_pref_cand_dist
                hisp_conf_W3_dist.at[hisp_conf_W3_dist["Election Set"] == elec_match_dict[elec], district] = prob_conf_conversion(hisp_pref_prob_dist)        
                neither_conf_W3_dist.at[neither_conf_W3_dist["Election Set"] == elec_match_dict[elec], district] = prob_conf_conversion(hisp_pref_prob_dist*black_pref_prob_dist)        
                
                black_align_prim_dist.at[black_align_prim_dist["Election Set"] == elec_match_dict[elec], district] = \
                sum(cand_counts_dist["BCVAP"+ '.' + black_pref_cand_dist])/(sum(cand_counts_dist["BCVAP"+ '.' + black_pref_cand_dist]) + sum(cand_counts_dist["HCVAP"+ '.' + black_pref_cand_dist]) + sum(cand_counts_dist["WCVAP"+ '.' + black_pref_cand_dist]) + sum(cand_counts_dist["OCVAP"+ '.' + black_pref_cand_dist]))                       
                hisp_align_prim_dist.at[hisp_align_prim_dist["Election Set"] == elec_match_dict[elec], district] = \
                sum(cand_counts_dist["HCVAP"+ '.' + hisp_pref_cand_dist])/(sum(cand_counts_dist["BCVAP"+ '.' + hisp_pref_cand_dist]) + sum(cand_counts_dist["HCVAP"+ '.' + hisp_pref_cand_dist]) + sum(cand_counts_dist["WCVAP"+ '.' + hisp_pref_cand_dist]) + sum(cand_counts_dist["OCVAP"+ '.' + hisp_pref_cand_dist]))
                
                black_align_prim_state.at[black_align_prim_state["Election Set"] == elec_match_dict[elec], district] = \
                sum(cand_counts_dist["BCVAP"+ '.' + black_pref_cand_state])/(sum(cand_counts_dist["BCVAP"+ '.' + black_pref_cand_state]) + sum(cand_counts_dist["HCVAP"+ '.' + black_pref_cand_state]) + sum(cand_counts_dist["WCVAP"+ '.' + black_pref_cand_state]) + sum(cand_counts_dist["OCVAP"+ '.' + black_pref_cand_state]))                       
                hisp_align_prim_state.at[hisp_align_prim_state["Election Set"] == elec_match_dict[elec], district] = \
                sum(cand_counts_dist["HCVAP"+ '.' + hisp_pref_cand_state])/(sum(cand_counts_dist["BCVAP"+ '.' + hisp_pref_cand_state]) + sum(cand_counts_dist["HCVAP"+ '.' + hisp_pref_cand_state]) + sum(cand_counts_dist["WCVAP"+ '.' + hisp_pref_cand_state]) + sum(cand_counts_dist["OCVAP"+ '.' + hisp_pref_cand_state]))
            
            else:
                black_pref_cands_runoffs_dist.at[black_pref_cands_runoffs_dist["Election Set"] == elec_match_dict[elec], district] = black_pref_cand_dist
                hisp_pref_cands_runoffs_dist.at[hisp_pref_cands_runoffs_dist["Election Set"] == elec_match_dict[elec], district] = hisp_pref_cand_dist
    
    black_align_prim_dist =  black_align_prim_dist.drop(['Election Set'], axis = 1)
    hisp_align_prim_dist =  hisp_align_prim_dist.drop(['Election Set'], axis = 1)      
    
    black_align_prim_state =  black_align_prim_state.drop(['Election Set'], axis = 1)
    hisp_align_prim_state =  hisp_align_prim_state.drop(['Election Set'], axis = 1) 
              
    ################################################################################
    #get election weight 2 (minority preferred minority_ and combine for final            
    min_cand_black_W2_dist, min_cand_hisp_W2_dist, min_cand_neither_W2_dist = compute_W2(elec_sets, \
          dist_changes, min_cand_weights_dict, black_pref_cands_prim_dist, hisp_pref_cands_prim_dist, cand_race_dict)
    
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
                 cand_race_table, num_districts, candidates, elec_sets, elec_set_dict,  \
                 black_align_prim_state, hisp_align_prim_state, "statewide", logit_params, logit = True, single_map = False)
    
    #district probability distribution: equal
    final_equal_prob_dict = compute_final_dist(map_winners, black_pref_cands_prim_state, black_pref_cands_runoffs_state,\
             hisp_pref_cands_prim_state, hisp_pref_cands_runoffs_state, neither_weight_equal, \
             black_weight_equal, hisp_weight_equal, dist_elec_results, dist_changes,
             cand_race_table, num_districts, candidates, elec_sets, elec_set_dict, \
             black_align_prim_state, hisp_align_prim_state, "equal", logit_params, logit = True, single_map = False)
    
       
    final_dist_prob_dict = compute_final_dist(map_winners, black_pref_cands_prim_dist, black_pref_cands_runoffs_dist,\
             hisp_pref_cands_prim_dist, hisp_pref_cands_runoffs_dist, neither_weight_dist, \
             black_weight_dist, hisp_weight_dist, dist_elec_results, dist_changes,
             cand_race_table, num_districts, candidates, elec_sets, elec_set_dict, \
             black_align_prim_dist, hisp_align_prim_dist, 'district', logit_params, logit = True, single_map = False)

    #new vector of probability distributions-by-district is the same as last ReCom step, except in 2 districts 
    if step_Num == 0:
         final_state_prob = {key:final_state_prob_dict[key] for key in sorted(final_state_prob_dict)}
         final_equal_prob = {key:final_equal_prob_dict[key] for key in sorted(final_equal_prob_dict)}
         final_dist_prob = {key:final_dist_prob_dict[key] for key in sorted(final_dist_prob_dict)}
         
    elif step_Num % store_interval == 0 and step_Num > 0:
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
                        
    black_threshold = model_cutoffs.loc[((model_cutoffs["model"] == model_mode) & (model_cutoffs["demog"] == 'BCVAP')), 'Cutoff'].values[0]
    hisp_threshold = model_cutoffs.loc[((model_cutoffs["model"] == model_mode) & (model_cutoffs["demog"] == 'HCVAP')), 'Cutoff'].values[0]
    
    hisp_effective = [i+l for i,j,k,l in optimize_dict.values()]
    black_effective = [j+l for i,j,k,l in optimize_dict.values()]
    
    hisp_effect_index = [i for i,n in enumerate(hisp_effective) if n >= hisp_threshold]
    black_effect_index = [i for i,n in enumerate(black_effective) if n >= black_threshold]
    
    total_hisp_final = len(hisp_effect_index)
    total_black_final = len(black_effect_index)
    total_distinct = len(set(hisp_effect_index + black_effect_index))
     
    return final_state_prob_dict, final_equal_prob_dict, final_dist_prob_dict, \
            total_hisp_final, total_black_final, total_distinct, optimize_dict
                 
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

def num_cut_edges(partition):
    return len(partition["cut_edges"])

def num_county_splits(partition, df = state_gdf):
    df["current"] = df.index.map(partition.assignment.to_dict())
    return sum(df.groupby(county_split_id)['current'].nunique() > 1)

#####construct updaters for Chain
my_updaters = {
    "population": updaters.Tally(tot_pop, alias = "population"),
    "CVAP": updaters.Tally(CVAP, alias = "CVAP"),
    "WCVAP": updaters.Tally(WCVAP, alias = "WCVAP"),
    "HCVAP": updaters.Tally(HCVAP, alias = "HCVAP"),
    "BCVAP": updaters.Tally(BCVAP, alias = "BCVAP"),
    "Sum_CX": updaters.Tally(C_X, alias = "Sum_CX"),
    "Sum_CY": updaters.Tally(C_Y, alias = "Sum_CY"),
    "cut_edges": cut_edges,
    "num_cut_edges": num_cut_edges,
    "num_county_splits": num_county_splits,
    "demo_percents": demo_percents,
    "final_elec_model": final_elec_model,
    "centroids": centroids
}

#add elections updaters
elections_track = [
    Election("PRES16", {"Democratic": 'ClintonD_16G_President' , "Republican": 'TrumpR_16G_President'}, alias = "PRES16"),
    Election("PRES12", {"Democratic": 'ObamaD_12G_President' , "Republican": 'RomneyR_12G_President'}, alias = "PRES12"),
    Election("SEN18", {"Democratic": "ORourkeD_18G_US_Sen" , "Republican": 'CruzR_18G_US_Sen'}, alias = "SEN18"),   
    Election("GOV18", {"Democratic": "ValdezD_18G_Governor" , "Republican": 'AbbottR_18G_Governor'}, alias = "GOV18"),   
    
]

election_updaters = {election.name: election for election in elections_track}
my_updaters.update(election_updaters)

election_functions = [Election(j, candidates[j]) for j in elections]
election_updaters = {election.name: election for election in election_functions}
my_updaters.update(election_updaters)

#initial partition#######################################################
step_Num = 0
total_population = state_gdf[tot_pop].sum()
ideal_population = total_population/num_districts

if ensemble_inclusion_demo:
    seed_plans = demo_seed_plans
elif model_mode == 'statewide':
    seed_plans = state_seed_plans
elif model_mode == 'equal':
    seed_plans = equal_seed_plans
else:
    seed_plans = dist_seed_plans
    
assignment = assignment1 if start_map == 'enacted' else dict(zip(seed_plans["Index"], seed_plans[start_map]))
initial_partition = GeographicPartition(graph = graph, assignment = assignment, updaters = my_updaters)

initial_partition.plot()
proposal = partial(
    recom, pop_col=tot_pop, pop_target=ideal_population, epsilon= pop_tol, node_repeats=3
)


#constraints ######################
def inclusion(partition):
    hisp_vra_dists = partition["final_elec_model"][3] 
    black_vra_dists = partition["final_elec_model"][4]    
    total_distinct = partition["final_elec_model"][5]   
    return total_distinct >= enacted_distinct and \
          black_vra_dists >= enacted_black and hisp_vra_dists >= enacted_hisp
          
def inclusion_demo(partition):
    bcvap_share_dict = {d:partition["BCVAP"][d]/partition["CVAP"][d] for d in partition.parts}
    hcvap_share_dict = {d:partition["HCVAP"][d]/partition["CVAP"][d] for d in partition.parts}       
    bcvap_share = list(bcvap_share_dict.values())
    hcvap_share = list(hcvap_share_dict.values())
    
    hcvap_over_thresh = len([k for k in hcvap_share if k > .45])
    bcvap_over_thresh = len([k for k in bcvap_share if k > .25 ])
    return (hcvap_over_thresh >= 8 and bcvap_over_thresh >= 4)

          
#acceptance functions #####################################
accept = accept.always_accept
         
    
#define chain
chain = MarkovChain(
    proposal = proposal,
    constraints = [constraints.within_percent_of_ideal_population(initial_partition, pop_tol), inclusion] \
            if ensemble_inclusion else [constraints.within_percent_of_ideal_population(initial_partition, pop_tol), inclusion_demo]\
            if ensemble_inclusion_demo else [constraints.within_percent_of_ideal_population(initial_partition, pop_tol)],
    accept = accept,
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
gov18_df = pd.DataFrame(columns = range(num_districts), index = list(range(store_interval)))
centroids_df = pd.DataFrame(columns = range(num_districts), index = list(range(store_interval)))

map_metric = pd.DataFrame(columns = ["HO", "BO", "Distinct", "Cut Edges", "County Splits"], index = list(range(store_interval)))
#effective precincts 
effect_precincts = pd.DataFrame(columns = ["BO", "HO", "Either"], index = store_plans["Index"])
effect_precincts = effect_precincts.fillna(0)

count_moves = 0
step_Num = 0
last_step_stored = 0
black_threshold = model_cutoffs.loc[((model_cutoffs["model"] == model_mode) & (model_cutoffs["demog"] == 'BCVAP')), 'Cutoff'].values[0]
hisp_threshold = model_cutoffs.loc[((model_cutoffs["model"] == model_mode) & (model_cutoffs["demog"] == 'HCVAP')), 'Cutoff'].values[0]
#run chain and collect data
start_time_total = time.time()
for step in chain:
    print("chain starting")
    final_state_prob_dict, final_equal_prob_dict, final_dist_prob_dict, \
    total_hisp_final, total_black_final, total_distinct, optimize_dict = step["final_elec_model"]
    map_metric.loc[step_Num] = [total_hisp_final, total_black_final, total_distinct, step["num_cut_edges"], step["num_county_splits"]]

    #saving at intervals
    if step_Num % store_interval == 0 and step_Num > 0:
        store_plans.to_csv(DIR + "outputs/store_plans_{}.csv".format(run_name), index= False)
        #store data and reset data frames
        if step_Num == store_interval:
            print("store data, step", step_Num)
            pres16_df.to_csv(DIR + "outputs/pres16_df_{}.csv".format(run_name), index = False)
            pres16_df = pd.DataFrame(columns = range(num_districts), index = list(range(store_interval)))
            pres12_df.to_csv(DIR + "outputs/pres12_df_{}.csv".format(run_name), index = False)
            pres12_df = pd.DataFrame(columns = range(num_districts), index = list(range(store_interval)))
            sen18_df.to_csv(DIR + "outputs/sen18_df_{}.csv".format(run_name), index = False)
            sen18_df = pd.DataFrame(columns = range(num_districts), index = list(range(store_interval)))
            gov18_df.to_csv(DIR + "outputs/gov18_df_{}.csv".format(run_name), index = False)
            gov18_df = pd.DataFrame(columns = range(num_districts), index = list(range(store_interval)))            
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
            gov18_df.to_csv(DIR + "outputs/gov18_df_{}.csv".format(run_name), mode = 'a', header = False, index = False)
            gov18_df = pd.DataFrame(columns = range(num_districts), index = list(range(store_interval)))           
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
      
    keys = list(percents["GOV18"].keys())
    values = list(percents["GOV18"].values())
    gov18_df.loc[step_Num % store_interval] = [value for _,value in sorted(zip(keys,values))]
      
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

    #store precincts that were in effective districts in this step
    step_distributions = final_state_prob_df.loc[step_Num % store_interval]  if model_mode == 'statewide' else \
            final_equal_prob_df.loc[step_Num % store_interval] if model_mode == 'equal' else \
            final_dist_prob_df.loc[step_Num % store_interval]
    step_dict = dict(zip((list(range(num_districts))), step_distributions))
    
    hisp_effective = [i+l for i,j,k,l in step_dict.values()]
    black_effective = [j+l for i,j,k,l in step_dict.values()]
    
    hisp_effect_index = [i for i,n in enumerate(hisp_effective) if n >= hisp_threshold]
    black_effect_index = [i for i,n in enumerate(black_effective) if n >= black_threshold]
    either_effect_index = [i for i in range(num_districts) if i in hisp_effect_index or i in black_effect_index]
    
    hisp_precincts = [prec for prec in step.assignment.keys() if step.assignment[prec] in hisp_effect_index]
    black_precincts = [prec for prec in step.assignment.keys() if step.assignment[prec] in black_effect_index]
    either_precincts = [prec for prec in step.assignment.keys() if step.assignment[prec] in either_effect_index]
              
    effect_precincts.loc[effect_precincts.index.isin(hisp_precincts), "HO"] = effect_precincts.loc[effect_precincts.index.isin(hisp_precincts), "HO"]+1
    effect_precincts.loc[effect_precincts.index.isin(black_precincts), "BO"] = effect_precincts.loc[effect_precincts.index.isin(black_precincts), "BO"]+1
    effect_precincts.loc[effect_precincts.index.isin(either_precincts), "Either"] = effect_precincts.loc[effect_precincts.index.isin(either_precincts), "Either"]+1
       
    #store plans     
    if (inclusion(step) and (step_Num - last_step_stored) > 500) or \
        ((not inclusion(step)) and step_Num % 1000 == 0): 
            
        last_step_stored = step_Num
        store_plans["Map{}".format(step_Num)] = store_plans["Index"].map(dict(step.assignment))
        print("stored new map!", total_hisp_final, total_black_final, total_distinct)
    
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
gov18_df.to_csv(DIR + "outputs/gov18_df_{}.csv".format(run_name), mode = 'a', header = False, index = False)
centroids_df.to_csv(DIR + "outputs/centroids_df_{}.csv".format(run_name), mode = 'a', header = False, index = False)

effect_precincts.to_csv(DIR + "outputs/effect_precincts_{}.csv".format(run_name), index = True)
map_metric.to_csv(DIR + "outputs/map_metric_{}.csv".format(run_name), index = True)
#vra data
if total_steps <= store_interval:
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

