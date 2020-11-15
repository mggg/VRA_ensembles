# -*- coding: utf-8 -*-
"""
Created on Sun Nov 15 14:09:14 2020

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

#final_elec_model is edited in this script to be an updater that evaluates all districts in a 
#partition. However, because we can compute statewide election weights just based on EI outputs, 
#we still keep those computations (as well as setting up data and election data structions) outside
#of the updater. The updater also still uses functions stored in the run_functions script

#fixed parameters#################################################
num_districts = 36 #36 Congressional districts
plot_path = 'TX_VTDs/TX_VTDs.shp'  #for shapefile
run_type = 'free' 

DIR = ''

##################################################################
#key column names from Texas VTD shapefile
geo_id = 'CNTYVTD'
C_X = "C_X"
C_Y = "C_Y"

#read files###################################################################
elec_data = pd.read_csv("TX_elections.csv")
TX_columns = list(pd.read_csv("TX_columns.csv")["Columns"])
dropped_elecs = pd.read_csv("dropped_elecs.csv")["Dropped Elections"]
recency_weights = pd.read_csv("recency_weights.csv")
min_cand_weights = pd.read_csv("ingroup_weight.csv")
cand_race_table = pd.read_csv("Candidate_Race_Party.csv")
EI_statewide = pd.read_csv("statewide_rxc_EI_preferences.csv")
prec_ei_df = pd.read_csv("prec_count_quants.csv", dtype = {'CNTYVTD':'str'})
mean_prec_counts = pd.read_csv("mean_prec_vote_counts.csv", dtype = {'CNTYVTD':'str'})
logit_params = pd.read_csv("TX_logit_params.csv")

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

#pre-compute recency_W1 df for all model modes 
for elec_set in elec_sets:
        elec_year = elec_data_trunc.loc[elec_data_trunc["Election Set"] == elec_set, 'Year'].values[0].astype(str)
        for dist in range(num_districts):
            recency_W1.at[recency_W1["Election Set"] == elec_set, dist] = recency_weights[elec_year][0]
   
#pre-compute W2 and W3 dfs for statewide/equal modes   
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

#compute final election weights by taking product of W1, W2, and W3 for each election set and district
#Note: because these are statewide weights, an election set will have the same weight across districts
black_weight_state = recency_W1.drop(["Election Set"], axis=1)*min_cand_black_W2_state.drop(["Election Set"], axis=1)*black_conf_W3_state.drop(["Election Set"], axis=1)
black_weight_state["Election Set"] = elec_sets
hisp_weight_state = recency_W1.drop(["Election Set"], axis=1)*min_cand_hisp_W2_state.drop(["Election Set"], axis=1)*hisp_conf_W3_state.drop(["Election Set"], axis=1)    
hisp_weight_state["Election Set"] = elec_sets
neither_weight_state = recency_W1.drop(["Election Set"], axis=1)*min_cand_neither_W2_state.drop(["Election Set"], axis=1)*neither_conf_W3_state.drop(["Election Set"], axis=1)    
neither_weight_state["Election Set"] = elec_sets

#equal weights are all 1
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
    
precs = list(state_gdf[geo_id])
prec_draws_outcomes = cand_pref_all_draws_outcomes(prec_ei_df, precs, bases, outcomes)

#####################################################################################################
#####################################################################################################
#The elections model function (used as an updater). Takes in partition and returns effectiveness distribution per district
#for statewide, equal and district scores
#The pre-computed statewide election weights and data structures are used in the updater
def final_elec_model(partition):  
    #Overview#####################################################  
    #The output of the elections model is a probability distribution for each district:
    #% Latino, Black, Neither or Overlap effective
    #To compute these, each election set is first weighted (different for Black and Latino)
    #by multiplying a recency weight (W1), "in-group"-minority-preference weight (W2) and 
    #a preferred-candidate-confidence weight (W3).
    #If the Black (Latino) preferred candidate wins the election (set) a number of points equal to
    #the set's weight is accrued. The ratio of the accrued points points to the total possible points
    #is the raw Black (Latino)-effectiviness score for the district. 
    
    # After the raw scores are computed, they are adjusted using an "Alignment" score, or a score
    #the share of votes cast for a minority-preferred candidate by the minority group itself.
    
    # Finally, the Black, Latino, Overlap, and Neither distribution (the values sum to 1) 
    # is computed, by feeding the adjusted effectiveness scores through a logit function,
    # and interpolating for the final four values.
    
    #We need to track several entities in the model, which will be dataframes, whose columns are districts and
    #rows are election sets (or sometimes individual elections)
    #These dataframes each store one of the following: Black (latino) preferred candidates (in the
    #election set's primary), Black (Latino) preferred candidates in runoffs, winners of primary,
    #runoff and general elections, election winners, weights W1, W2 and W3, Alignment scores
    #and final election set weight for Black and Latino voters
    ###########################################################    
    dist_changes = range(num_districts) #run on all districts
      
    #dictionary to store district-level candidate vote shares
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
    
    #store runoff preferences for instances where minority-preferred candidate needs to switch between primary and runoff
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
    
    #Compute Alignment score district by district - even in statewide/equal modes! (district and statewide preferred candidates can be different)
    black_align_prim_state = pd.DataFrame(columns = range(num_districts))
    black_align_prim_state["Election Set"] = elec_sets
    hisp_align_prim_state = pd.DataFrame(columns = range(num_districts))
    hisp_align_prim_state["Election Set"] = elec_sets
    
    #Compute W3 for the district mode and the Alignment score for all modes
    for district in dist_changes:        
        state_gdf["New Map"] = state_gdf.index.map(dict(partition.assignment))
        dist_prec_list =  list(state_gdf[state_gdf["New Map"] == district][geo_id])
        dist_prec_indices = state_gdf.index[state_gdf[geo_id].isin(dist_prec_list)].tolist()
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
              
    #compute W2 ("in-group"-minority-preference weight)        
    min_cand_black_W2_dist, min_cand_hisp_W2_dist, min_cand_neither_W2_dist = compute_W2(elec_sets, \
          dist_changes, min_cand_weights_dict, black_pref_cands_prim_dist, hisp_pref_cands_prim_dist, cand_race_dict)
    ################################################################################    
    black_weight_dist = recency_W1.drop(["Election Set"], axis=1)*min_cand_black_W2_dist.drop(["Election Set"], axis=1)*black_conf_W3_dist.drop(["Election Set"], axis=1)
    black_weight_dist["Election Set"] = elec_sets
    hisp_weight_dist = recency_W1.drop(["Election Set"], axis=1)*min_cand_hisp_W2_dist.drop(["Election Set"], axis=1)*hisp_conf_W3_dist.drop(["Election Set"], axis=1)    
    hisp_weight_dist["Election Set"] = elec_sets
    neither_weight_dist = recency_W1.drop(["Election Set"], axis=1)*min_cand_neither_W2_dist.drop(["Election Set"], axis=1)*neither_conf_W3_dist.drop(["Election Set"], axis=1)    
    neither_weight_dist["Election Set"] = elec_sets
                                  
    #################################################################################  
    #district probability distribution: statewide
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
    
    #district probability distribution: district   
    final_dist_prob_dict = compute_final_dist(map_winners, black_pref_cands_prim_dist, black_pref_cands_runoffs_dist,\
             hisp_pref_cands_prim_dist, hisp_pref_cands_runoffs_dist, neither_weight_dist, \
             black_weight_dist, hisp_weight_dist, dist_elec_results, dist_changes,
             cand_race_table, num_districts, candidates, elec_sets, elec_set_dict, \
             black_align_prim_dist, hisp_align_prim_dist, 'district', logit_params, logit = True, single_map = False)

    
    return sorted(final_state_prob_dict), sorted(final_equal_prob_dict), sorted(final_dist_prob_dict)
