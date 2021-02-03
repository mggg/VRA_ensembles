# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 12:19:57 2020

@author: darac
"""
import random
a = random.randint(0,10000000000)
a = 10
import networkx as nx
from gerrychain.random import random
random.seed(a)
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
import scipy
from scipy import stats
import sys
from functools import partial
from run_functions import compute_final_dist, compute_W2, prob_conf_conversion, cand_pref_outcome_sum, \
cand_pref_all_draws_outcomes, precompute_state_weights, compute_district_weights, compute_align_scores
from ast import literal_eval

#user input parameters######################################
total_steps = 1000
pop_tol = .01 #U.S. Cong (deviation from ideal district population)
run_name = 'Texas_neutral_run'
start_map = 'CD' #CD, 'Seed_Demo', or "new_seed"
effectiveness_cutoff = .6
ensemble_inclusion = True
ensemble_inclusion_demo = False
record_statewide_modes = True
record_district_mode = True
model_mode = 'statewide' #'district', 'equal', 'statewide'

store_interval = 200  #how many Markov chain steps between data storage

#fixed parameters#################################################
num_districts = 36 #36 Congressional districts
enacted_black = 4 #number of districts in enacted map with B + Ov > 60%
enacted_hisp = 8 #number of districts in enacted map with L + Ov > 60%
enacted_distinct = 11 #number of districts in enacted map with B + Ov > 60% or L + Ov > 60% or both
plot_path = 'TX_VTDs/TX_VTDs.shp'  #for shapefile

DIR = ''
if not os.path.exists(DIR + 'outputs'):
    os.mkdir(DIR + 'outputs')
    
##################################################################
#key column names from Texas VTD shapefile
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

#initialize state_gdf########################################################
#reformat/re-index enacted map plans
state_gdf = gpd.read_file(plot_path)
state_gdf["CD"] = state_gdf["CD"].astype('int')
state_gdf["Seed_Demo"] = state_gdf["Seed_Demo"].astype('int')
state_gdf.columns = state_gdf.columns.str.replace("-", "_")

#replace cut-off candidate names from shapefile with full names
state_gdf_cols = list(state_gdf.columns)
cand1_index = state_gdf_cols.index('RomneyR_12')
cand2_index = state_gdf_cols.index('ObamaD_12P')
state_gdf_cols[cand1_index:cand2_index+1] = TX_columns
state_gdf.columns = state_gdf_cols
state_df = pd.DataFrame(state_gdf)
state_df = state_df.drop(['geometry'], axis = 1)

#build graph from geo_dataframe #####################################################
graph = Graph.from_geodataframe(state_gdf)
graph.add_data(state_gdf)
centroids = state_gdf.centroid
c_x = centroids.x
c_y = centroids.y
for node in graph.nodes():
    graph.nodes[node]["C_X"] = c_x[node]
    graph.nodes[node]["C_Y"] = c_y[node]
#    
#set up elections data structures ################################################
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

########################################## pre-compute as much as possible for elections updater ##########
#pre-compute recency_W1 df for all model modes 
elec_years = [elec_data_trunc.loc[elec_data_trunc["Election Set"] == elec_set, 'Year'].values[0].astype(str) \
              for elec_set in elec_sets]
recency_scores = [recency_weights[elec_year][0] for elec_year in elec_years]
recency_W1 = np.tile(recency_scores, (num_districts, 1)).transpose()

#precompute statewide EI and W1, W2, W3 for statewide/equal modes 
if record_statewide_modes:
    black_weight_state, hisp_weight_state, neither_weight_state, black_weight_equal,\
    hisp_weight_equal, neither_weight_equal, black_pref_cands_prim_state, hisp_pref_cands_prim_state, \
    black_pref_cands_runoffs_state, hisp_pref_cands_runoffs_state\
                     = precompute_state_weights(num_districts, elec_sets, elec_set_dict, recency_W1, EI_statewide, primary_elecs, \
                     runoff_elecs, elec_match_dict, min_cand_weights_dict, cand_race_dict)

#precompute set-up for district mode, if used (need precinct level EI set up)
#need to precompute all the column bases and dictionary for all (demog, election) pairs
if record_district_mode:             
    demogs = ['BCVAP','HCVAP']
    bases = {col.split('.')[0]+'.'+col.split('.')[1] for col in prec_ei_df.columns if col[:5] in demogs and 'abstain' not in col and \
              not any(x in col for x in general_elecs)}
    base_dict = {b:(b.split('.')[0].split('_')[0],'_'.join(b.split('.')[1].split('_')[1:-1])) for b in bases}
    outcomes = {val:[] for val in base_dict.values()}
    for b in bases:
        outcomes[base_dict[b]].append(b) 
        
    precs = list(state_gdf[geo_id])
    prec_draws_outcomes = cand_pref_all_draws_outcomes(prec_ei_df, precs, bases, outcomes)

############################################################################################################       
#UPDATERS FOR CHAIN

#The elections model function (used as an updater). Takes in partition and returns effectiveness distribution per district
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
    # that measures the share of votes cast for a minority-preferred candidate by the minority group itself.
    
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
    #We only need to run model on two ReCom districts that have changed in each step
    if partition.parent is not None:
        dict1 = dict(partition.parent.assignment)
        dict2 = dict(partition.assignment)
        differences = set([dict1[k] for k in dict1.keys() if dict1[k] != dict2[k]]).union(set([dict2[k] for k in dict2.keys() if dict1[k] != dict2[k]]))
        
    dist_changes = range(num_districts) if partition.parent is None else sorted(differences)  
 
    #dictionary to store district-level candidate vote shares
    dist_elec_results = {}
    order = [x for x in partition.parts]
    for elec in elections:
        cands = candidates[elec]
        dist_elec_results[elec] = {}
        outcome_list = [dict(zip(order, partition[elec].percents(cand))) for cand in cands.keys()]      
        dist_elec_results[elec] = {d: {cands[i]: outcome_list[i][d] for i in cands.keys()} for d in range(num_districts)}
    ##########################################################################################   
    #compute winners of each election in each district and store:
    map_winners = pd.DataFrame(columns = dist_changes)
    map_winners["Election"] = elections
    map_winners["Election Set"] = elec_data_trunc["Election Set"]
    map_winners["Election Type"] = elec_data_trunc["Type"]
    for i in dist_changes:
        map_winners[i] = [max(dist_elec_results[elec][i].items(), key=operator.itemgetter(1))[0] for elec in elections]

#########################################################################################
    #If we compute statewide modes: compute alignment/group-control scores for each district #################
    #and final probability distributions
    if record_statewide_modes: 
        black_align_prim_state, hisp_align_prim_state = compute_align_scores(dist_changes, elec_sets, state_gdf, partition, primary_elecs, \
                                                        black_pref_cands_prim_state, hisp_pref_cands_prim_state, elec_match_dict, \
                                                        mean_prec_counts, geo_id)
        
                                           
        #district probability distribution: statewide
        final_state_prob_dict = compute_final_dist(map_winners, black_pref_cands_prim_state, black_pref_cands_runoffs_state,\
                                hisp_pref_cands_prim_state, hisp_pref_cands_runoffs_state, neither_weight_state, \
                                black_weight_state, hisp_weight_state, dist_elec_results, dist_changes,
                                cand_race_table, num_districts, candidates, elec_sets, elec_set_dict,  \
                                black_align_prim_state, hisp_align_prim_state, "statewide", logit_params, logit = True)
        
        #district probability distribution: equal
        final_equal_prob_dict = compute_final_dist(map_winners, black_pref_cands_prim_state, black_pref_cands_runoffs_state,\
                                hisp_pref_cands_prim_state, hisp_pref_cands_runoffs_state, neither_weight_equal, \
                                black_weight_equal, hisp_weight_equal, dist_elec_results, dist_changes,
                                cand_race_table, num_districts, candidates, elec_sets, elec_set_dict, \
                                black_align_prim_state, hisp_align_prim_state, "equal", logit_params, logit = True)
    
    #If we are computing district mode: ######################################################
    #compute district weights, preferred candidates, alignment/group control and district probability distribution: district   
    if record_district_mode: 
        black_weight_dist, hisp_weight_dist, neither_weight_dist, black_pref_cands_prim_dist,\
        black_pref_cands_runoffs_dist, hisp_pref_cands_prim_dist, hisp_pref_cands_runoffs_dist\
                                 = compute_district_weights(dist_changes, elec_sets, elec_set_dict, state_gdf, partition, prec_draws_outcomes,\
                                 geo_id, primary_elecs, runoff_elecs, elec_match_dict, bases, outcomes,\
                                 recency_W1, cand_race_dict, min_cand_weights_dict)
        
        black_align_prim_dist, hisp_align_prim_dist = compute_align_scores(dist_changes, elec_sets, state_gdf, partition, primary_elecs, \
                                                      black_pref_cands_prim_dist, hisp_pref_cands_prim_dist, elec_match_dict, \
                                                      mean_prec_counts, geo_id)

        final_dist_prob_dict = compute_final_dist(map_winners, black_pref_cands_prim_dist, black_pref_cands_runoffs_dist,\
                               hisp_pref_cands_prim_dist, hisp_pref_cands_runoffs_dist, neither_weight_dist, \
                               black_weight_dist, hisp_weight_dist, dist_elec_results, dist_changes,
                               cand_race_table, num_districts, candidates, elec_sets, elec_set_dict, \
                               black_align_prim_dist, hisp_align_prim_dist, 'district', logit_params, logit = True)

    #new vector of probability distributions-by-district is the same as last ReCom step, 
    #except in 2 changed districts 
    if partition.parent == None:
         final_state_prob = {key:final_state_prob_dict[key] for key in sorted(final_state_prob_dict)}\
         if record_statewide_modes else {key:"N/A" for key in sorted(dist_changes)}
         
         final_equal_prob = {key:final_equal_prob_dict[key] for key in sorted(final_equal_prob_dict)}\
         if record_statewide_modes else {key:"N/A" for key in sorted(dist_changes)}
         
         final_dist_prob = {key:final_dist_prob_dict[key] for key in sorted(final_dist_prob_dict)}\
         if record_district_mode else {key:"N/A" for key in sorted(dist_changes)}
         
    else:
        final_state_prob = partition.parent["final_elec_model"][0].copy()
        final_equal_prob =  partition.parent["final_elec_model"][1].copy()
        final_dist_prob = partition.parent["final_elec_model"][2].copy()
        
        for i in dist_changes:
            if record_statewide_modes:
                final_state_prob[i] = final_state_prob_dict[i]
                final_equal_prob[i] = final_equal_prob_dict[i]
            
            if record_district_mode:
                final_dist_prob[i] = final_dist_prob_dict[i]
    
    return final_state_prob, final_equal_prob, final_dist_prob

def effective_districts(dictionary):
    black_threshold = effectiveness_cutoff
    hisp_threshold = effectiveness_cutoff
    
    if "N/A" not in dictionary.values():
        hisp_effective = [i+l for i,j,k,l in dictionary.values()]
        black_effective = [j+l for i,j,k,l in dictionary.values()]
        
        hisp_effect_index = [i for i,n in enumerate(hisp_effective) if n >= hisp_threshold]
        black_effect_index = [i for i,n in enumerate(black_effective) if n >= black_threshold]
        
        total_hisp_final = len(hisp_effect_index)
        total_black_final = len(black_effect_index)
        total_distinct = len(set(hisp_effect_index + black_effect_index))
       
        return total_hisp_final, total_black_final, total_distinct
    
    else:
        return "N/A", "N/A", "N/A"
    
                 
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

#####construct updaters for Chain###############################################
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
    
if start_map == 'new_seed':
    start_map = recursive_tree_part(graph, range(num_districts), ideal_population, tot_pop, pop_tol, 3)    

initial_partition = GeographicPartition(graph = graph, assignment = start_map, updaters = my_updaters)

proposal = partial(
    recom, pop_col=tot_pop, pop_target=ideal_population, epsilon= pop_tol, node_repeats=3
)

#constraints ######################
def inclusion(partition):
    final_state_prob, final_equal_prob, final_dist_prob = partition["final_elec_model"]
    inclusion_dict = final_state_prob if model_mode == 'statewide' else final_equal_prob if model_mode == 'equal' else final_dist_prob
    hisp_vra_dists, black_vra_dists, total_distinct = effective_districts(inclusion_dict)
    
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
          
#set Markov chain parameters
chain = MarkovChain(
    proposal = proposal,
    constraints = [constraints.within_percent_of_ideal_population(initial_partition, pop_tol), inclusion] \
            if ensemble_inclusion else [constraints.within_percent_of_ideal_population(initial_partition, pop_tol), inclusion_demo]\
            if ensemble_inclusion_demo else [constraints.within_percent_of_ideal_population(initial_partition, pop_tol)],
    accept = accept,
    initial_state = initial_partition,
    total_steps = total_steps
)

#prep plan storage #################################################################################
store_plans = pd.DataFrame(columns = ["Index", "GEOID" ])
store_plans["Index"] = list(initial_partition.assignment.keys())
state_gdf_geoid = state_gdf[[geo_id]]
store_plans["GEOID"] = [state_gdf_geoid.iloc[i][0] for i in store_plans["Index"]]
map_metric = pd.DataFrame(columns = ["Latino_state", "Black_state", "Distinct_state",\
                                     "Latino_equal", "Black_equal", "Distinct_equal", \
                                     "Latino_dist", "Black_dist", "Distinct_dist", \
                                     "Cut Edges", "County Splits"], index = list(range(store_interval)))

#prep district-by-district storage
#score distributions
score_dfs = []
score_df_names = []
if record_statewide_modes:
    final_state_prob_df = pd.DataFrame(columns = range(num_districts), index = list(range(store_interval)))
    final_equal_prob_df = pd.DataFrame(columns = range(num_districts), index = list(range(store_interval)))
    score_dfs.extend([final_state_prob_df, final_equal_prob_df])
    score_df_names.extend(['final_state_prob_df', 'final_equal_prob_df'])
if record_district_mode:
    final_dist_prob_df = pd.DataFrame(columns = range(num_districts), index = list(range(store_interval)))
    score_dfs.append(final_dist_prob_df)
    score_df_names.append('final_dist_prob_df')
    
    
#demographic data storage (uses 2018 CVAP)
hisp_prop_df = pd.DataFrame(columns = range(num_districts), index = list(range(store_interval)))
black_prop_df = pd.DataFrame(columns = range(num_districts), index = list(range(store_interval)))
white_prop_df = pd.DataFrame(columns = range(num_districts), index = list(range(store_interval)))
#partisan data storage
pres16_df = pd.DataFrame(columns = range(num_districts), index = list(range(store_interval)))
pres12_df = pd.DataFrame(columns = range(num_districts), index = list(range(store_interval)))
sen18_df = pd.DataFrame(columns = range(num_districts), index = list(range(store_interval)))
gov18_df = pd.DataFrame(columns = range(num_districts), index = list(range(store_interval)))
#district centroids storage
centroids_df = pd.DataFrame(columns = range(num_districts), index = list(range(store_interval)))

#run chain and collect data ##############################################################################
count_moves = 0
step_Num = 0
last_step_stored = 0
start_time_total = time.time()

print("chain starting")
for step in chain:  
    final_state_prob, final_equal_prob, final_dist_prob = step["final_elec_model"]
            
    total_hisp_final_state, total_black_final_state, total_distinct_state = effective_districts(final_state_prob)
    total_hisp_final_equal, total_black_final_equal, total_distinct_equal = effective_districts(final_equal_prob)
    total_hisp_final_dist, total_black_final_dist, total_distinct_dist = effective_districts(final_dist_prob)
    
    #store all scores for all distribution types    
    map_metric.loc[step_Num] = [total_hisp_final_state, total_black_final_state, total_distinct_state,\
                      total_hisp_final_equal, total_black_final_equal, total_distinct_equal, \
                      total_hisp_final_dist, total_black_final_dist, total_distinct_dist,\
                      step["num_cut_edges"], step["num_county_splits"]]
    
    #saving all data at intervals
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
            
            for score_df, score_df_name in zip(score_dfs, score_df_names):
                score_df.to_csv(DIR + "outputs/{}_{}.csv".format(score_df_name,run_name), index= False)
                score_df = pd.DataFrame(columns = range(num_districts), index = list(range(store_interval)))           
              
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
            
            for score_df, score_df_name in zip(score_dfs, score_df_names):
                score_df.to_csv(DIR + "outputs/{}_{}.csv".format(score_df_name,run_name), mode = 'a', header = False, index= False)
                score_df = pd.DataFrame(columns = range(num_districts), index = list(range(store_interval)))           
            
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
    
    order = [int(x) for x in step.parts]
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
                  
    if record_statewide_modes:
        final_state_prob_df.loc[step_Num % store_interval] = list(final_state_prob.values())                
        final_equal_prob_df.loc[step_Num % store_interval] = list(final_equal_prob.values())               
   
    if record_district_mode:
        final_dist_prob_df.loc[step_Num % store_interval] = list(final_dist_prob.values())                

    #store plans     
    if (step_Num - last_step_stored) == store_interval or step_Num == 0:          
        last_step_stored = step_Num
        store_plans["Map{}".format(step_Num)] = store_plans["Index"].map(dict(step.assignment))               
        print("stored new map!", "step num", step_Num)
    
    step_Num += 1

#output data #########################################################
store_plans.to_csv(DIR + "outputs/store_plans_{}.csv".format(run_name), index= False)
hisp_prop_df.to_csv(DIR + "outputs/hisp_prop_df_{}.csv".format(run_name), mode = 'a', header = False, index= False)
black_prop_df.to_csv(DIR + "outputs/black_prop_df_{}.csv".format(run_name), mode = 'a', header = False, index= False)
white_prop_df.to_csv(DIR + "outputs/white_prop_df_{}.csv".format(run_name), mode = 'a', header = False, index= False)
pres16_df.to_csv(DIR + "outputs/pres16_df_{}.csv".format(run_name), mode = 'a', header = False, index = False)
pres12_df.to_csv(DIR + "outputs/pres12_df_{}.csv".format(run_name), mode = 'a', header = False, index = False)
sen18_df.to_csv(DIR + "outputs/sen18_df_{}.csv".format(run_name), mode = 'a', header = False, index = False)
gov18_df.to_csv(DIR + "outputs/gov18_df_{}.csv".format(run_name), mode = 'a', header = False, index = False)
centroids_df.to_csv(DIR + "outputs/centroids_df_{}.csv".format(run_name), mode = 'a', header = False, index = False)
map_metric.to_csv(DIR + "outputs/map_metric_{}.csv".format(run_name), index = True)
#vra data
if total_steps <= store_interval:
    for score_df, score_df_name in zip(score_dfs, score_df_names):        
        score_df.to_csv(DIR + "outputs/{}_{}.csv".format(score_df_name, run_name), index= False)
else:  
    for score_df, score_df_name in zip(score_dfs, score_df_names):      
        score_df.to_csv(DIR + "outputs/{}_{}.csv".format(score_df_name, run_name), mode = 'a', header = False, index= False)

############# final print outs ####################################################
print("--- %s TOTAL seconds ---" % (time.time() - start_time_total))
print("ave sec per step", (time.time() - start_time_total)/total_steps)
print("total moves", count_moves)
print("run name:", run_name)
print("num steps", total_steps)
print("current step", step_Num)

