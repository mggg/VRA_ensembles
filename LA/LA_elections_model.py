# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 12:19:57 2020

@author: darac
"""
import random
a = random.randint(0,10000000000)
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
import statsmodels.api as sm
import scipy
from scipy import stats
import sys
from functools import partial
from run_functions_LA import compute_final_dist, compute_W2, prob_conf_conversion, cand_pref_outcome_sum, \
cand_pref_all_draws_outcomes, compute_district_weights, precompute_state_weights
from ast import literal_eval
#############################################################################################################

#user input parameters ######################################################
total_steps = 1000
pop_tol = .01  
run_name = 'LA_neutral_Cong_run' 
start_map = 'SEND' # CD or 'new_seed'
effectiveness_cutoff = .65
ensemble_inclusion = False
record_statewide_modes = True
record_district_mode = True
model_mode = 'statewide' #'district', 'equal', 'statewide'
store_interval = 200  #number of steps between data storage intervals

#fixed parameters ########################################################
enacted_black = 1 #1 Congress, 11 State Senate
num_districts = 6 #39 senate, 6 Cong
plot_path = 'LA_final/LA_final.shp' 
county_split_id = 'COUNTYFP'

DIR = ''
if not os.path.exists(DIR + 'outputs'):
    os.mkdir(DIR + 'outputs')

##################################################################
#key column names from Texas VTD shapefile
tot_pop = 'TOTPOP'
white_pop = 'NH_WHITE'
CVAP = "CVAP18"
WCVAP = "WCVAP18"
HCVAP = "HCVAP18"
BCVAP = "BCVAP18" 
geo_id = 'GEOID10'
C_X = "C_X"
C_Y = "C_Y"

#read files#####################################################################
elec_data = pd.read_csv("LA_elections.csv")
elec_columns = list(pd.read_csv("LA_column_names.csv")["new"])
dropped_elecs = pd.read_csv("LA_dropped_elecs.csv")["Dropped Elections"]
recency_weights = pd.read_csv("recency_weights.csv")
min_cand_weights = pd.read_csv("min_pref_weight_binary.csv")
cand_race_table = pd.read_csv("LA_Candidate_Race_Party.csv")
EI_statewide = pd.read_csv("LA_statewide_rxc_pref.csv")
prec_ei_df = pd.read_csv("LA_prec_quant_counts.csv", dtype = {'CNTYVTD':'str'})
mean_prec_counts = pd.read_csv("LA_prec_means_counts.csv", dtype = {'CNTYVTD':'str'})
logit_params = pd.read_csv("LA_logit_params.csv")

#initialize state_gdf##################################################
state_gdf = gpd.read_file(plot_path)
state_gdf.columns = state_gdf.columns.str.replace("-", "_")

#replace cut-off candidate names from shapefile with full names
state_gdf_cols = list(state_gdf.columns)
cand1_index = state_gdf_cols.index('DeatonD_15')
cand2_index = state_gdf_cols.index('WolfeD_16P')
state_gdf_cols[cand1_index:cand2_index+1] = elec_columns
state_gdf.columns = state_gdf_cols
state_gdf['LandrieuI_19P_Governor'] = state_gdf['LandrieuI_19P_Governor'].astype(float)
state_df = pd.DataFrame(state_gdf)
state_df = state_df.drop(['geometry'], axis = 1)

##build graph from geo_dataframe#####################################
graph = Graph.from_geodataframe(state_gdf)
graph.add_data(state_gdf)
centroids = state_gdf.centroid
c_x = centroids.x
c_y = centroids.y
for node in graph.nodes():
    graph.nodes[node]["C_X"] = c_x[node]
    graph.nodes[node]["C_Y"] = c_y[node]

#set up elections data structures###########################################################
elections = list(elec_data["Election"]) 
elec_type = elec_data["Type"]
elec_cand_list = elec_columns

elecs_bool = ~elec_data.Election.isin(list(dropped_elecs))
elec_data_trunc = elec_data[elecs_bool].reset_index(drop = True)
elec_sets = list(set(elec_data_trunc["Election Set"]))
elections = list(elec_data_trunc["Election"])
general_elecs = list(elec_data_trunc[elec_data_trunc["Type"] == 'General'].Election)
primary_elecs = list(elec_data_trunc[elec_data_trunc["Type"] == 'Primary'].Election)

#this dictionary matches a specific election with the set it belongs to
elec_set_dict = {}
for elec_set in elec_sets:
    elec_set_df = elec_data_trunc[elec_data_trunc["Election Set"] == elec_set]
    elec_set_dict[elec_set] = dict(zip(elec_set_df.Type, elec_set_df.Election))
elec_match_dict = dict(zip(elec_data_trunc["Election"], elec_data_trunc["Election Set"]))


#make dictionary that maps an election to its candidates
candidates = {}
for elec in elections:
    cands = [y for y in elec_cand_list if elec in y]    
    candidates[elec] = dict(zip(list(range(len(cands))), cands))

cand_race_dict = cand_race_table.set_index("Candidates").to_dict()["Race"]
min_cand_weights_dict = {key:min_cand_weights.to_dict()[key][0] for key in  min_cand_weights.to_dict().keys()}     

########################################## pre-compute as much as possible for elections updater ##########
#precompute election recency weights W1 for all scores
elec_years = [elec_data_trunc.loc[elec_data_trunc["Election Set"] == elec_set, 'Year'].values[0].astype(str) \
              for elec_set in elec_sets]
recency_scores = [recency_weights[elec_year][0] for elec_year in elec_years]
recency_W1 = np.tile(recency_scores, (num_districts, 1)).transpose()
      
#precompute statewide EI and W1, W2, W3 for statewide/equal modes 
if record_statewide_modes:
    black_weight_state,  black_weight_equal, black_pref_cands_prim_state \
                     = precompute_state_weights(num_districts, elec_sets, elec_set_dict, recency_W1, EI_statewide, primary_elecs, \
                       elec_match_dict, min_cand_weights_dict, cand_race_dict)

#precompute set-up for district mode, if used (need precinct level EI set up)
#need to precompute all the column bases and dictionary for all (demog, election) pairs
if record_district_mode:             
    demogs = ['BCVAP']
    bases = {col.split('.')[0]+'.'+col.split('.')[1] for col in prec_ei_df.columns if col[:5] in demogs and 'ABSTAIN' not in col and \
          not any(x in col for x in general_elecs)}
    base_dict = {b:(b.split('.')[0], '_'.join(b.split('.')[1].split('_')[1:])) for b in bases}
    outcomes = {val:[] for val in base_dict.values()}
    for b in bases:
        outcomes[base_dict[b]].append(b) 
        
    precs = list(state_gdf[geo_id])
    prec_draws_outcomes = cand_pref_all_draws_outcomes(prec_ei_df, precs, bases, outcomes)

############################################################################################################       
#UPDATERS FOR CHAIN

#The elections model function (used as an updater). Takes in partition and returns Black-effectiveness likelihood per district 
def final_elec_model(partition):  
    """
    The output of the elections model is tke likelihood each distict is Black-effective:
    To compute this, each election set is first weighted
    by multiplying a recency weight (W1), "in-group"-minority-preference weight (W2) and 
    a preferred-candidate-confidence weight (W3).
    If the Black preferred candidate wins the election (set) a number of points equal to
    the set's weight is accrued. The ratio of the accrued points points to the total possible points
    is the raw Black -effectiviness score for the district. 
    
    Raw scores are adjusted by multiplying them by a "Group Control" factor,
    which measures the share of votes cast 
    for a minority-preferred candidate by the minority group itself.
    
    Finally, the adjusted Black-effectiveness score is fed through a logit function,
    transforming it into a probability that districts are Black-effective.
    
    We need to track several entities in the model, which will be dataframes or arrays,
    whose columns are districts and rows are election sets (or sometimes individual elections)
    These dataframes each store one of the following: Black preferred candidates (in the
    election set's primary), Black preferred candidates in runoffs, winners of primary,
    runoff and general elections, election winners, weights W1, W2 and W3, 
    and final election set weight for Black voters.
    """
    ###########################################################
    #only need to run model on two ReCom districts that have changed
    if partition.parent is not None:
        dict1 = dict(partition.parent.assignment)
        dict2 = dict(partition.assignment)
        differences = set([dict1[k] for k in dict1.keys() if dict1[k] != dict2[k]]).union(set([dict2[k] for k in dict2.keys() if dict1[k] != dict2[k]]))
        
        
    dist_changes = range(num_districts) if partition.parent is None else sorted(differences)
   
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

    ###########################################################################################
    #If we compute statewide modes: compute effectiveness probabilities for each district #################
    if record_statewide_modes:     
    #district probability distribution: statewide
        final_state_prob_dict = compute_final_dist(map_winners, black_pref_cands_prim_state, \
                                black_weight_state, dist_elec_results, dist_changes,\
                                cand_race_table, num_districts, candidates, elec_sets, elec_set_dict, \
                                "statewide", partition, logit_params, logit = True)
        
        #district probability distribution: equal
        final_equal_prob_dict = compute_final_dist(map_winners, black_pref_cands_prim_state, \
                                black_weight_equal, dist_elec_results, dist_changes,
                                cand_race_table, num_districts, candidates, elec_sets, elec_set_dict, \
                                "equal", partition, logit_params, logit = True)
    
        
    if record_district_mode: 
        black_weight_dist, black_pref_cands_prim_dist \
                                 = compute_district_weights(dist_changes, elec_sets, elec_set_dict, state_gdf, partition, prec_draws_outcomes,\
                                 geo_id, primary_elecs, elec_match_dict, bases, outcomes,\
                                 recency_W1, cand_race_dict, min_cand_weights_dict)
        
        final_dist_prob_dict = compute_final_dist(map_winners, black_pref_cands_prim_dist,
                               black_weight_dist, dist_elec_results, dist_changes,
                               cand_race_table, num_districts, candidates, elec_sets, elec_set_dict, \
                               'district', partition, logit_params, logit = True)

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
    """
    Given district effectiveness distributions, this function returns the total districts
    that are above the effectivness threshold for Black voters.
    """
    black_threshold = effectiveness_cutoff
    
    if "N/A" not in dictionary.values():
        black_effective = [i for i,j in dictionary.values()]
        black_effect_index = [i for i,n in enumerate(black_effective) if n >= black_threshold]        
        total_black_final = len(black_effect_index)
       
        return total_black_final
    else:
        return "N/A" 
                 
def demo_percents(partition): 
    black_pct = {k: partition["BCVAP"][k]/partition["CVAP"][k] for k in partition["BCVAP"].keys()}
    white_pct = {k: partition["WCVAP"][k]/partition["CVAP"][k] for k in partition["WCVAP"].keys()}
    return black_pct, white_pct

def num_cut_edges(partition):
    return len(partition["cut_edges"])

def num_county_splits(partition, df = state_gdf):
    df["current"] = df.index.map(partition.assignment.to_dict())
    return sum(df.groupby(county_split_id)['current'].nunique() > 1)

my_updaters = {
    "population": updaters.Tally(tot_pop, alias = "population"),
    "CVAP": updaters.Tally(CVAP, alias = "CVAP"),
    "WCVAP": updaters.Tally(WCVAP, alias = "WCVAP"),
    "HCVAP": updaters.Tally(HCVAP, alias = "HCVAP"),
    "BCVAP": updaters.Tally(BCVAP, alias = "BCVAP"),
    "Sum_CX": updaters.Tally(C_X, alias = "Sum_CX"),
    "Sum_CY": updaters.Tally(C_Y, alias = "Sum_CY"),
    "cut_edges": cut_edges,
    "demo_percents": demo_percents,
    "final_elec_model": final_elec_model,
    "num_county_splits": num_county_splits,
    "num_cut_edges": num_cut_edges
}

#updater functions
elections_track = [
    Election("PRES16", {"Democratic": 'ClintonD_16G_President' , "Republican": 'TrumpR_16G_President'}, alias = "PRES16"),
    Election("SEN16", {"Democratic": 'CampbellD_16G_US_Sen' , "Republican": 'KennedyR_16G_US_Sen'}, alias = "SEN16"), 
]

election_updaters = {election.name: election for election in elections_track}
my_updaters.update(election_updaters)

election_functions = [Election(j, candidates[j]) for j in elections]
election_updaters = {election.name: election for election in election_functions}
my_updaters.update(election_updaters)

#initial partition########################
total_population = state_gdf[tot_pop].sum()
ideal_population = total_population/num_districts
if start_map == 'new_seed':
    start_map = recursive_tree_part(graph, range(num_districts), ideal_population, tot_pop, pop_tol, 3)    

step_Num = 0
initial_partition = GeographicPartition(graph = graph, assignment = start_map, updaters = my_updaters)
proposal = partial(
    recom, pop_col=tot_pop, pop_target=ideal_population, epsilon= pop_tol, node_repeats=3
)


#constraints##############################################################
def inclusion(partition):
    final_state_prob, final_equal_prob, final_dist_prob = partition["final_elec_model"]
    inclusion_dict = final_state_prob if model_mode == 'statewide' else final_equal_prob if model_mode == 'equal' else final_dist_prob
    black_vra_dists = effective_districts(inclusion_dict)
    return black_vra_dists >= enacted_black

#acceptance functions #####################################
accept = accept.always_accept
    
#define chain
chain = MarkovChain(
    proposal = proposal,
    constraints = [constraints.within_percent_of_ideal_population(initial_partition, pop_tol), inclusion] \
            if ensemble_inclusion else [constraints.within_percent_of_ideal_population(initial_partition, pop_tol)],
    accept = accept, 
    initial_state = initial_partition,
    total_steps = total_steps
)

#prep storage for plans################################################
store_plans = pd.DataFrame(columns = ["Index", "GEOID" ])
store_plans["Index"] = list(initial_partition.assignment.keys())
state_gdf_geoid = state_gdf[[geo_id]]
store_plans["GEOID"] = [state_gdf_geoid.iloc[i][0] for i in store_plans["Index"]]
map_metric = pd.DataFrame(columns = ["B_state", "B_equal", "B_dist", \
                                     "Cut Edges", "County Splits"], index = list(range(store_interval)))

  #prep district-by-district storage (each metric in its own df)
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
black_prop_df = pd.DataFrame(columns = range(num_districts), index = list(range(store_interval)))
white_prop_df = pd.DataFrame(columns = range(num_districts), index = list(range(store_interval)))
  #partisan data storage
pres16_df = pd.DataFrame(columns = range(num_districts), index = list(range(store_interval)))
sen16_df = pd.DataFrame(columns = range(num_districts), index = list(range(store_interval)))
centroids_df = pd.DataFrame(columns = range(num_districts), index = list(range(store_interval)))

#run chain and collect data #############################################################################
count_moves = 0
best_score = 0
last_step_stored = 0
black_threshold = effectiveness_cutoff

start_time_total = time.time()
for step in chain:
    final_state_prob, final_equal_prob, final_dist_prob = step["final_elec_model"]     
    total_black_final_state = effective_districts(final_state_prob)
    total_black_final_equal = effective_districts(final_equal_prob)
    total_black_final_dist = effective_districts(final_dist_prob)
    
    map_metric.loc[step_Num] = [total_black_final_state, total_black_final_equal, \
                  total_black_final_equal, step["num_cut_edges"], step["num_county_splits"]]

    #saving at intervals
    if step_Num % store_interval == 0 and step_Num > 0:
        print("step", step_Num)
        store_plans.to_csv(DIR + "outputs/store_plans_{}.csv".format(run_name), index= False)
        
        #dump data and reset data frames
        if step_Num == store_interval:
            pres16_df.to_csv(DIR + "outputs/pres16_df_{}.csv".format(run_name), index = False)
            pres16_df = pd.DataFrame(columns = range(num_districts), index = list(range(store_interval)))
            sen16_df.to_csv(DIR + "outputs/sen16_df_{}.csv".format(run_name), index = False)
            sen16_df = pd.DataFrame(columns = range(num_districts), index = list(range(store_interval)))
             
            black_prop_df.to_csv(DIR + "outputs/black_prop_df_{}.csv".format(run_name), index = False)
            black_prop_df = pd.DataFrame(columns = range(num_districts), index = list(range(store_interval)))
            white_prop_df.to_csv(DIR + "outputs/white_prop_df_{}.csv".format(run_name), index = False)
            white_prop_df = pd.DataFrame(columns = range(num_districts), index = list(range(store_interval)))
                 
            for score_df, score_df_name in zip(score_dfs, score_df_names):
                score_df.to_csv(DIR + "outputs/{}_{}.csv".format(score_df_name,run_name), index= False)
                score_df = pd.DataFrame(columns = range(num_districts), index = list(range(store_interval)))           
           
            map_metric.to_csv(DIR + "outputs/map_metric_{}.csv".format(run_name), index = True)
        else:
            pres16_df.to_csv(DIR + "outputs/pres16_df_{}.csv".format(run_name), mode = 'a', header = False, index = False)
            pres16_df = pd.DataFrame(columns = range(num_districts), index = list(range(store_interval)))            
            sen16_df.to_csv(DIR + "outputs/sen16_df_{}.csv".format(run_name), mode = 'a', header = False, index = False)
            sen16_df = pd.DataFrame(columns = range(num_districts), index = list(range(store_interval)))
            
            black_prop_df.to_csv(DIR + "outputs/black_prop_df_{}.csv".format(run_name), mode = 'a', header = False, index = False)
            black_prop_df = pd.DataFrame(columns = range(num_districts), index = list(range(store_interval)))
            white_prop_df.to_csv(DIR + "outputs/white_prop_df_{}.csv".format(run_name), mode = 'a', header = False, index = False)
            white_prop_df = pd.DataFrame(columns = range(num_districts), index = list(range(store_interval)))
                    
            for score_df, score_df_name in zip(score_dfs, score_df_names):
                score_df.to_csv(DIR + "outputs/{}_{}.csv".format(score_df_name,run_name), mode = 'a', header = False, index= False)
                score_df = pd.DataFrame(columns = range(num_districts), index = list(range(store_interval)))           
            
            map_metric.to_csv(DIR + "outputs/map_metric_{}.csv".format(run_name), index = True)
    
    if step.parent is not None:
        if step.assignment != step.parent.assignment:
            count_moves += 1
            
    #district-by-district storage       
    black_prop_data = step["demo_percents"][0]
    keys = list(black_prop_data.keys())
    values = list(black_prop_data.values())
    black_prop_df.loc[step_Num % store_interval] = [value for _,value in sorted(zip(keys,values))]
    
    white_prop_data = step["demo_percents"][1]
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
        
    keys = list(percents["SEN16"].keys())
    values = list(percents["SEN16"].values())
    sen16_df.loc[step_Num % store_interval] = [value for _,value in sorted(zip(keys,values))]
       
    if record_statewide_modes:
        final_state_prob_df.loc[step_Num] = list(final_state_prob.values())                
        final_equal_prob_df.loc[step_Num] = list(final_equal_prob.values())               
   
    if record_district_mode:
        final_dist_prob_df.loc[step_Num] = list(final_dist_prob.values())                

    #store plans     
    if (step_Num - last_step_stored) == store_interval or step_Num == 0:          
        last_step_stored = step_Num
        store_plans["Map{}".format(step_Num)] = store_plans["Index"].map(dict(step.assignment))               
        print("stored new map!", "step num", step_Num)
   
    step_Num += 1

#output data
store_plans.to_csv(DIR + "outputs/store_plans_{}.csv".format(run_name), index= False)
black_prop_df.to_csv(DIR + "outputs/black_prop_df_{}.csv".format(run_name), mode = 'a', header = False, index= False)
white_prop_df.to_csv(DIR + "outputs/white_prop_df_{}.csv".format(run_name), mode = 'a', header = False, index= False)
pres16_df.to_csv(DIR + "outputs/pres16_df_{}.csv".format(run_name), mode = 'a', header = False, index = False)
sen16_df.to_csv(DIR + "outputs/sen16_df_{}.csv".format(run_name), mode = 'a', header = False, index = False)
map_metric.to_csv(DIR + "outputs/map_metric_{}.csv".format(run_name), index = True)
if total_steps <= store_interval:
    for score_df, score_df_name in zip(score_dfs, score_df_names):        
        score_df.to_csv(DIR + "outputs/{}_{}.csv".format(score_df_name, run_name), index= False)
else:  
    for score_df, score_df_name in zip(score_dfs, score_df_names):      
        score_df.to_csv(DIR + "outputs/{}_{}.csv".format(score_df_name, run_name), mode = 'a', header = False, index= False)
############# final print outs
print("--- %s TOTAL seconds ---" % (time.time() - start_time_total))
print("ave sec per step", (time.time() - start_time_total)/total_steps)
print("total moves", count_moves)
print("run name:", run_name)
print("num steps", total_steps)
print("current step", step_Num)

