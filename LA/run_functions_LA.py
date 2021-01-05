# -*- coding: utf-8 -*-
"""
Created on Tue Jul 28 15:50:57 2020

@author: darac
"""
import random
import networkx as nx
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
import collections
from enum import Enum
import re
import scipy
from scipy import stats
import time
import heapq
import operator

DIR = ''

def precompute_state_weights(num_districts, elec_sets, elec_set_dict, recency_W1, EI_statewide, primary_elecs, \
                             elec_match_dict, min_cand_weights_dict, cand_race_dict):
    #map data storage: set up all dataframes to be filled   
    black_pref_cands_prim_state = pd.DataFrame(columns = range(num_districts))
    black_pref_cands_prim_state["Election Set"] = elec_sets  
    
    #pre-compute W2 and W3 dfs for statewide/equal modes   
    black_ei_prob = [EI_statewide.loc[((EI_statewide["Election"] == elec_set_dict[elec_set]['Primary']) & \
                                       (EI_statewide["Demog"] == 'BCVAP')), "prob"].values[0] \
                                       for elec_set in elec_sets]
    
    black_ei_conf = [prob_conf_conversion(x) for x in black_ei_prob]    
    black_conf_W3_state = np.tile(black_ei_conf, (num_districts, 1)).transpose()
    
    for elec in primary_elecs:
        black_pref_cand = EI_statewide.loc[((EI_statewide["Election"] == elec) & (EI_statewide["Demog"] == 'BCVAP')), "Candidate"].values[0]       
        
        for district in range(num_districts):        
            black_pref_cands_prim_state.at[black_pref_cands_prim_state["Election Set"] == elec_match_dict[elec], district] = black_pref_cand
                  
    min_cand_black_W2_state = compute_W2(elec_sets, range(num_districts), min_cand_weights_dict, black_pref_cands_prim_state, cand_race_dict)

    #compute final election weights (for statewide and equal scores) by taking product of W1, W2, and W3 for each election set and district
    #Note: because these are statewide weights, an election set will have the same weight across districts
    black_weight_state = recency_W1*min_cand_black_W2_state*black_conf_W3_state
    #equal-score weights are all 1
    black_weight_equal = np.ones((len(elec_sets), num_districts))
        
    return black_weight_state, black_weight_equal, black_pref_cands_prim_state 

def compute_align_scores(dist_changes, elec_sets, state_gdf, partition, primary_elecs, \
                         black_pref_cands_prim, elec_match_dict, \
                         mean_prec_counts, geo_id):
       
    black_align_prim = np.empty((len(elec_sets),0), float)

    for district in dist_changes:        
        state_gdf["New Map"] = state_gdf.index.map(dict(partition.assignment))
        dist_prec_list =  list(state_gdf[state_gdf["New Map"] == district][geo_id])        
        cand_counts_dist = mean_prec_counts[mean_prec_counts[geo_id].isin(dist_prec_list)]
        
        black_align_dist = [sum(cand_counts_dist["BCVAP"+ '.' + black_pref_cand])/\
                            (sum(cand_counts_dist["BCVAP"+ '.' + black_pref_cand]) + \
                             sum(cand_counts_dist["WCVAP"+ '.' + black_pref_cand]) + \
                             sum(cand_counts_dist["OCVAP"+ '.' + black_pref_cand])) for \
                             black_pref_cand in black_pref_cands_prim[district]]
        
        black_align_prim = np.append(black_align_prim, np.array([black_align_dist]).transpose(), axis = 1)                
    
    return black_align_prim

def compute_district_weights(dist_changes, elec_sets, elec_set_dict, state_gdf, partition, prec_draws_outcomes,\
                             geo_id, primary_elecs, elec_match_dict, bases, outcomes,\
                             recency_W1, cand_race_dict, min_cand_weights_dict):
    black_pref_cands_prim_dist = pd.DataFrame(columns = dist_changes)
    black_pref_cands_prim_dist["Election Set"] = elec_sets
    black_conf_W3_dist = np.empty((len(elec_sets),0), float)
    
    for district in dist_changes:        
        state_gdf["New Map"] = state_gdf.index.map(dict(partition.assignment))
        dist_prec_list = list(state_gdf[state_gdf["New Map"] == district][geo_id])
        dist_prec_indices = state_gdf.index[state_gdf[geo_id].isin(dist_prec_list)].tolist()
        district_support_all = cand_pref_outcome_sum(prec_draws_outcomes, dist_prec_indices, bases, outcomes)
        
        black_pref_prob_single_dist = []
        
        for elec_set in elec_sets:
            BCVAP_support_elec = district_support_all[('BCVAP', elec_set_dict[elec_set]['Primary'])]
            black_pref_cand_dist = max(BCVAP_support_elec.items(), key=operator.itemgetter(1))[0]
            black_pref_prob_dist = BCVAP_support_elec[black_pref_cand_dist]
            
            black_pref_cands_prim_dist.at[black_pref_cands_prim_dist["Election Set"] == elec_set, district] = black_pref_cand_dist
            black_pref_prob_single_dist.append(black_pref_prob_dist)
            
        black_pref_conf_single_dist = [prob_conf_conversion(x) for x in black_pref_prob_single_dist]
        black_conf_W3_dist = np.append(black_conf_W3_dist, np.array([black_pref_conf_single_dist]).transpose(), axis = 1) 
                  
    #compute W2 ("in-group"-minority-preference weight)        
    min_cand_black_W2_dist = compute_W2(elec_sets, dist_changes, min_cand_weights_dict, black_pref_cands_prim_dist, cand_race_dict)
   
    ################################################################################    
    #compute final election weights per district
    recency_W1 = recency_W1.copy()[:, dist_changes]
    black_weight_dist = recency_W1*min_cand_black_W2_dist*black_conf_W3_dist
    
    return black_weight_dist, black_pref_cands_prim_dist
           
           
def prob_conf_conversion(cand_prob):
    #parameters chosen to be 0-ish confidence until 50% then rapid ascenion to high confidence
    cand_conf = 1/(1+np.exp(18-26*cand_prob))    
    return cand_conf

def compute_final_dist(map_winners, black_pref_cands_df,
                 black_weight_array, dist_elec_results, dist_changes,
                 cand_race_table, num_districts, candidates, \
                 elec_sets, elec_set_dict, black_align_prim, \
                 mode, logit_params, logit = False):
    #determine if election set accrues points by district for black 
    primary_winners = map_winners[map_winners["Election Type"] == 'Primary'].reset_index(drop = True)
    general_winners = map_winners[map_winners["Election Type"] == 'General'].reset_index(drop = True)
           
    black_pref_wins = np.empty((len(elec_sets),0), float)

    primary_second_df = pd.DataFrame(columns = range(num_districts))
    primary_second_df["Election Set"] = elec_sets
    
    primary_races = [elec_set_dict[elec_set]["Primary"] for elec_set in elec_sets]
    cand_party_dict = cand_race_table.set_index("Candidates").to_dict()["Party"]

    for dist in dist_changes:
        black_pref_cands = list(black_pref_cands_df[dist])
        primary_dict = primary_winners.set_index("Election Set").to_dict()[dist]
        primary_winner_list = [primary_dict[es] for es in elec_sets]
                
        general_dict = general_winners.set_index("Election Set").to_dict()[dist]
        general_winner_list = ["N/A" if es not in list(general_winners["Election Set"]) \
        else general_dict[es] for es in elec_sets]
        
        primary_race_share_dict = {primary_race:dist_elec_results[primary_race][dist] for primary_race in primary_races}
        primary_ranking = {primary_race:{key: rank for rank, key in \
                           enumerate(sorted(primary_race_share_dict[primary_race], \
                           key=primary_race_share_dict[primary_race].get, reverse=True), 1)} \
                                            for primary_race in primary_race_share_dict.keys()} 

        second_place_primary = {primary_race: [cand for cand, value in primary_ranking[primary_race].items() \
                                               if primary_ranking[primary_race][cand] == 2] for primary_race in primary_races}

        primary_second_df[dist] = [second_place_primary[key][0] for key in second_place_primary.keys()]
        
        black_pref_prim_rank = [primary_ranking[pr][bpc] for pr, bpc in zip(primary_races, black_pref_cands)]     
        party_general_winner = [cand_party_dict[gw] if gw in cand_party_dict.keys() else None for gw in general_winner_list]
                  
        #winning conditions (conditions to accrue points for election set/minority group):
        black_accrue = [(prim_win == bpc and party_win == 'D') if 'President' in prim_race else \
                        ((primary_race_share_dict[prim_race][bpc]) > .5 or (bpp_rank < 3 and party_win == 'D') or (bpp_rank == 1 and party_win == None))
                        for bpc, prim_win, party_win, bpp_rank, prim_race \
                        in zip(black_pref_cands, primary_winner_list, party_general_winner, \
                               black_pref_prim_rank, primary_races)]        
        black_pref_wins = np.append(black_pref_wins, np.array([black_accrue]).transpose(), axis = 1)
    

    #extract 2 dist_change columns from election weight array
    if len(black_weight_array[0]) > 2: 
        black_weight_array = black_weight_array[:, dist_changes]
    
    black_points_accrued = black_weight_array*black_pref_wins    
########################################################################################
    #Compute district probabilities: black, Latino, neither and overlap 
    black_vra_prob = list(np.sum(black_points_accrued*black_align_prim, axis = 0)/np.sum(black_weight_array, axis = 0))             
    
    #feed through logit:
    if logit == True:
        logit_coef_black = logit_params.loc[(logit_params['model_type'] == mode) & (logit_params['subgroup'] == 'Black'), 'coef'].values[0]
        logit_intercept_black = logit_params.loc[(logit_params['model_type'] == mode) & (logit_params['subgroup'] == 'Black'), 'intercept'].values[0]       
        black_vra_prob = [1/(1+np.exp(-(logit_coef_black*y+logit_intercept_black))) for y in black_vra_prob]
   
    not_effect_vra_prob = [1-i for i in black_vra_prob]
    
    return dict(zip(dist_changes, zip(black_vra_prob, not_effect_vra_prob)))
    
 
def compute_W2(elec_sets, districts, min_cand_weights_dict, black_pref_cands_df,\
               cand_race_dict):
    
    min_cand_black_W2 = np.empty((len(elec_sets),0), float)
    for dist in districts:
        black_pref = list(black_pref_cands_df[dist])
        
        black_pref_race = [cand_race_dict[bp] for bp in black_pref]
        black_cand_weight = [min_cand_weights_dict["Relevant Minority"] if "Black" in bpr else \
                             min_cand_weights_dict["Other"] for bpr in black_pref_race]
        
        min_cand_black_W2 = np.append(min_cand_black_W2, np.array([black_cand_weight]).transpose(), axis = 1)
    
    return min_cand_black_W2


#to aggregrate precinct EI to district EI for district model mode
def cand_pref_all_draws_outcomes(prec_quant_df, precs, bases, outcomes, sample_size = 1000 ):
    quant_vals = np.array([0,125,250,375,500,625,750,875,1000])
    draws = {}
    for outcome in outcomes.keys():
        draw_base_list = []
        for base in outcomes[outcome]:
            dist_prec_quant = prec_quant_df.copy()
            vec_rand = np.random.rand(sample_size,len(dist_prec_quant))
            vec_rand_shift = np.array(dist_prec_quant[base +'.'+ '0'])+ sum(np.minimum(np.maximum(vec_rand-quant_vals[qv]/1000,0),.125)*8*np.array(dist_prec_quant[base + '.' +  str(quant_vals[qv+1])]-dist_prec_quant[base + '.'+ str(quant_vals[qv])]) for qv in range(len(quant_vals)-1))
            draw_base_list.append(vec_rand_shift.astype('float32').T)
        draws[outcome] = np.transpose(np.stack(draw_base_list),(1,0,2))
    return draws

def cand_pref_outcome_sum(prec_draws_outcomes, dist_prec_indices, bases, outcomes):
    dist_draws = {}
    for outcome in outcomes:
        summed_outcome = prec_draws_outcomes[outcome][dist_prec_indices].sum(axis=0)
        unique, counts = np.unique(np.argmax(summed_outcome, axis=0), return_counts=True)
       
        prefs = {x.split('.')[1].split('_counts')[0]:0.0 for x in outcomes[outcome]}
        prefs_counts = dict(zip(unique, counts))
        prefs.update({outcomes[outcome][key].split('.')[1].split('_counts')[0]: prefs_counts[key]/len(summed_outcome[0]) for key in prefs_counts.keys()})
        dist_draws[outcome] = prefs
    return dist_draws
