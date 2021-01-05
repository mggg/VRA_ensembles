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
                             runoff_elecs, elec_match_dict, min_cand_weights_dict, cand_race_dict):
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
   
    black_ei_prob = [EI_statewide.loc[((EI_statewide["Election"] == elec_set_dict[elec_set]['Primary']) & \
                                       (EI_statewide["Demog"] == 'BCVAP')), "prob"].values[0] \
                                       for elec_set in elec_sets]
    
    black_ei_conf = [prob_conf_conversion(x) for x in black_ei_prob]    
    black_conf_W3_state = np.tile(black_ei_conf, (num_districts, 1)).transpose()
    
    hisp_ei_prob = [EI_statewide.loc[((EI_statewide["Election"] == elec_set_dict[elec_set]['Primary']) & 
                                       (EI_statewide["Demog"] == 'HCVAP')), "prob"].values[0] 
                                       for elec_set in elec_sets]
    
    hisp_ei_conf = [prob_conf_conversion(x) for x in hisp_ei_prob]    
    hisp_conf_W3_state = np.tile(hisp_ei_conf, (num_districts, 1)).transpose()
    
    neither_ei_conf = [prob_conf_conversion(x*y) for x,y in zip(black_ei_prob, hisp_ei_prob)]
    neither_conf_W3_state = np.tile(neither_ei_conf, (num_districts, 1)).transpose()
    
    #pre-compute W2 and W3 dfs for statewide/equal modes   
    for elec in primary_elecs + runoff_elecs:
        black_pref_cand = EI_statewide.loc[((EI_statewide["Election"] == elec) & (EI_statewide["Demog"] == 'BCVAP')), "Candidate"].values[0]
        hisp_pref_cand = EI_statewide.loc[((EI_statewide["Election"] == elec) & (EI_statewide["Demog"] == 'HCVAP')), "Candidate"].values[0]
      
        for district in range(num_districts):
            if elec in primary_elecs:           
                black_pref_cands_prim_state.at[black_pref_cands_prim_state["Election Set"] == elec_match_dict[elec], district] = black_pref_cand              
                hisp_pref_cands_prim_state.at[hisp_pref_cands_prim_state["Election Set"] == elec_match_dict[elec], district] = hisp_pref_cand
              
            else:                        
                black_pref_cands_runoffs_state.at[black_pref_cands_runoffs_state["Election Set"] == elec_match_dict[elec], district] = black_pref_cand       
                hisp_pref_cands_runoffs_state.at[hisp_pref_cands_runoffs_state["Election Set"] == elec_match_dict[elec], district] = hisp_pref_cand
                  
    min_cand_black_W2_state, min_cand_hisp_W2_state, min_cand_neither_W2_state = compute_W2(elec_sets, \
                  range(num_districts), min_cand_weights_dict, black_pref_cands_prim_state, hisp_pref_cands_prim_state, cand_race_dict)

    
    #compute final election weights (for statewide and equal scores) by taking product of W1, W2, and W3 for each election set and district
    #Note: because these are statewide weights, an election set will have the same weight across districts
    black_weight_state = recency_W1*min_cand_black_W2_state*black_conf_W3_state
    hisp_weight_state = recency_W1*min_cand_hisp_W2_state*hisp_conf_W3_state    
    neither_weight_state = recency_W1*min_cand_neither_W2_state*neither_conf_W3_state
    
    #equal-score weights are all 1
    black_weight_equal = np.ones((len(elec_sets), num_districts))
    hisp_weight_equal = np.ones((len(elec_sets), num_districts))
    neither_weight_equal = np.ones((len(elec_sets), num_districts))
    
    return black_weight_state, hisp_weight_state, neither_weight_state, black_weight_equal,\
        hisp_weight_equal, neither_weight_equal, black_pref_cands_prim_state, hisp_pref_cands_prim_state, \
        black_pref_cands_runoffs_state, hisp_pref_cands_runoffs_state

def compute_district_weights(dist_changes, elec_sets, elec_set_dict, state_gdf, partition, prec_draws_outcomes,\
                             geo_id, primary_elecs, runoff_elecs, elec_match_dict, bases, outcomes,\
                             recency_W1, cand_race_dict, min_cand_weights_dict):
   
    black_pref_cands_prim_dist = pd.DataFrame(columns = dist_changes)
    black_pref_cands_prim_dist["Election Set"] = elec_sets
    hisp_pref_cands_prim_dist = pd.DataFrame(columns = dist_changes)
    hisp_pref_cands_prim_dist["Election Set"] = elec_sets
    #store runoff preferences for instances where minority-preferred candidate needs to switch between primary and runoff
    black_pref_cands_runoffs_dist = pd.DataFrame(columns = dist_changes)
    black_pref_cands_runoffs_dist["Election Set"] = elec_sets
    hisp_pref_cands_runoffs_dist = pd.DataFrame(columns = dist_changes)
    hisp_pref_cands_runoffs_dist["Election Set"] = elec_sets 
    
    black_conf_W3_dist = np.empty((len(elec_sets),0), float)
    hisp_conf_W3_dist = np.empty((len(elec_sets),0), float)
    neither_conf_W3_dist = np.empty((len(elec_sets),0), float)
    
    for district in dist_changes:        
        state_gdf["New Map"] = state_gdf.index.map(dict(partition.assignment))
        dist_prec_list =  list(state_gdf[state_gdf["New Map"] == district][geo_id])
        dist_prec_indices = state_gdf.index[state_gdf[geo_id].isin(dist_prec_list)].tolist()
        district_support_all = cand_pref_outcome_sum(prec_draws_outcomes, dist_prec_indices, bases, outcomes)
        
        black_pref_prob_single_dist = []
        hisp_pref_prob_single_dist = []
        
        for elec_set in elec_sets:                    
            HCVAP_support_elec = district_support_all[('HCVAP', elec_set_dict[elec_set]['Primary'])]
            hisp_pref_cand_dist = max(HCVAP_support_elec.items(), key=operator.itemgetter(1))[0]
            hisp_pref_prob_dist = HCVAP_support_elec[hisp_pref_cand_dist]
            hisp_pref_prob_single_dist.append(hisp_pref_prob_dist)   
            
            BCVAP_support_elec = district_support_all[('BCVAP', elec_set_dict[elec_set]['Primary'])]
            black_pref_cand_dist = max(BCVAP_support_elec.items(), key=operator.itemgetter(1))[0]
            black_pref_prob_dist = BCVAP_support_elec[black_pref_cand_dist]
            black_pref_prob_single_dist.append(black_pref_prob_dist)
            
            black_pref_cands_prim_dist.at[black_pref_cands_prim_dist["Election Set"] == elec_set, district] = black_pref_cand_dist
            hisp_pref_cands_prim_dist.at[hisp_pref_cands_prim_dist["Election Set"] == elec_set, district] = hisp_pref_cand_dist      
            
            if 'Runoff' in elec_set_dict[elec_set].keys():
                HCVAP_support_elec = district_support_all[('HCVAP', elec_set_dict[elec_set]['Runoff'])]
                hisp_pref_cand_dist = max(HCVAP_support_elec.items(), key=operator.itemgetter(1))[0]
                hisp_pref_cands_runoffs_dist.at[hisp_pref_cands_runoffs_dist["Election Set"] == elec_set, district] = hisp_pref_cand_dist
           
                BCVAP_support_elec = district_support_all[('BCVAP', elec_set_dict[elec_set]['Runoff'])]
                black_pref_cand_dist = max(BCVAP_support_elec.items(), key=operator.itemgetter(1))[0]                               
                black_pref_cands_runoffs_dist.at[black_pref_cands_runoffs_dist["Election Set"] == elec_set, district] = black_pref_cand_dist
        
        black_pref_conf_single_dist = [prob_conf_conversion(x) for x in black_pref_prob_single_dist]
        black_conf_W3_dist = np.append(black_conf_W3_dist, np.array([black_pref_conf_single_dist]).transpose(), axis = 1) 
        
        hisp_pref_conf_single_dist = [prob_conf_conversion(x) for x in hisp_pref_prob_single_dist]
        hisp_conf_W3_dist = np.append(hisp_conf_W3_dist, np.array([hisp_pref_conf_single_dist]).transpose(), axis = 1) 
                
        neither_pref_conf_single_dist = [prob_conf_conversion(x*y) for x,y in zip(black_pref_prob_single_dist,hisp_pref_prob_single_dist)]
        neither_conf_W3_dist = np.append(neither_conf_W3_dist, np.array([neither_pref_conf_single_dist]).transpose(), axis = 1) 
                       
    #compute W2 ("in-group"-minority-preference weight)        
    min_cand_black_W2_dist, min_cand_hisp_W2_dist, min_cand_neither_W2_dist = compute_W2(elec_sets, \
          dist_changes, min_cand_weights_dict, black_pref_cands_prim_dist, hisp_pref_cands_prim_dist, cand_race_dict)
    ################################################################################    
    #compute final election weights per district
    recency_W1 = recency_W1.copy()[:, dist_changes]
    black_weight_dist = recency_W1*min_cand_black_W2_dist*black_conf_W3_dist
    hisp_weight_dist = recency_W1*min_cand_hisp_W2_dist*hisp_conf_W3_dist    
    neither_weight_dist = recency_W1*min_cand_neither_W2_dist*neither_conf_W3_dist
        
    return black_weight_dist, hisp_weight_dist, neither_weight_dist, black_pref_cands_prim_dist, \
            black_pref_cands_runoffs_dist, hisp_pref_cands_prim_dist, hisp_pref_cands_runoffs_dist
                                     

def compute_align_scores(dist_changes, elec_sets, state_gdf, partition, primary_elecs, \
                         black_pref_cands_prim, hisp_pref_cands_prim, elec_match_dict, \
                         mean_prec_counts, geo_id):
    
    black_align_prim = np.empty((len(elec_sets),0), float)
    hisp_align_prim = np.empty((len(elec_sets),0), float)
    
    for district in dist_changes:        
        state_gdf["New Map"] = state_gdf.index.map(dict(partition.assignment))
        dist_prec_list =  list(state_gdf[state_gdf["New Map"] == district][geo_id])        
        cand_counts_dist = mean_prec_counts[mean_prec_counts[geo_id].isin(dist_prec_list)]
                
        black_align_dist = [sum(cand_counts_dist["BCVAP"+ '.' + black_pref_cand])/\
                            (sum(cand_counts_dist["BCVAP"+ '.' + black_pref_cand]) + \
                             sum(cand_counts_dist["HCVAP"+ '.' + black_pref_cand]) + \
                             sum(cand_counts_dist["WCVAP"+ '.' + black_pref_cand]) + \
                             sum(cand_counts_dist["OCVAP"+ '.' + black_pref_cand])) for \
                             black_pref_cand in black_pref_cands_prim[district]]
        
        hisp_align_dist = [sum(cand_counts_dist["HCVAP"+ '.' + hisp_pref_cand])/\
                            (sum(cand_counts_dist["BCVAP"+ '.' + hisp_pref_cand]) + \
                             sum(cand_counts_dist["HCVAP"+ '.' + hisp_pref_cand]) + \
                             sum(cand_counts_dist["WCVAP"+ '.' + hisp_pref_cand]) + \
                             sum(cand_counts_dist["OCVAP"+ '.' + hisp_pref_cand])) for \
                             hisp_pref_cand in hisp_pref_cands_prim[district]]
        
        black_align_prim = np.append(black_align_prim, np.array([black_align_dist]).transpose(), axis = 1)                
        hisp_align_prim = np.append(hisp_align_prim, np.array([hisp_align_dist]).transpose(), axis = 1)                
    
    return black_align_prim, hisp_align_prim   
     
    
def prob_conf_conversion(cand_prob):
    #parameters chosen to be ~0 confidence until 50% then rapid ascension to confidence ~ 1
    cand_conf = 1/(1+np.exp(18-26*cand_prob))    
    return cand_conf

def compute_final_dist(map_winners, black_pref_cands_df, black_pref_cands_runoffs,\
                 hisp_pref_cands_df, hisp_pref_cands_runoffs, neither_weight_array, \
                 black_weight_array, hisp_weight_array, dist_elec_results, dist_changes,
                 cand_race_table, num_districts, candidates, \
                 elec_sets, elec_set_dict, black_align_prim, hisp_align_prim, \
                 mode, logit_params, logit = False):
    #determine if election set accrues points by district for Black and Latino voters
    general_winners = map_winners[map_winners["Election Type"] == 'General'].reset_index(drop = True)
    primary_winners = map_winners[map_winners["Election Type"] == 'Primary'].reset_index(drop = True)
    runoff_winners = map_winners[map_winners["Election Type"] == 'Runoff'].reset_index(drop = True)
        
    black_pref_wins = np.empty((len(elec_sets),0), float)
    hisp_pref_wins = np.empty((len(elec_sets),0), float)
    
    primary_second_df = pd.DataFrame(columns = range(num_districts))
    primary_second_df["Election Set"] = elec_sets
    
    prim_share_hpc = pd.DataFrame(columns = range(num_districts))
    prim_share_hpc["Election Set"] = elec_sets
    prim_share_bpc = pd.DataFrame(columns = range(num_districts))
    prim_share_bpc["Election Set"] = elec_sets
    party_gen_winner = pd.DataFrame(columns = range(num_districts))
    party_gen_winner["Election Set"] = elec_sets
    
    primary_races = [elec_set_dict[elec_set]["Primary"] for elec_set in elec_sets]
    runoff_races = [None if 'Runoff' not in elec_set_dict[elec_set].keys() else elec_set_dict[elec_set]["Runoff"] for elec_set in elec_sets]
    cand_party_dict = cand_race_table.set_index("Candidates").to_dict()["Party"]

    
    for dist in dist_changes:
        black_pref_cands = list(black_pref_cands_df[dist])
        hisp_pref_cands = list(hisp_pref_cands_df[dist])
        
        primary_dict = primary_winners.set_index("Election Set").to_dict()[dist]
        general_dict = general_winners.set_index("Election Set").to_dict()[dist]
        runoffs_dict = runoff_winners.set_index("Election Set").to_dict()[dist]
        primary_winner_list = [primary_dict[es] for es in elec_sets]
        general_winner_list = [general_dict[es] for es in elec_sets]
        runoff_winner_list = ["N/A" if es not in list(runoff_winners["Election Set"]) \
        else runoffs_dict[es] for es in elec_sets]
        
        primary_race_share_dict = {primary_race:dist_elec_results[primary_race][dist] for primary_race in primary_races}
        primary_ranking = {primary_race:{key: rank for rank, key in \
                           enumerate(sorted(primary_race_share_dict[primary_race], \
                           key=primary_race_share_dict[primary_race].get, reverse=True), 1)} \
                                            for primary_race in primary_race_share_dict.keys()} 

        second_place_primary = {primary_race: [cand for cand, value in primary_ranking[primary_race].items() \
                                               if primary_ranking[primary_race][cand] == 2] for primary_race in primary_races}

        primary_second_df[dist] = [second_place_primary[key][0] for key in second_place_primary.keys()]
        
        black_pref_prim_rank = [primary_ranking[pr][bpc] for pr, bpc in zip(primary_races, black_pref_cands)]
        hisp_pref_prim_rank = [primary_ranking[pr][hpc] for pr, hpc in zip(primary_races, hisp_pref_cands)]
        
        prim_share_hpc[dist] = [primary_race_share_dict[prim_race][hpc] for prim_race,hpc in zip(primary_races, hisp_pref_cands)]
        prim_share_bpc[dist] = [primary_race_share_dict[prim_race][bpc] for prim_race,bpc in zip(primary_races, black_pref_cands)]
        party_general_winner = [cand_party_dict[gw] for gw in general_winner_list]
        party_gen_winner[dist] = party_general_winner
        
        #we always care who preferred candidate is in runoff if the minority preferred primary
        #candidate wins in district primary
        runoff_black_pref = ["N/A" if rw == "N/A" else \
                     bpc for rw,bpc in zip(runoff_winner_list, list(black_pref_cands_runoffs[dist]))]

        runoff_hisp_pref = ["N/A" if rw == "N/A" else \
                     hpc for rw,hpc in zip(runoff_winner_list, list(hisp_pref_cands_runoffs[dist]))]               
        #winning conditions (conditions to accrue points for election set/minority group):

        black_accrue = [(prim_win == bpc and party_win == 'D') if run_race == None else \
                        ((bpp_rank < 3 and run_win == runbp and party_win == 'D') or \
                        (primary_race_share_dict[prim_race][bpc] > .5 and party_win == 'D')) \
                        for run_race, prim_win, bpc, party_win, bpp_rank, run_win, runbp, prim_race \
                        in zip(runoff_races, primary_winner_list,black_pref_cands, \
                        party_general_winner, black_pref_prim_rank,runoff_winner_list, \
                        runoff_black_pref, primary_races)]
        
        black_pref_wins = np.append(black_pref_wins, np.array([black_accrue]).transpose(), axis = 1)

        hisp_accrue = [(prim_win == hpc and party_win == 'D') if run_race == None else \
                       ((hpp_rank < 3 and run_win == runhp and party_win == 'D') or \
                       (primary_race_share_dict[prim_race][hpc] > .5 and party_win == 'D'))\
                       for run_race, prim_win, hpc, party_win, hpp_rank, run_win, runhp, \
                       prim_race in zip(runoff_races, primary_winner_list,hisp_pref_cands, \
                       party_general_winner, hisp_pref_prim_rank,runoff_winner_list, \
                       runoff_hisp_pref, primary_races)]
                       
        hisp_pref_wins = np.append(hisp_pref_wins, np.array([hisp_accrue]).transpose(), axis = 1)
               
    neither_pref_wins = (1-black_pref_wins)*(1-hisp_pref_wins)
    
    if len(black_weight_array[0]) > 2: 
        black_weight_array = black_weight_array[:, dist_changes]
        hisp_weight_array = hisp_weight_array[:, dist_changes]
        neither_weight_array = neither_weight_array[:, dist_changes]
        
    #election set weight's number of points are accrued if Black or Latino preferred candidate(s) win (or proxies do)
    neither_points_accrued = neither_weight_array*neither_pref_wins
    black_points_accrued = black_weight_array*black_pref_wins
    hisp_points_accrued = hisp_weight_array*hisp_pref_wins    
 
########################################################################################
    #Compute district probabilities: Black, Latino, Neither and Overlap 
    black_vra_prob = list(np.sum(black_points_accrued*black_align_prim, axis = 0)/np.sum(black_weight_array, axis = 0))             
    hisp_vra_prob = list(np.sum(hisp_points_accrued*hisp_align_prim, axis = 0)/np.sum(hisp_weight_array, axis = 0))             
    neither_vra_prob = list(np.sum(neither_points_accrued, axis = 0)/np.sum(neither_weight_array, axis = 0))             

    #feed through logit:
    if logit == True:
        logit_coef_black = logit_params.loc[(logit_params['model_type'] == mode) & (logit_params['subgroup'] == 'Black'), 'coef'].values[0]
        logit_intercept_black = logit_params.loc[(logit_params['model_type'] == mode) & (logit_params['subgroup'] == 'Black'), 'intercept'].values[0]
        logit_coef_hisp = logit_params.loc[(logit_params['model_type'] == mode) & (logit_params['subgroup'] == 'Latino'), 'coef'].values[0]
        logit_intercept_hisp = logit_params.loc[(logit_params['model_type'] == mode) & (logit_params['subgroup'] == 'Latino'), 'intercept'].values[0]
        logit_coef_neither = logit_params.loc[(logit_params['model_type'] == mode) & (logit_params['subgroup'] == 'Neither'), 'coef'].values[0]
        logit_intercept_neither = logit_params.loc[(logit_params['model_type'] == mode) & (logit_params['subgroup'] == 'Neither'), 'intercept'].values[0]
                
        black_vra_prob = [1/(1+np.exp(-(logit_coef_black*y+logit_intercept_black))) for y in black_vra_prob]
        hisp_vra_prob = [1/(1+np.exp(-(logit_coef_hisp*y+logit_intercept_hisp))) for y in hisp_vra_prob]
        neither_vra_prob = [1/(1+np.exp(-(logit_coef_neither*y+logit_intercept_neither))) for y in neither_vra_prob]
    
    min_neither = [0 if (black_vra_prob[i] + hisp_vra_prob[i]) > 1 else 1 -(black_vra_prob[i] + hisp_vra_prob[i]) for i in range(len(dist_changes))]
    max_neither = [1 - max(black_vra_prob[i], hisp_vra_prob[i]) for i in range(len(dist_changes))]
    
    #uses ven diagram overlap/neither method 
    final_neither = [min_neither[i] if neither_vra_prob[i] < min_neither[i] else max_neither[i] \
                     if neither_vra_prob[i] > max_neither[i] else neither_vra_prob[i] for i in range(len(dist_changes))]    
    final_overlap = [final_neither[i] + black_vra_prob[i] + hisp_vra_prob[i] - 1 for i in range(len(dist_changes))]
    final_black_prob = [black_vra_prob[i] - final_overlap[i] for i in range(len(dist_changes))]
    final_hisp_prob = [hisp_vra_prob[i] - final_overlap[i] for i in range(len(dist_changes))]
    
    #when fitting logit, comment in:
#    final_neither = neither_vra_prob
#    final_overlap = ["N/A"]*len(dist_changes)
#    final_black_prob = black_vra_prob #[black_vra_prob[i] - final_overlap[i] for i in range(len(dist_changes))]
#    final_hisp_prob = hisp_vra_prob

    return dict(zip(dist_changes, zip(final_hisp_prob, final_black_prob, final_neither, final_overlap)))
    
 
def compute_W2(elec_sets, districts, min_cand_weights_dict, black_pref_cands_df, hisp_pref_cands_df, \
               cand_race_dict):
    
    min_cand_black_W2 = np.empty((len(elec_sets),0), float)
    min_cand_hisp_W2 = np.empty((len(elec_sets),0), float)
    min_cand_neither_W2 = np.empty((len(elec_sets),0), float)
    
    for dist in districts:
        black_pref = list(black_pref_cands_df[dist])

        black_pref_race = [cand_race_dict[bp] for bp in black_pref]
        black_cand_weight = [min_cand_weights_dict["Relevant Minority"] if "Black" in bpr else \
                             min_cand_weights_dict["Other"] for bpr in black_pref_race]
        min_cand_black_W2 = np.append(min_cand_black_W2, np.array([black_cand_weight]).transpose(), axis = 1)
        
        hisp_pref = list(hisp_pref_cands_df[dist])
        hisp_pref_race = [cand_race_dict[hp] for hp in hisp_pref]
        hisp_cand_weight = [min_cand_weights_dict["Relevant Minority"] if "Hispanic" in hpr else \
                             min_cand_weights_dict["Other"] for hpr in hisp_pref_race]
        min_cand_hisp_W2 = np.append(min_cand_hisp_W2, np.array([hisp_cand_weight]).transpose(), axis = 1)
       
        neither_cand_weight = [min_cand_weights_dict['Relevant Minority'] if ('Hispanic' in hpr and 'Black' in bpr) else\
        min_cand_weights_dict['Other'] if ('Hispanic' not in hpr and 'Black' not in bpr) else \
           min_cand_weights_dict['Partial '] for bpr,hpr in zip(black_pref_race, hisp_pref_race)]
        min_cand_neither_W2 = np.append(min_cand_neither_W2, np.array([neither_cand_weight]).transpose(), axis = 1)
    
        
    return min_cand_black_W2, min_cand_hisp_W2, min_cand_neither_W2


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
