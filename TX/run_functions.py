# -*- coding: utf-8 -*-
"""
Created on Tue Jul 28 15:50:57 2020

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
import collections
from enum import Enum
import re
import statsmodels.formula.api as smf
import statsmodels.api as sm
import scipy
from scipy import stats
import time
import heapq
import operator

DIR = ''

def prob_conf_conversion(cand_prob):
    #parameters chosen to be 0-ish confidence until 50% then rapid ascenion to high confidence
    cand_conf = 1/(1+np.exp(18-26*cand_prob))    
    return cand_conf

def compute_final_dist(map_winners, black_pref_cands_df, black_pref_cands_runoffs,\
                 hisp_pref_cands_df, hisp_pref_cands_runoffs, neither_weight_df, \
                 black_weight_df, hisp_weight_df, dist_elec_results, dist_changes,
                 cand_race_table, num_districts, candidates, \
                 elec_sets, elec_set_dict, black_align_prim, hisp_align_prim, \
                 mode, logit_params, logit = False, single_map = False):
    #determine if election set accrues points by district for black and Latino voters
    general_winners = map_winners[map_winners["Election Type"] == 'General'].reset_index(drop = True)
    primary_winners = map_winners[map_winners["Election Type"] == 'Primary'].reset_index(drop = True)
    runoff_winners = map_winners[map_winners["Election Type"] == 'Runoff'].reset_index(drop = True)
        
    black_pref_wins = pd.DataFrame(columns = range(num_districts))
    black_pref_wins["Election Set"] = elec_sets
    hisp_pref_wins = pd.DataFrame(columns = range(num_districts))
    hisp_pref_wins["Election Set"] = elec_sets
    
    primary_second_df = pd.DataFrame(columns = range(num_districts))
    primary_second_df["Election Set"] = elec_sets
    
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
        
        party_general_winner = [cand_party_dict[gw] for gw in general_winner_list]
        
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
        
        black_pref_wins[dist] = black_accrue

        hisp_accrue = [(prim_win == hpc and party_win == 'D') if run_race == None else \
                       ((hpp_rank < 3 and run_win == runhp and party_win == 'D') or \
                       (primary_race_share_dict[prim_race][hpc] > .5 and party_win == 'D'))\
                       for run_race, prim_win, hpc, party_win, hpp_rank, run_win, runhp, \
                       prim_race in zip(runoff_races, primary_winner_list,hisp_pref_cands, \
                       party_general_winner, hisp_pref_prim_rank,runoff_winner_list, \
                       runoff_hisp_pref, primary_races)]
                       
        hisp_pref_wins[dist] = hisp_accrue
        
        
    neither_pref_wins = (1-black_pref_wins.drop(['Election Set'], axis = 1))*(1-hisp_pref_wins.drop(['Election Set'], axis = 1))
    neither_pref_wins["Election Set"] = elec_sets
    #election weight's number of points are accrued if black or latino preferred candidate(s) win (or proxies do)
    neither_points_accrued = neither_weight_df.drop(['Election Set'], axis = 1)*neither_pref_wins.drop(['Election Set'], axis = 1)  
    neither_points_accrued["Election Set"] = elec_sets
    black_points_accrued = black_weight_df.drop(['Election Set'], axis = 1)*black_pref_wins.drop(['Election Set'], axis = 1)  
    black_points_accrued["Election Set"] = elec_sets
    hisp_points_accrued = hisp_weight_df.drop(['Election Set'], axis = 1)*hisp_pref_wins.drop(['Election Set'], axis = 1)      
    hisp_points_accrued["Election Set"] = elec_sets
    
########################################################################################
    #Compute district probabilities: black, Latino, neither and overlap 
    black_vra_prob = [0 if sum(black_weight_df[i]) == 0 else sum((black_points_accrued.drop(['Election Set'], axis = 1)*black_align_prim)[i])/sum(black_weight_df[i]) for i in dist_changes]
    hisp_vra_prob = [0 if sum(hisp_weight_df[i])  == 0 else sum((hisp_points_accrued.drop(['Election Set'], axis = 1)*hisp_align_prim)[i])/sum(hisp_weight_df[i]) for i in dist_changes]         
    neither_vra_prob = [0 if sum(neither_weight_df[i])  == 0 else sum(neither_points_accrued[i])/sum(neither_weight_df[i]) for i in dist_changes]   
               
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
    if single_map:
        return  dict(zip(dist_changes, zip(final_hisp_prob, final_black_prob, final_neither, final_overlap))), \
                black_pref_wins, hisp_pref_wins, neither_pref_wins, black_points_accrued, hisp_points_accrued, \
                neither_points_accrued, primary_second_df
    else:
        return dict(zip(dist_changes, zip(final_hisp_prob, final_black_prob, final_neither, final_overlap)))
    
 
def compute_W2(elec_sets, districts, min_cand_weights_dict, black_pref_cands_df, hisp_pref_cands_df, \
               cand_race_dict):
    min_cand_black_W2 = pd.DataFrame(columns = districts)
    min_cand_black_W2["Election Set"] = elec_sets
    min_cand_hisp_W2 = pd.DataFrame(columns = districts)
    min_cand_hisp_W2["Election Set"] = elec_sets
    min_cand_neither_W2 = pd.DataFrame(columns = districts)
    min_cand_neither_W2["Election Set"] = elec_sets

    for dist in districts:
        black_pref = list(black_pref_cands_df[dist])

        black_pref_race = [cand_race_dict[bp] for bp in black_pref]
        black_cand_weight = [min_cand_weights_dict["Relevant Minority"] if "Black" in bpr else \
                             min_cand_weights_dict["Other"] for bpr in black_pref_race]
        min_cand_black_W2[dist] = black_cand_weight
        
        hisp_pref = list(hisp_pref_cands_df[dist])
        hisp_pref_race = [cand_race_dict[hp] for hp in hisp_pref]
        hisp_cand_weight = [min_cand_weights_dict["Relevant Minority"] if "Hispanic" in hpr else \
                             min_cand_weights_dict["Other"] for hpr in hisp_pref_race]
        min_cand_hisp_W2[dist] = hisp_cand_weight
    
         
        neither_cand_weight = [min_cand_weights_dict['Relevant Minority'] if ('Hispanic' in hpr and 'Black' in bpr) else\
        min_cand_weights_dict['Other'] if ('Hispanic' not in hpr and 'Black' not in bpr) else \
           min_cand_weights_dict['Partial '] for bpr,hpr in zip(black_pref_race, hisp_pref_race)]
        min_cand_neither_W2[dist] = neither_cand_weight
        
    return min_cand_black_W2, min_cand_hisp_W2, min_cand_neither_W2


#to aggregrate precinct EI to district EI for district model mode
def cand_pref_all(prec_quant_df, dist_prec_list, bases, outcomes, sample_size = 1000 ):
    ''' 
    prec_quant_df: precinct ei data drame
    dist_prec_list: list of precinct ids (for TX, CTYVTDS)
    bases: all demog/election/cand column name bases
    outcomes: dictionary of demog/elec pair to relavent column name bases
    sample_size: how many draws from precinct distributions
    '''
    quant_vals = [0,125,250,375,500,625,750,875,1000]
    dist_prec_quant = prec_quant_df[prec_quant_df['CNTYVTD'].isin(dist_prec_list)]    
    draws = {}
    for base in bases:
        vec_rand = np.random.rand(sample_size,len(dist_prec_quant))
        vec_rand_shift = np.array(dist_prec_quant[base +'.'+ '0'])+ sum(np.minimum(np.maximum(vec_rand-quant_vals[qv]/1000,0),.125)*8*np.array(dist_prec_quant[base + '.' +  str(quant_vals[qv+1])]-dist_prec_quant[base + '.'+ str(quant_vals[qv])]) for qv in range(len(quant_vals)-1))
        draws[base] = vec_rand_shift.sum(axis=1)  
        
    return {outcome:{base.split('.')[1].split('_counts')[0]:sum([1 if draws[base][i]==max([draws[base_][i] for base_ in outcomes[outcome]]) else 0 for i in range(sample_size)])/sample_size for base in outcomes[outcome]} for outcome in outcomes.keys()}

def cand_pref_all_alt_qv(prec_quant_df, dist_prec_list, bases, outcomes, sample_size = 1000 ):
    quant_vals = np.array([0,125,250,375,500,625,750,875,1000])
    dist_prec_quant = prec_quant_df[prec_quant_df['CNTYVTD'].isin(dist_prec_list)]    
    draws = {}
    for base in bases:
        vec_rand_shift = np.concatenate([np.random.rand(int(sample_size/8),len(dist_prec_quant))*(np.array(dist_prec_quant[base + '.' +  str(quant_vals[qv+1])])-np.array(dist_prec_quant[base + '.'+ str(quant_vals[qv])]))+np.array(dist_prec_quant[base + '.'+ str(quant_vals[qv])]) for qv in range(len(quant_vals)-1)])
        list(map(np.random.shuffle, vec_rand_shift.T))
        draws[base] = vec_rand_shift.sum(axis=1) 
    return {outcome:{base.split('.')[1].split('_counts')[0]:sum([1 if draws[base][i]==max([draws[base_][i] for base_ in outcomes[outcome]]) else 0 for i in range(sample_size)])/sample_size for base in outcomes[outcome]} for outcome in outcomes.keys()}

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
