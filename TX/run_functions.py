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
def f(x, params):
    product = 1
    for i in params.keys():
        mean = params[i][0]
        std = params[i][1]
        dist_from_mean = abs(x-mean)
        ro = scipy.stats.norm.cdf(mean+dist_from_mean, mean, std) - scipy.stats.norm.cdf(mean-dist_from_mean, mean, std)
        product = product*ro
    return product

def norm_dist_params(y, y_pred, sum_params, pop_weights): #y_predict is vector of predicted values, sum_params is prediction when x = 100%
    mean = sum_params #predicted value at x = 100%
    n = len(y)
    y_resid = [len(pop_weights)*w_i*(y_i - y_hat)**2 for w_i,y_i, y_hat in zip(pop_weights,y,y_pred)]
    var = sum(y_resid)/(n-2)   
    std = np.sqrt(var)
    return mean, std #CHECK- std not var right?

def ER_run(cand, elec, district, group_share, cand_cvap_share, pop_weights, \
           share_norm_params_dict, display_dist = 1, display_elec = 1,race = 1, verbose_bool = False):
    group_share_add = sm.add_constant(group_share)
    model = sm.WLS(cand_cvap_share, group_share_add, weights = pop_weights)            
    model = model.fit()
    cand_cvap_share_pred = model.predict()
    mean, std = norm_dist_params(cand_cvap_share, cand_cvap_share_pred, sum(model.params), pop_weights)
    if district == display_dist and elec == display_elec and verbose_bool:
        plt.figure(figsize=(12, 6))
        plt.scatter(group_share, cand_cvap_share, c = pop_weights, cmap = 'viridis_r')  
            # scatter plot showing actual data
        plt.plot(group_share +[1], list(cand_cvap_share_pred) + [sum(model.params)], 'r', linewidth=2) #extend lin regresssion line to 1
        plt.xticks(np.arange(0,1.1,.1))
        plt.yticks(np.arange(0,1.1,.1))
        plt.xlabel("{} share of Precinct CVAP".format(race))
        plt.ylabel("{}'s share of precinct CVAP".format(cand))
        plt.title("ER, {} support for {}, district {}".format(race, cand, district+1))
        plt.savefig(DIR + "outputs/{}_{}_support_{}_{}.png".format(race, cand, district+1, elec))
    
    return mean, std

def preferred_cand(district, elec, cand_norm_params, display_dist = 1, display_elec = 1, race = 1, verbose_bool = False):
    if len(cand_norm_params) == 1:
            pref_cand = list(cand_norm_params.keys())[0]
            pref_confidence = 1
    else:
        cand_norm_params_copy = cand_norm_params.copy()
        dist1_index = max(cand_norm_params_copy.items(), key=operator.itemgetter(1))[0]
        dist1 = cand_norm_params_copy[dist1_index]
        del cand_norm_params_copy[dist1_index]
        dist2_index = max(cand_norm_params_copy.items(), key=operator.itemgetter(1))[0]
        dist2 = cand_norm_params_copy[dist2_index]        
        pref_cand = dist1_index        

        
        del_indices = [k for k,v in cand_norm_params.items() if cand_norm_params[k] == [0.0,0.0]]
        for z in del_indices:
            cand_norm_params.pop(z, None)
            
        res = scipy.optimize.minimize(lambda x, cand_norm_params: -f(x, cand_norm_params), \
                                      (dist1[0]- dist2[0])/2+ dist2[0] , args=(cand_norm_params), \
                                      bounds = [(dist2[0], dist1[0])])           
        pref_confidence = abs(res.fun)[0]
        
        
        if district == display_dist and elec == display_elec and verbose_bool:
            print("elec", elec)
            print("race", race)
            print("params", cand_norm_params)
            print("first choice", dist1_index)
            print("second choice", dist2_index)
            print("conf in 1st:", pref_confidence)
            
            plt.figure(figsize=(12, 6))
            for j in cand_norm_params.keys(): 
                if j != dist1_index and j != dist2_index:
                    continue
                print(j)
                mean = cand_norm_params[j][0]
                std = cand_norm_params[j][1]
                x = np.linspace(mean - 3*std, mean + 3*std)
                plt.plot(x,scipy.stats.norm.pdf(x, cand_norm_params[j][0], cand_norm_params[j][1]))
                plt.axvline(x= mean, color = 'black')
                dist_from_mean = abs(res.x[0]-mean)
                iq=stats.norm(mean,std)
                section = np.arange(mean-dist_from_mean, mean+dist_from_mean, .01)
                plt.fill_between(section,iq.pdf(section)) 
            plt.title("Candidate Distributions {}, {}, {}".format(elec, district, race))
            plt.savefig(DIR + "outputs/{}_{}_cand_dists_{}.png".format(race, district+1, elec))
        
        return pref_cand, pref_confidence


def compute_final_dist(map_winners, black_pref_cands_df, black_pref_cands_runoffs,\
                 hisp_pref_cands_df, hisp_pref_cands_runoffs, neither_weight_df, \
                 black_weight_df, hisp_weight_df, dist_elec_results, dist_changes,
                 cand_race_table, num_districts, candidates, \
                 elec_sets, elec_set_dict):
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
    black_vra_prob = [0 if sum(black_weight_df[i]) == 0 else sum(black_points_accrued[i])/sum(black_weight_df[i]) for i in dist_changes]
    hisp_vra_prob = [0 if sum(hisp_weight_df[i])  == 0 else sum(hisp_points_accrued[i])/sum(hisp_weight_df[i]) for i in dist_changes]   
    neither_vra_prob = [0 if sum(neither_weight_df[i])  == 0 else sum(neither_points_accrued[i])/sum(neither_weight_df[i]) for i in dist_changes]   
    
    min_neither = [0 if (black_vra_prob[i] + hisp_vra_prob[i]) > 1 else 1 -(black_vra_prob[i] + hisp_vra_prob[i]) for i in range(len(dist_changes))]
    max_neither = [1 - max(black_vra_prob[i], hisp_vra_prob[i]) for i in range(len(dist_changes))]
    
    #uses ven diagram overlap/neither method
    final_neither = [min_neither[i] + neither_vra_prob[i]*(max_neither[i]-min_neither[i]) for i in range(len(dist_changes))]
    final_overlap = [final_neither[i] + black_vra_prob[i] + hisp_vra_prob[i] - 1 for i in range(len(dist_changes))]
    final_black_prob = [black_vra_prob[i] - final_overlap[i] for i in range(len(dist_changes))]
    final_hisp_prob = [hisp_vra_prob[i] - final_overlap[i] for i in range(len(dist_changes))]
        
    return  dict(zip(dist_changes, zip(final_hisp_prob, final_black_prob, final_neither, final_overlap)))
    
 
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