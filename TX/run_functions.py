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
import intervals as I
import time
import heapq
import operator

def f(x, dist_list):
    product = 1
    for i in dist_list.keys():
        mean = dist_list[i][0]
        std = dist_list[i][1]
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
           share_norm_params_dict, display_dist = 1, display_elec = 1,race = 1):
    group_share_add = sm.add_constant(group_share)
    model = sm.WLS(cand_cvap_share, group_share_add, weights = pop_weights)            
    model = model.fit()
    cand_cvap_share_pred = model.predict()
    mean, std = norm_dist_params(cand_cvap_share, cand_cvap_share_pred, sum(model.params), pop_weights)
    if district == display_dist and elec == display_elec:
        plt.figure(figsize=(12, 6))
        plt.scatter(group_share, cand_cvap_share, c = pop_weights, cmap = 'viridis_r')  
            # scatter plot showing actual data
        plt.plot(group_share +[1], list(cand_cvap_share_pred) + [sum(model.params)], 'r', linewidth=2) #extend lin regresssion line to 1
        plt.xticks(np.arange(0,1.1,.1))
        plt.yticks(np.arange(0,1.1,.1))
        plt.xlabel("{} share of Precinct CVAP".format(race))
        plt.ylabel("{}'s share of precinct CVAP".format(cand))
        plt.title("ER, {} support for {}, district {}".format(race, cand, district+1))
      #  plt.savefig("{} {} support_{}.png".format(race, cand, district+1))
    
    return mean, std

def preferred_cand(district, elec, cand_norm_params, display_dist = 1, display_elec = 1, race = 1):
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

        if [0.0,0.0] in list(cand_norm_params.values()):
            blank_index = [k for k,v in cand_norm_params.items() if v == [0.0,0.0]][0]
            del cand_norm_params[blank_index]
            
        res = scipy.optimize.minimize(lambda x, cand_norm_params: -f(x, cand_norm_params), \
                                      (dist1[0]- dist2[0])/2+ dist2[0] , args=(cand_norm_params), \
                                      bounds = [(dist2[0], dist1[0])])           
        pref_confidence = abs(res.fun)[0]
        
        
        if district == display_dist and elec == display_elec:
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
        
        return pref_cand, pref_confidence

def accrue_points(primary_winner, min_pref_cand, party_general_winner, min_pref_prim_rank, \
                  runoff_winner, runoff_min_pref, candidates, runoff_race):
        if runoff_race == None:
            accrue = ((primary_winner == min_pref_cand) & (party_general_winner == 'D')) 
        else:
            accrue = ((min_pref_prim_rank < 3) \
            & (runoff_winner == runoff_min_pref) & (party_general_winner == 'D'))       
        return accrue

def compute_dist(map_winners, black_pref_cands_df, black_pref_cands_runoffs,\
                 hisp_pref_cands_df, hisp_pref_cands_runoffs, neither_weight_df, \
                 black_weight_df, hisp_weight_df, dist_elec_results, dist_list,
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
    
    for i in dist_list:
        for elec_set in elec_sets:
            black_pref_cand = black_pref_cands_df.loc[black_pref_cands_df["Election Set"] == elec_set, i].values[0]
            hisp_pref_cand = hisp_pref_cands_df.loc[hisp_pref_cands_df["Election Set"] == elec_set, i].values[0]       
            
            primary_race = elec_set_dict[elec_set]["Primary"]
            runoff_race = None if 'Runoff' not in elec_set_dict[elec_set].keys() else elec_set_dict[elec_set]["Runoff"]
            
            primary_winner = primary_winners.loc[primary_winners["Election Set"] == elec_set, i].values[0]
            general_winner = general_winners.loc[general_winners["Election Set"] == elec_set, i].values[0]
            runoff_winner = "N/A" if elec_set not in list(runoff_winners["Election Set"]) \
            else runoff_winners.loc[runoff_winners["Election Set"] == elec_set, i].values[0]
            
            primary_race_shares = dist_elec_results[primary_race][i]
            primary_ranking = {key: rank for rank, key in enumerate(sorted(primary_race_shares, key=primary_race_shares.get, reverse=True), 1)}        
            second_place_primary = [cand for cand, value in primary_ranking.items() if primary_ranking[cand] == 2]
            primary_second_df.at[primary_second_df["Election Set"] == elec_set, i] = second_place_primary[0]
            
            black_pref_prim_rank = primary_ranking[black_pref_cand]
            hisp_pref_prim_rank = primary_ranking[hisp_pref_cand]
            
            party_general_winner = cand_race_table.loc[cand_race_table["Candidates"] == general_winner, "Party"].values[0] 
             
            #we always care who preferred candidate is in runoff if the minority preferred primary
            #candidate wins in district primary
            runoff_black_pref = "N/A" if runoff_winner == "N/A" else \
                        black_pref_cands_runoffs.loc[black_pref_cands_runoffs["Election Set"] == elec_set, i].values[0]
            
            runoff_hisp_pref = "N/A" if runoff_winner == 'N/A' else \
                        hisp_pref_cands_runoffs.loc[hisp_pref_cands_runoffs["Election Set"] == elec_set, i].values[0]
                             
            #winning conditions (conditions to accrue points for election set/minority group):
            black_accrue = accrue_points(primary_winner, black_pref_cand, party_general_winner, black_pref_prim_rank, \
                      runoff_winner, runoff_black_pref, candidates, runoff_race)
            black_pref_wins.at[black_pref_wins["Election Set"] == elec_set, i] = black_accrue 
            
            hisp_accrue = accrue_points(primary_winner, hisp_pref_cand, party_general_winner, hisp_pref_prim_rank, 
                      runoff_winner, runoff_hisp_pref, candidates, runoff_race)
            hisp_pref_wins.at[hisp_pref_wins["Election Set"] == elec_set, i] = hisp_accrue 
            
        
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
    black_vra_prob = [0 if sum(black_weight_df[i]) == 0 else sum(black_points_accrued[i])/sum(black_weight_df[i]) for i in dist_list]
    hisp_vra_prob = [0 if sum(hisp_weight_df[i])  == 0 else sum(hisp_points_accrued[i])/sum(hisp_weight_df[i]) for i in dist_list]   
    neither_vra_prob = [0 if sum(neither_weight_df[i])  == 0 else sum(neither_points_accrued[i])/sum(neither_weight_df[i]) for i in dist_list]   
    
    min_neither = [0 if (black_vra_prob[i] + hisp_vra_prob[i]) > 1 else 1 -(black_vra_prob[i] + hisp_vra_prob[i]) for i in range(len(dist_list))]
    max_neither = [1 - max(black_vra_prob[i], hisp_vra_prob[i]) for i in range(len(dist_list))]
    
    #uses ven diagram overlap/neither method
    final_neither = [min_neither[i] + neither_vra_prob[i]*(max_neither[i]-min_neither[i]) for i in range(len(dist_list))]
    final_overlap = [final_neither[i] + black_vra_prob[i] + hisp_vra_prob[i] - 1 for i in range(len(dist_list))]
    final_black_prob = [black_vra_prob[i] - final_overlap[i] for i in range(len(dist_list))]
    final_hisp_prob = [hisp_vra_prob[i] - final_overlap[i] for i in range(len(dist_list))]
        
    return  dict(zip(dist_list, zip(final_hisp_prob, final_black_prob, final_neither, final_overlap)))
    
 
def compute_W2(elec_sets, districts, min_cand_weights, black_pref_cands_df, hisp_pref_cands_df, \
               cand_race_table):
    min_cand_black_W2 = pd.DataFrame(columns = districts)
    min_cand_black_W2["Election Set"] = elec_sets
    min_cand_hisp_W2 = pd.DataFrame(columns = districts)
    min_cand_hisp_W2["Election Set"] = elec_sets
    min_cand_neither_W2 = pd.DataFrame(columns = districts)
    min_cand_neither_W2["Election Set"] = elec_sets

    for elec_set in elec_sets:
        for dist in districts:             
            black_pref = black_pref_cands_df.loc[black_pref_cands_df["Election Set"] == elec_set, dist].values[0]
            black_pref_race = cand_race_table.loc[cand_race_table["Candidates"] == black_pref, "Race"].values[0]
            black_pref_black = True if 'Black' in black_pref_race else False        
            black_cand_weight_type = 'Relevant Minority' if black_pref_black else 'Other'
            min_cand_black_W2.at[min_cand_black_W2["Election Set"] == elec_set, dist] = min_cand_weights[black_cand_weight_type][0] 
            
            hisp_pref = hisp_pref_cands_df.loc[hisp_pref_cands_df["Election Set"] == elec_set, dist].values[0]
            hisp_pref_race = cand_race_table.loc[cand_race_table["Candidates"] == hisp_pref, "Race"].values[0]
            hisp_pref_hisp = True if 'Hispanic' in hisp_pref_race else False             
            hisp_cand_weight_type = 'Relevant Minority' if hisp_pref_hisp else 'Other'
            min_cand_hisp_W2.at[min_cand_hisp_W2["Election Set"] == elec_set, dist] = min_cand_weights[hisp_cand_weight_type][0] 
            
            neither_cand_weight_type = 'Relevant Minority' if (hisp_pref_hisp & black_pref_black) else\
                    'Other' if (not hisp_pref_hisp and not black_pref_black) else 'Partial '
            min_cand_neither_W2.at[min_cand_neither_W2["Election Set"] == elec_set, dist] = min_cand_weights[neither_cand_weight_type][0] 
      
    return min_cand_black_W2, min_cand_hisp_W2, min_cand_neither_W2