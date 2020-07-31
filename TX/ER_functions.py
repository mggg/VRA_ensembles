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
           share_norm_params_dict, display_dist = 1, display_elec = 1):
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
        plt.xlabel("BCVAP share of Precinct CVAP")
        plt.ylabel("{}'s share of precinct CVAP".format(cand))
        plt.title("ER, Black support for {}, district {}".format(cand, district+1))
        plt.savefig("Black {} support_{}.png".format(cand, district+1))
    
    return mean, std

def preferred_cand(district, elec, cand_norm_params, display_dist = 1, display_elec = 1):
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
        
        if [0.0,0.0] in list(cand_norm_params.values()):
            blank_index = [k for k,v in cand_norm_params.items() if v == [0.0,0.0]][0]
            del cand_norm_params[blank_index]
            
        res = scipy.optimize.minimize(lambda x, cand_norm_params: -f(x, cand_norm_params), \
                                      (dist1[0]- dist2[0])/2+ dist2[0] , args=(cand_norm_params), \
                                      bounds = [(dist2[0], dist1[0])])       
        pref_cand = dist1_index
        pref_confidence = abs(res.fun)[0]
        
        if district == display_dist and elec == display_elec:
            print("elec", elec)
            print("params", cand_norm_params)
            print("black first choice", dist1_index, dist1, dist1_index)
            
            plt.figure(figsize=(12, 6))
            for j in cand_norm_params.keys(): 
                if j != dist1_index and j != dist2_index:
                    continue
                mean = cand_norm_params[j][0]
                std = cand_norm_params[j][1]
                x = np.linspace(mean - 3*std, mean + 3*std)
                plt.plot(x,scipy.stats.norm.pdf(x, cand_norm_params[j][0], cand_norm_params[j][1]))
                plt.axvline(x= mean, color = 'black')
                dist_from_mean = abs(res.x[0]-mean)
                iq=stats.norm(mean,std)
                section = np.arange(mean-dist_from_mean, mean+dist_from_mean, .01)
                plt.fill_between(section,iq.pdf(section)) 
            plt.title("Candidate Distributions {}, {}".format(elec, district))
        
        return pref_cand, pref_confidence

def accrue_points(primary_winner, min_pref_cand, party_general_winner, min_pref_prim_rank, \
                  runoff_winner, runoff_min_pref, candidates, runoff_race):
        if runoff_race == None:
            accrue = ((primary_winner == min_pref_cand) & (party_general_winner == 'D')) 
        else:
            accrue = ((min_pref_prim_rank < 3) \
            & (runoff_winner == runoff_min_pref) & (party_general_winner == 'D'))       
        return accrue