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
from min_bound_circle import *
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
from operator import itemgetter
from functools import partial

#input parameters
#columns from GeoDF for processing
tot_pop = 'TOTPOP_x'
white_pop = 'NH_WHITE'
other_pop = 'NH_OTHER'
hisp_pop = "HISP"
black_pop = "NH_BLACK"
CVAP = "CVAP"
WCVAP = "WCVAP"
HCVAP = "HCVAP"
BCVAP = "BCVAP"
geo_id = 'CNTYVTD'
county_split_id = "CNTY_x"
PRES16_DEM = 'ClintonD_16G_President'
PRES16_REP = 'TrumpR_16G_President'
PRES12_DEM = 'ObamaD_12G_President' 
PRES12_REP = 'RomneyR_12G_President'
SEN18_DEM = "O'RourkeD_18G_U.S. Sen"
SEN18_REP = 'CruzR_18G_U.S. Sen'
C_X = "C_X"
C_Y = "C_Y"


#run parameters
start_time_total = time.time()
total_steps = 20
pop_tol = .005 #U.S. Cong
num_samples = 200
assignment1= 'CD'
run_name = 'test run'#sys.argv[1]
run_type = 'free' #sys.argv[2]
elec_weighting = 'district' #sys.argv[3]
min_group = 'black' #sys.argv[4]
num_districts = 36
degrandy_hisp = 11 #10.39 rounded up
degrandy_black = 5 #4.69 rounded up
ineffect_cutoff = 0 #percent district demo under which assume ineffective

store_interval = 2000 #how many steps until storage
stuck_length = 1000 #steps at same score until break

#need if running hillclimb with bound
bound = .1#float(sys.argv[5]) 
#need if simAnneal run (with cycles)
cycle_length = 10#float(sys.argv[6])
start_cool = 3#float(sys.argv[7])
stop_cool = 7#float(sys.argv[8])

start_map = 'random' #sys.argv[9]

store_score = -2
#read files
plot_path = '2018_vtd_cvap_elec/2018_vtd_cvap_elec.shp' 

elec_data = pd.read_csv("TX_elections.csv")
elections = elec_data["Election"] #elections we care about
elec_type = elec_data["Type"]
election_returns = pd.read_csv("TX_statewide_election_returns.csv")
elec_cand_list = list(election_returns.columns)[2:] #all candidates in all elections
recency_weights = pd.read_csv("recency_weights.csv")
min_cand_weights = pd.read_csv("min_pref_weight_binary.csv")
cand_race_table = pd.read_csv("CandidateRace.csv")

#initialize state_gdf
state_gdf = gpd.read_file(plot_path)
state_gdf["CD"] = [int(i) for i in state_gdf["CD"]]

#to edit cut-off shape file columns
election_return_cols = list(election_returns.columns)
cand1_index = election_return_cols.index('RomneyR_12G_President')
cand2_index = election_return_cols.index('WhiteD_18R_Governor')
elec_results_trunc = election_return_cols[cand1_index:cand2_index+1]

state_gdf_cols = list(state_gdf.columns)
cand1_index = state_gdf_cols.index('RomneyR_12')
cand2_index = state_gdf_cols.index('WhiteD_18R')

state_gdf_cols[cand1_index:cand2_index+1] = elec_results_trunc
state_gdf.columns = state_gdf_cols

#build graph from geo_dataframe
graph = Graph.from_geodataframe(state_gdf)
graph.add_data(state_gdf)
centroids = state_gdf.centroid
c_x = centroids.x
c_y = centroids.y
for node in graph.nodes():
    graph.nodes[node]["C_X"] = c_x[node]
    graph.nodes[node]["C_Y"] = c_y[node]

#reformat elections return - get summary stats by precinct
state_df = pd.DataFrame(state_gdf)
state_df = state_df.drop(['geometry'], axis = 1)

state_df["WCVAP%"] = np.where(state_df["CVAP"] == 0, 0, state_df["WCVAP"]/state_df["CVAP"])
state_df["HCVAP%"] = np.where(state_df["CVAP"] == 0, 0, state_df["HCVAP"]/state_df["CVAP"])
state_df["BCVAP%"] = np.where(state_df["CVAP"] == 0, 0, state_df["BCVAP"]/state_df["CVAP"])


candidates = {}
for elec in elections: #only elections we care about
    #get rid of republican candidates in primaries or runoffs (primary runoffs)
    cands = [y for y in elec_cand_list if elec in y and "R_" not in re.sub(elec, '', y) ] if \
    "R_" in elec or "P_" in elec else [y for y in elec_cand_list if elec in y] 
    
    candidates[elec] = dict(zip(list(range(len(cands))), cands)) #only candidates we care about in elections we care about
    cand_list = list(candidates[elec].values())
    pattern = '|'.join(cand_list)
    elec_df = state_df.loc[:, state_df.columns.str.contains(pattern)]
    elec_df["Total"] = elec_df.sum(axis=1)
    elec_df["CVAP"] = state_df["CVAP"]
    for cand in candidates[elec].values():
        state_df["{}%CVAP".format(cand)] = state_df["{}".format(cand)]/elec_df["CVAP"]
        

def norm_dist_params(y, y_pred, sum_params, pop_weights): #y_predict is vector of predicted values, sum_params is prediction when x = 100%
    mean = sum_params #predicted value at x = 100%
    n = len(y)
    y_resid = [len(pop_weights)*w_i*(y_i - y_hat)**2 for w_i,y_i, y_hat in zip(pop_weights,y,y_pred)]
    var = sum(y_resid)/(n-2)   
    std = np.sqrt(var)
    return mean, std

def old_elec_model(partition):  
           
    def winner(partition, election, elec_cands):
        order = [x for x in partition.parts]
        perc_for_cand = {}
        for j in range(len(elec_cands)):
            perc_for_cand[j] = dict(zip(order, partition[election].percents(j)))
        winners = {}
        for i in range(len(partition)):
            dist_percents = [perc_for_cand[j][i] for j in range(len(elec_cands))]
            winner_index = dist_percents.index(max(dist_percents))
            winners[i] = elec_cands[winner_index]
        return winners
    
    #get differences in assignment dictionaries between current and former map
    if partition.parent is not None:
        dict1 = dict(partition.parent.assignment)
        dict2 = dict(partition.assignment)
        differences = set([dict1[k] for k in dict1.keys() if dict1[k] != dict2[k]])
        
    dist_list = range(num_districts) if partition.parent is None else differences
    #map data storage    
    black_pref_cands_df = pd.DataFrame(columns = dist_list)
    black_pref_cands_df["Election"] = elections
    hisp_pref_cands_df = pd.DataFrame(columns = dist_list)
    hisp_pref_cands_df["Election"] = elections
  
    recency_W1 = pd.DataFrame(columns = range(num_districts))
    recency_W1["Election"] = elections
    
    min_cand_black_W2 = pd.DataFrame(columns = range(num_districts))
    min_cand_black_W2["Election"] = elections
    
    min_cand_hisp_W2 = pd.DataFrame(columns = range(num_districts))
    min_cand_hisp_W2["Election"] = elections
    
    black_conf_W3 = pd.DataFrame(columns = range(num_districts))
    black_conf_W3["Election"] = elections
    hisp_conf_W3 = pd.DataFrame(columns = range(num_districts))
    hisp_conf_W3["Election"] = elections
    
    black_weight_df = pd.DataFrame(columns = range(num_districts))
    hisp_weight_df = pd.DataFrame(columns = range(num_districts))
    
    
    #winners df:
    dist_winners = {} #adding district winners for each election to that election's df
    map_winners = pd.DataFrame(columns = range(num_districts))

    for j in elections:
        dist_winners[j] = winner(partition, j, candidates[j])
        keys = list(dist_winners[j].keys())
        values = list(dist_winners[j].values())
        map_winners.loc[len(map_winners)] =  [value for _,value in sorted(zip(keys,values))]
    
    map_winners = map_winners.reset_index(drop = True)
    map_winners["Election"] = elections
    
    dist_Pbcvap = {}
    dist_Phcvap = {}
    
    for district in dist_list: #get vector of precinct values for each district  
        state_df2 = state_df.copy()
        state_df2["Assign"] = state_gdf.index.map(dict(partition.assignment))
        #clean data - more?
        state_df2 = state_df2[state_df2["CVAP"] > 0]
        dist_df = state_df2[state_df2["Assign"] == district]
        black_share = list(dist_df["BCVAP%"])
        hisp_share = list(dist_df["HCVAP%"])
        pop_weights = list(dist_df.loc[:,"CVAP"].apply(lambda x: x/sum(dist_df["CVAP"])))  
        
        dist_Pbcvap[district] = sum(dist_df["BCVAP"])/sum(dist_df["CVAP"])
        dist_Phcvap[district] = sum(dist_df["HCVAP"])/sum(dist_df["CVAP"])
               
################################################################################
#run ER for candidate support!
        for elec in elections:  
            #for each regression, remove points with cand-share-of-cvap >1 (cvap disagg error)
            cand_cvap_share_dict = {}
            black_share_dict = {}
            hisp_share_dict = {}
            pop_weights_dict = {}
            black_norm_params = {}
            hisp_norm_params = {}
            for cand in candidates[elec].values():
                cand_cvap_share = list(dist_df["{}%CVAP".format(cand)])
                cand_cvap_share_indices = [i for i,elem in enumerate(cand_cvap_share) if elem < 1]
                
                cand_cvap_share_dict[cand] = list(itemgetter(*cand_cvap_share_indices)(cand_cvap_share))
                black_share_dict[cand] = list(itemgetter(*cand_cvap_share_indices)(black_share))
                hisp_share_dict[cand] = list(itemgetter(*cand_cvap_share_indices)(hisp_share))
                pop_weights_dict[cand] = list(itemgetter(*cand_cvap_share_indices)(pop_weights))
                                                                                  
                #run ER, black
                black_share_add = sm.add_constant(black_share_dict[cand])
                model = sm.WLS(cand_cvap_share_dict[cand], black_share_add, weights = pop_weights_dict[cand])            
                model = model.fit()
                cand_cvap_share_pred = model.predict()
                mean, std = norm_dist_params(cand_cvap_share_dict[cand], cand_cvap_share_pred, sum(model.params), pop_weights_dict[cand])
                black_norm_params[cand] = [mean,std]
                
                #run ER, hisp
                hisp_share_add = sm.add_constant(hisp_share_dict[cand])
                model = sm.WLS(cand_cvap_share_dict[cand], hisp_share_add, weights = pop_weights_dict[cand])            
                model = model.fit()
                cand_cvap_share_pred = model.predict()
                mean, std = norm_dist_params(cand_cvap_share_dict[cand], cand_cvap_share_pred, sum(model.params), pop_weights_dict[cand])
                hisp_norm_params[cand] = [mean,std]
##################################################################################
            #optimizations for confidence! (W3)
            #populate black pref candidate and confidence in candidate (df 1a and 2aii)
            black_norm_params_copy = black_norm_params.copy()
            dist1_index = max(black_norm_params_copy.items(), key=operator.itemgetter(1))[0]
            dist1 = black_norm_params_copy[dist1_index]
            del black_norm_params_copy[dist1_index]
            dist2_index = max(black_norm_params_copy.items(), key=operator.itemgetter(1))[0]
            dist2 = black_norm_params_copy[dist2_index]
            
            if [0.0,0.0] in list(black_norm_params.values()):
                blank_index = [k for k,v in black_norm_params.items() if v == [0.0,0.0]][0]
                del black_norm_params[blank_index]
                
            res = scipy.optimize.minimize(lambda x, black_norm_params: -f(x, black_norm_params), (dist1[0]- dist2[0])/2+ dist2[0] , args=(black_norm_params), bounds = [(dist2[0], dist1[0])])
            black_er_conf = abs(res.fun)[0]                          
            #final black pref and confidence in choice
            black_pref_cand = dist1_index
            black_pref_cands_df.at[black_pref_cands_df["Election"] == elec, district] = black_pref_cand
            black_conf_W3.at[black_conf_W3["Election"] == elec, district] = black_er_conf
            
            #populate hisp pref candidate and confidence in candidate (df 1b and 2bii)
            hisp_norm_params_copy = hisp_norm_params.copy()
            dist1_index = max(hisp_norm_params_copy.items(), key=operator.itemgetter(1))[0]
            dist1 = hisp_norm_params_copy[dist1_index]
            del hisp_norm_params_copy[dist1_index]
            dist2_index = max(hisp_norm_params_copy.items(), key=operator.itemgetter(1))[0]
            dist2 = hisp_norm_params_copy[dist2_index]
            
            if [0.0,0.0] in list(hisp_norm_params.values()):
                blank_index = [k for k,v in hisp_norm_params.items() if v == [0.0,0.0]][0]
                del hisp_norm_params[blank_index]
                
            res = scipy.optimize.minimize(lambda x, hisp_norm_params: -f(x, hisp_norm_params), (dist1[0]- dist2[0])/2+ dist2[0] , args=(hisp_norm_params), bounds = [(dist2[0], dist1[0])])
            hisp_er_conf = abs(res.fun)[0]
            #final hisp pref and confidence in choice
            hisp_pref_cand = dist1_index
            hisp_pref_cands_df.at[hisp_pref_cands_df["Election"] == elec, district] = hisp_pref_cand
            hisp_conf_W3.at[hisp_conf_W3["Election"] == elec, district] = hisp_er_conf
         
#######################################################################################
    #get election weights 1 and 2 and combine for final            
    for elec in elections:
        elec_year = elec_data.loc[elec_data["Election"] == elec, 'Year'].values[0].astype(str)
        for dist in dist_list:      
            recency_W1.at[recency_W1["Election"] == elec, dist] = recency_weights[elec_year][0]
            black_pref = black_pref_cands_df.loc[black_pref_cands_df["Election"] == elec, dist].values[0]
            black_pref_race = cand_race_table.loc[cand_race_table["Candidates"] == black_pref, "Race"].values[0]
            black_pref_black = True if 'Black' in black_pref_race else False
            
            min_cand_weight_type = 'Relevant Minority' if black_pref_black else 'Other'
            min_cand_black_W2.at[min_cand_black_W2["Election"] == elec, dist] = min_cand_weights[min_cand_weight_type][0] 
            
            hisp_pref = hisp_pref_cands_df.loc[hisp_pref_cands_df["Election"] == elec, dist].values[0]
            hisp_pref_race = cand_race_table.loc[cand_race_table["Candidates"] == hisp_pref, "Race"].values[0]
            hisp_pref_hisp = True if 'Hispanic' in hisp_pref_race else False 
                
            min_cand_weight_type = 'Relevant Minority' if hisp_pref_hisp else 'Other'
            min_cand_hisp_W2.at[min_cand_hisp_W2["Election"] == elec, dist] = min_cand_weights[min_cand_weight_type][0] 
             
      #  min_cand_W2.to_csv("W2_df.csv")
    #final 2a and 2b election probativity scores
    black_weight_df = recency_W1.drop(["Election"], axis=1)*min_cand_black_W2.drop(["Election"], axis=1)*black_conf_W3.drop(["Election"], axis=1)
    hisp_weight_df = recency_W1.drop(["Election"], axis=1)*min_cand_hisp_W2.drop(["Election"], axis=1)*hisp_conf_W3.drop(["Election"], axis=1)    
    
    if elec_weighting == 'equal':
        for col in black_weight_df.columns:
            black_weight_df[col].values[:] = 1
        for col in hisp_weight_df.columns:
            hisp_weight_df[col].values[:] = 1                        
        
#################################################################################
#combine points and get Prob distribution for district!! (in prim, gen and final)    
    #accrue points for black and hispanic voters if cand-of-choice wins  
    black_pref_cands_df.to_csv("black_pref_cands.csv")
    map_winners.to_csv("winners.csv")
    
    map_winners_new = map_winners[dist_list]
    black_pref_wins = map_winners_new == black_pref_cands_df.drop(["Election"], axis = 1) 
    black_points_accrued = black_weight_df*black_pref_wins   
      
    hisp_pref_wins = map_winners_new == hisp_pref_cands_df.drop(["Election"], axis = 1) 
    hisp_points_accrued = hisp_weight_df*hisp_pref_wins 
        
    #add in election type info for batching and summing accrues points
    hisp_points_accrued["Type"] = elec_type
    black_points_accrued["Type"] = elec_type
    hisp_weight_df["Type"] = elec_type
    black_weight_df["Type"] = elec_type
    
    recency_W1["Type"] = elec_type
    min_cand_hisp_W2["Type"] = elec_type
    min_cand_black_W2["Type"] = elec_type
    hisp_conf_W3["Type"] = elec_type
    black_conf_W3["Type"] = elec_type
    
    hisp_points_accrued_prim = hisp_points_accrued[hisp_points_accrued["Type"] != 'General']
    black_points_accrued_prim = black_points_accrued[black_points_accrued["Type"] != 'General']   
    hisp_points_accrued_gen = hisp_points_accrued[hisp_points_accrued["Type"] == 'General']
    black_points_accrued_gen = black_points_accrued[black_points_accrued["Type"] == 'General']
     
    hisp_weight_df_prim = hisp_weight_df[hisp_weight_df["Type"] != 'General']
    black_weight_df_prim = black_weight_df[black_weight_df["Type"] != 'General']
    hisp_weight_df_gen = hisp_weight_df[hisp_weight_df["Type"] == 'General']
    black_weight_df_gen = black_weight_df[black_weight_df["Type"] == 'General']    

########################################################################################
    #put final values into output dictionaries and compute vra score for map    
    #dictionary of probability distributions from prim, gen and final combined
    black_vra_prob_prim = [0 if sum(black_weight_df_prim[i]) == 0 else sum(black_points_accrued_prim[i])/sum(black_weight_df_prim[i]) for i in dist_list]
    hisp_vra_prob_prim = [0 if sum(hisp_weight_df_prim[i])  == 0 else sum(hisp_points_accrued_prim[i])/sum(hisp_weight_df_prim[i]) for i in dist_list]   
    black_vra_prob_gen = [0 if sum(black_weight_df_gen[i]) == 0 else sum(black_points_accrued_gen[i])/sum(black_weight_df_gen[i]) for i in dist_list]
    hisp_vra_prob_gen = [0 if sum(hisp_weight_df_gen[i]) == 0 else sum(hisp_points_accrued_gen[i])/sum(hisp_weight_df_gen[i]) for i in dist_list]
    
    black_prob_dist_dict = dict(zip(dist_list, [x*y for x,y in zip(black_vra_prob_prim, black_vra_prob_gen)]))
    hisp_prob_dist_dict = dict(zip(dist_list, [x*y for x,y in zip(hisp_vra_prob_prim, hisp_vra_prob_gen)]))

    black_prob_df_copy = black_prob_df.copy()
    hisp_prob_df_copy = hisp_prob_df.copy()

    if step_Num == 0:
        keys = list(black_prob_dist_dict.keys())
        values = list(black_prob_dist_dict.values())
        black_prob_df_copy.loc[len(black_prob_df_copy)] = [value for _,value in sorted(zip(keys,values))] 
        
        keys = list(hisp_prob_dist_dict.keys())
        values = list(hisp_prob_dist_dict.values())
        hisp_prob_df_copy.loc[len(hisp_prob_df_copy)] = [value for _,value in sorted(zip(keys,values))]
        
    else:
        black_prob_df_copy.loc[len(black_prob_df_copy)] = black_prob_df_copy.loc[len(black_prob_df_copy) -1]
        for i in black_prob_dist_dict.keys():
            black_prob_df_copy.at[len(black_prob_df_copy)-1, i] = black_prob_dist_dict[i]
        
        hisp_prob_df_copy.loc[len(hisp_prob_df_copy)] = hisp_prob_df_copy.loc[len(hisp_prob_df_copy) -1]
        for i in hisp_prob_dist_dict.keys():
            hisp_prob_df_copy.at[len(hisp_prob_df_copy)-1, i] = hisp_prob_dist_dict[i]
        

    total_black_final = sum(black_prob_df_copy.loc[len(black_prob_df_copy)-1])
    total_hisp_final = sum(hisp_prob_df_copy.loc[len(hisp_prob_df_copy)-1])
    
    return black_prob_dist_dict, hisp_prob_dist_dict, total_black_final, total_hisp_final, 
            
def num_cut_edges(partition):
    return len(partition["cut_edges"])
        
def f(x, dist_list):
    product = 1
    for i in list(dist_list.keys()):
        mean = dist_list[i][0]
        std = dist_list[i][1]
        dist_from_mean = abs(x-mean)
        ro = scipy.stats.norm.cdf(mean+dist_from_mean, mean, std) - scipy.stats.norm.cdf(mean-dist_from_mean, mean, std)
        product = product*ro
    return product

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

def num_splits(partition, df = state_gdf):
    df["current"] = df.index.map(partition.assignment.to_dict())
    return sum(df.groupby(county_split_id)['current'].nunique() > 1)

def vra_score(partition):
    black_vra_dists = partition["old_elec_model"][2] 
    hisp_vra_dists = partition["old_elec_model"][3]    
    min_group_dists = black_vra_dists if min_group == 'black' else hisp_vra_dists
    min_group_degrandy = degrandy_black if min_group == 'black' else degrandy_hisp
    return min(min_group_dists - min_group_degrandy,0)

#vra data "output" - chain storage, need to define here
# for use in old_elec_model updater
black_prob_df = pd.DataFrame(columns = range(num_districts))
hisp_prob_df = pd.DataFrame(columns = range(num_districts))
step_Num = 0
my_updaters = {
    "population": updaters.Tally(tot_pop, alias = "population"),
    "white_pop": updaters.Tally(white_pop, alias = "white_pop"),
    "other_pop": updaters.Tally(other_pop, alias = "other_pop"),
    "hisp_pop": updaters.Tally(hisp_pop, alias = "hisp_pop"),
    "black_pop": updaters.Tally(black_pop, alias = "black_pop"),
    "CVAP": updaters.Tally(CVAP, alias = "CVAP"),
    "WCVAP": updaters.Tally(WCVAP, alias = "WCVAP"),
    "HCVAP": updaters.Tally(HCVAP, alias = "HCVAP"),
    "BCVAP": updaters.Tally(BCVAP, alias = "BCVAP"),
    "PRES16_DEM": updaters.Tally(PRES16_DEM, alias = "PRES16_DEM"),
    "PRES16_REP": updaters.Tally(PRES16_REP, alias = "PRES16_REP"),
    "PRES12_DEM": updaters.Tally(PRES12_DEM, alias = "PRES12_DEM"),
    "PRES12_REP": updaters.Tally(PRES12_REP, alias = "PRES12_REP"),
    "SEN18_DEM": updaters.Tally(SEN18_DEM, alias = "SEN18_DEM"),
    "SEN18_REP": updaters.Tally(SEN18_REP, alias = "SEN18_REP"),
    "Sum_CX": updaters.Tally(C_X, alias = "Sum_CX"),
    "Sum_CY": updaters.Tally(C_Y, alias = "Sum_CY"),
    "cut_edges": cut_edges,
    "num_cut_edges": num_cut_edges,  
    "num_splits": num_splits,
    "demo_percents": demo_percents,
    "old_elec_model": old_elec_model,
    "vra_score": vra_score,
    "centroids": centroids
}


#updater functions
elections_track = [
    Election("PRES16", {"Democratic": 'ClintonD_16G_President' , "Republican": 'TrumpR_16G_President'}, alias = "PRES16"),
    Election("PRES12", {"Democratic": 'ObamaD_12G_President' , "Republican": 'RomneyR_12G_President'}, alias = "PRES12"),
    Election("SEN18", {"Democratic": "O'RourkeD_18G_U.S. Sen" , "Republican": 'CruzR_18G_U.S. Sen'}, alias = "SEN18"),   
]

election_updaters = {election.name: election for election in elections_track}
my_updaters.update(election_updaters)

election_functions = [Election(j, candidates[j]) for j in elections]

election_updaters = {election.name: election for election in election_functions}
my_updaters.update(election_updaters)


#initial partition
total_population = state_gdf[tot_pop].sum()
ideal_population = total_population/num_districts
random_assign = recursive_tree_part(graph, range(num_districts), ideal_population, tot_pop, pop_tol, node_repeats = 5)
assignment = assignment1 if start_map == 'enacted' else random_assign
initial_partition = GeographicPartition(graph = graph, assignment = assignment, updaters = my_updaters)

#basic plan info
total_population = state_gdf[tot_pop].sum()
ideal_population = total_population/len(initial_partition)

proposal = partial(
    recom, pop_col=tot_pop, pop_target=ideal_population, epsilon= pop_tol, node_repeats=3
)

#acceptance functions
accept = accept.always_accept

def hill_accept(partition):
    if not partition.parent:
        return True
    proposal_vra = partition["vra_score"]
    parent_vra = partition.parent["vra_score"]
    return proposal_vra > parent_vra

def hill_accept_bound(partition):
    if not partition.parent:
        return True
    proposal_vra = partition["vra_score"]
    parent_vra = partition.parent["vra_score"]
    if proposal_vra > parent_vra:
        return True
    else:
        draw = random.random()
        return draw < bound

def sim_anneal_accept(partition):
    if not partition.parent:
        return True
    proposal_vra = partition["vra_score"]
    parent_vra = partition.parent["vra_score"] 
    if step_Num % cycle_length < start_cool:
         return True

    elif step_Num % cycle_length > stop_cool:
        return proposal_vra > parent_vra

    else: 
        if proposal_vra > parent_vra:
            return True
        else:
            draw = random.random()      
            return draw < (stop_cool - (step_Num % cycle_length))/(stop_cool - start_cool)
       
#define chain
chain = MarkovChain(
    proposal = proposal,
    constraints=[
        constraints.within_percent_of_ideal_population(initial_partition, pop_tol), #% from ideal
    ],
            accept = accept if run_type == 'free' else \
            hill_accept_bound if run_type == 'hill_accept_bound' \
            else sim_anneal_accept,
    initial_state = initial_partition,
    total_steps = total_steps
)

#prep storage for plans
store_plans = pd.DataFrame(columns = ["Index", "GEOID" ])
store_plans["Index"] = list(initial_partition.assignment.keys())
state_gdf_geoid = state_gdf[[geo_id]]
store_plans["GEOID"] = [state_gdf_geoid.iloc[i][0] for i in store_plans["Index"]]

#prep map-wide metric storage
county_splits = []
num_hisp_dists = []
num_black_dists = []
vra_score = []
cut_edges = []
map_metric_df = pd.DataFrame(columns = ["Num Hisp Dists", "Num Black Dists", "County Splits", "VRA score"])

#prep district-by-district storage (each metric in its own df)
#demo data "input"
hisp_prop_df = pd.DataFrame(columns = range(num_districts))
black_prop_df = pd.DataFrame(columns = range(num_districts))
white_prop_df = pd.DataFrame(columns = range(num_districts))

#partisan data "input"
pres16_df = pd.DataFrame(columns = range(num_districts))
pres12_df = pd.DataFrame(columns = range(num_districts))
sen18_df = pd.DataFrame(columns = range(num_districts))
centroids_df = pd.DataFrame(columns = range(num_districts))

elec_dfs = {}
for j in elections:
    elec_dfs[j] =  pd.DataFrame(columns = range(num_districts))

count_moves = 0
temp_score = 0
stuck_step = 0
#run chain and collect data
for step in chain:
    #for storage)  
    if step_Num % store_interval == 0 and step_Num > 0:
        store_plans.to_csv("store_plans_{}.csv".format(run_name), index= False)
        map_metric_df = pd.DataFrame(columns = ["Num Hisp Dists", "Num Black Dists", "County Splits", "VRA score"])
        map_metric_df["County Splits"] = county_splits
        map_metric_df["Num Hisp Dists"] = num_hisp_dists
        map_metric_df["Num Black Dists"] = num_black_dists
        map_metric_df["VRA score"] = vra_score
        map_metric_df["Cut edges"] = cut_edges
        map_metric_df.to_csv("map_metric_df_{}.csv".format(run_name), index = False)
    if step.parent is not None:
        if step.assignment != step.parent.assignment:
            count_moves += 1
            
    #district-by-district storage
    centroids_data = step["centroids"]
    keys = list(centroids_data.keys())
    values = list(centroids_data.values())
    centroids_df.loc[len(centroids_df)] = [value for _,value in sorted(zip(keys,values))]
    
    hisp_prop_data = step["demo_percents"][0]
    keys = list(hisp_prop_data.keys())
    values = list(hisp_prop_data.values())
    hisp_prop_df.loc[len(hisp_prop_df)] = [value for _,value in sorted(zip(keys,values))]    
    
    black_prop_data = step["demo_percents"][1]
    keys = list(black_prop_data.keys())
    values = list(black_prop_data.values())
    black_prop_df.loc[len(black_prop_df)] = [value for _,value in sorted(zip(keys,values))]
    
    white_prop_data = step["demo_percents"][2]
    keys = list(white_prop_data.keys())
    values = list(white_prop_data.values())
    white_prop_df.loc[len(white_prop_df)] = [value for _,value in sorted(zip(keys,values))]
    
    order = [x for x in step.parts]
    percents = {}
    for elec in elections_track:
        percents[elec.name] = dict(zip(order, step[elec.name].percents("Democratic")))
    
    keys = list(percents["PRES16"].keys())
    values = list(percents["PRES16"].values())
    pres16_df.loc[len(pres16_df)] = [value for _,value in sorted(zip(keys,values))]
    
    keys = list(percents["PRES12"].keys())
    values = list(percents["PRES12"].values())
    pres12_df.loc[len(pres12_df)] = [value for _,value in sorted(zip(keys,values))]
    
    keys = list(percents["SEN18"].keys())
    values = list(percents["SEN18"].values())
    sen18_df.loc[len(sen18_df)] = [value for _,value in sorted(zip(keys,values))]
    
    black_prob_dist_dict, hisp_prob_dist_dict, \
    total_black_final, total_hisp_final = step["old_elec_model"]
 
    if step_Num == 0:
        keys = list(black_prob_dist_dict.keys())
        values = list(black_prob_dist_dict.values())
        black_prob_df.loc[len(black_prob_df)] = [value for _,value in sorted(zip(keys,values))] 
        
        keys = list(hisp_prob_dist_dict.keys())
        values = list(hisp_prob_dist_dict.values())
        hisp_prob_df.loc[len(hisp_prob_df)] = [value for _,value in sorted(zip(keys,values))]
               
    else:
        black_prob_df.loc[len(black_prob_df)] = black_prob_df.loc[len(black_prob_df) -1]
        for i in black_prob_dist_dict.keys():
            black_prob_df.at[len(black_prob_df)-1, i] = black_prob_dist_dict[i]
        
        hisp_prob_df.loc[len(hisp_prob_df)] = hisp_prob_df.loc[len(hisp_prob_df) -1]
        for i in hisp_prob_dist_dict.keys():
            hisp_prob_df.at[len(hisp_prob_df)-1, i] = hisp_prob_dist_dict[i]
    
    #map-wide storage    
    county_splits.append(step["num_splits"])
    num_hisp_dists.append(total_hisp_final)
    num_black_dists.append(total_black_final)
    vra_score.append(step["vra_score"])
    cut_edges.append(step["num_cut_edges"])

     #store plans
    if step["vra_score"] >= store_score:
        store_plans["Map{}".format(step_Num)] = store_plans["Index"].map(dict(step.assignment))
    if step["vra_score"] == temp_score:
        stuck_step += 1
    else:
        stuck_step = 0
    if stuck_step == stuck_length:
        print("Run stuck!")
        break
    
    temp_score = step["vra_score"]
    step_Num += 1

#output data
store_plans.to_csv("store_plans_{}.csv".format(run_name), index= False)

#store map-wide data
map_metric_df = pd.DataFrame(columns = ["Num Hisp Dists", "Num Black Dists", "Num Coal Dists", "County Splits", "VRA score"])
map_metric_df["County Splits"] = county_splits
map_metric_df["Num Hisp Dists"] = num_hisp_dists
map_metric_df["Num Black Dists"] = num_black_dists
map_metric_df["VRA score"] = vra_score
map_metric_df["Cut edges"] = cut_edges
map_metric_df.to_csv("map_metric_df_{}.csv".format(run_name), index = False)

#store district-by-district data
#demo data
hisp_prop_df.to_csv("hisp_prop_df_{}.csv".format(run_name), index= False)
black_prop_df.to_csv("black_prop_df_{}.csv".format(run_name), index= False)
white_prop_df.to_csv("white_prop_df_{}.csv".format(run_name), index= False)
#partisan data
pres16_df.to_csv("pres16_df_{}.csv".format(run_name), index = False)
pres12_df.to_csv("pres12_df_{}.csv".format(run_name), index = False)
sen18_df.to_csv("sen18_df_{}.csv".format(run_name), index = False)
centroids_df.to_csv("centroids_df_{}.csv".format(run_name), index = False)
#vra data
black_prob_df.to_csv("black_prob_df_{}.csv".format(run_name), index= False)
hisp_prob_df.to_csv("hisp_prob_df_{}.csv".format(run_name), index= False)

############# final print outs
print("--- %s TOTAL seconds ---" % (time.time() - start_time_total))
print("total moves", count_moves)
print("run name:", run_name)
print("num steps", total_steps)
print("current step", step_Num)
