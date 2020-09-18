import numpy as np
import pandas as pd
import sqlite3
import datetime as dt
from bs4 import BeautifulSoup as BS
from os.path import basename
import time
import requests
import csv
import re
import pickle
from pathlib import Path


def unpickler(file):
        with open(file, 'rb') as f:
            return pickle.load(f)

def goals_ranker(df,target_col,meth):
    ranker = {}
    for team in df.squad.unique():
        df2 = df[(df.squad == team) & (df.dom_comp_lvl == 1)]
        df2['rank_'] = df2[target_col].rank(ascending=meth)
        ranks = {}
        for k, v in zip(df2.season,df2.rank_):
            if k not in ranks.keys():
                ranks[k] = v
        if team not in ranker.keys():
            ranker[team] = ranks
    return ranker

def team_ranker(df,dict_):
    ranker = []
    for team,yr in zip(df.squad,df.season):
        if team in dict_.keys():
            yr_dict = dict_[team]
            if yr in yr_dict.keys():
                ranker.append(yr_dict[yr])
            else:
                ranker.append(0)
    return ranker

def europe_finder(df1,df2):
    europe = []
    col1 = []
    col2 = []
    for x,y in zip(df1.squad,df1.season):
        for j,k in zip(df2.squad,df2.season):
            if (x,y) == (j,k):
                col1.append(y)
                col2.append(1)
            else:
                col1.append((x,0))
    return col1



class LeagueDFEDA(pd.DataFrame):      # create a class for cleaning league dfs for model input
    @property
    def _constructor(self):
        return LeagueDFEDA     

    # def create_position_df(self,position):
    #     return self[self.position.str.contains(position)]
    def pickler(self, output):
        with open(output, 'wb') as f:
            pickle.dump(self,f,pickle.HIGHEST_PROTOCOL)

    # def unpickler(self,file):
    #     with open(file, 'rb') as f:
    #         return pickle.load(f)

    def new_int_col(self,col,new_col,slice=4):
        self[col] = self[col].astype(str)
        self[col] = self[col].str.replace('[','').str.replace("'",'').str.replace(']','').str.replace(',','').str.replace('+','')
        self[new_col] = self[col].apply(lambda x: x[:slice])
        self[new_col] = self[new_col].apply(lambda x: x.replace('','0') if len(x) == 0 else x)
        self[new_col] = self[new_col].astype(int)
        return self

    def col_name_cleaner(self):
        self.columns = self.columns.str.lower().str.replace(' ','_')
        return self

    def season_reducer(self,more_than,less_than):
        return self[(self.season > more_than) & (self.season < less_than)]

    def dom_trophy_counter(self):
        self['trophies'] = self['lg_finish'].apply(lambda x: 1 if x == 1 else 0)
        return self
    
    def dcup_trophy_counter(self):
        self['trophies'] = self['finish'].apply(lambda x: 1 if x == 1 else 0)
        return self

    def col_integerizer(self,col_lst):
        for col in col_lst:
            self[col] = self[col].astype(str)
            self[col] = self[col].astype(int)
        return self

    def int_trophy_counter(self):
        self['trophies'] = self['lgrank'].apply(lambda x: 1 if x == 'W' else 0)
        return self

if __name__ == '__main__':
    epl_dom_lg_fp = '/home/josh/Documents/dsi/caps/cap3/Football_Supporter_Recommender/data/pickles/epl_domestic_league_df_full.pickle'
    epl_dom_cp_fp = '/home/josh/Documents/dsi/caps/cap3/Football_Supporter_Recommender/data/pickles/epl_domestic_cup_df_full.pickle'
    epl_int_cp_fp = '/home/josh/Documents/dsi/caps/cap3/Football_Supporter_Recommender/data/pickles/epl_intnl_cup_df_full.pickle'
    
    epl_lg_df = unpickler(epl_dom_lg_fp)
    epl_dcup_df = unpickler(epl_dom_cp_fp)
    epl_icup_df = unpickler(epl_int_cp_fp)
    
    icup_df = LeagueDFEDA(epl_icup_df)
    icup_df = icup_df.col_name_cleaner()
    icup_df = icup_df.new_int_col('year','season',4)
    icup_df = icup_df.new_int_col('comp','competition',1)           # champions league = 1, europa league = 2
    icup_df = icup_df.new_int_col('attendance','attendance',None)
    icup_df = icup_df.new_int_col('gdiff','gdiff',None)
    icup_df = icup_df.int_trophy_counter()
    icup_df = icup_df.season_reducer(2009,2020)
    icup_df.drop(['comp','top_team_scorer','goalkeeper','year','notes'],axis=1,inplace=True)
    icup_df.pickler('/home/josh/Documents/dsi/caps/cap3/Football_Supporter_Recommender/data/pickles/epl_intnl_cup_df_clean_update.pickle')
    
    dcup_df = LeagueDFEDA(epl_dcup_df)
    dcup_df = dcup_df.col_name_cleaner()
    dcup_df = dcup_df.new_int_col('year','season',4)
    dcup_df = dcup_df.new_int_col('lgrank','finish',-2)
    dcup_df = dcup_df.new_int_col('attendance','attendance',None)
    dcup_df = dcup_df.new_int_col('gdiff','gdiff',None)
    dcup_df['competition'] = dcup_df.comp.apply(lambda x: 1 if x == 'FA Cup' else 2)   # FA Cup = 1, EFL Cup =2 
    dcup_df = dcup_df.dcup_trophy_counter()
    dcup_df = dcup_df.season_reducer(2009,2020)
    dcup_df.drop(['top_team_scorer','goalkeeper','year','lgrank','country','notes','comp'],axis=1,inplace=True)
    dcup_df.pickler('/home/josh/Documents/dsi/caps/cap3/Football_Supporter_Recommender/data/pickles/epl_domestic_cup_df_clean_update.pickle')
    
    dom_df = LeagueDFEDA(epl_lg_df)
    dom_df = dom_df.col_name_cleaner()
    dom_df = dom_df.new_int_col('year','season',4)
    dom_df = dom_df.new_int_col('lgrank','lg_finish',-2)
    dom_df = dom_df.new_int_col('comp','dom_comp_lvl',1)            # premier league = 1
    dom_df = dom_df.new_int_col('attendance','attendance',None)
    dom_df = dom_df.new_int_col('gdiff','gdiff',None)
    dom_df = dom_df.new_int_col('mp','mp',None)
    dom_df = dom_df.new_int_col('w','w',None)
    dom_df = dom_df.new_int_col('l','l',None)
    dom_df = dom_df.new_int_col('d','d',None)
    dom_df = dom_df.new_int_col('pts','pts',None)
    dom_df = dom_df.new_int_col('ga','ga',None)
    dom_df = dom_df.new_int_col('gf','gf',None)
    # dom_df = dom_df.dom_trophy_counter()
    dom_df['europe'] = dom_df.lg_finish.apply(lambda x: 1 if x >= 6 else 0) 
    dom_df = dom_df.season_reducer(2009,2020)
    dom_df['wins'] = dom_df.apply(lambda row: row['w'] * 0.75 if row['dom_comp_lvl'] != 1 else row['w'],axis=1)
    dom_df['losses'] = dom_df.apply(lambda row: row['l'] * 1.25 if row['dom_comp_lvl'] != 1 else row['l'],axis=1)
    dom_df['dom_win%'] = dom_df.wins / dom_df.mp
    dom_df['dom_loss%'] = dom_df.losses / dom_df.mp
    dom_df['dom_draw%'] = dom_df.d / dom_df.mp
    # dom_df['europe'] = dom_df.apply(europe_finder(dom_df,icup_df),axis=1)
    dom_df.country = dom_df.country.str[-3:]
    dom_df.drop(['lgrank','comp','top_team_scorer','goalkeeper','year','notes'],axis=1,inplace=True)
    
    gf_ranks_yr = goals_ranker(dom_df,'gf',False)
    ga_ranks_yr = goals_ranker(dom_df,'ga',True)
    gdiff_ranks_yr = goals_ranker(dom_df,'gdiff',False)
    dom_df['gf_ranks'] = team_ranker(dom_df,gf_ranks_yr)
    dom_df['ga_ranks'] = team_ranker(dom_df,ga_ranks_yr)
    dom_df['gdiff_ranks'] = team_ranker(dom_df,gdiff_ranks_yr)
    print('done')
    dom_df.pickler('/home/josh/Documents/dsi/caps/cap3/Football_Supporter_Recommender/data/pickles/epl_domestic_league_df_clean_update.pickle')
    
    # euro_df = icup_df.groupby(['squad','season']).mean()
    # euro_df['europe'] = 1
    

    # goals_total_df = dom_df.groupby(['season','dom_comp_lvl']).sum()
    # goals_total_df.reset_index(inplace=True)
    # g_total = goals_total_df.copy()
    # gt.drop(['mp','w','d','l','attendance','lg_finish','trophies','dom_win%','dom_loss%','dom_draw%'],axis=1,inplace=True)
    # g_total = g_total.reset_index()
    # g_total = g_total[g_total.dom_comp_lvl == 1] 

    # dom_df['gf_pct'] = dom_df.apply(lambda row: row['gf'] / goals_for_percenterizer(row,dom_df,g_total),axis=0)
    # dom_df['ga_pct'] = dom_df.season.apply(lambda x: goals_for_percenterizer(x,dom_df,g_total,'ga'))
    
  
    # print(g_total.head())
    # print(ga_ranks_yr['Manchester City'])
    # print(icup_df.head())
    # print(dcup_df.head())
    # print(icup_dict['Arsenal'])
    
    # print(dom_df.head())
    # print(icup_df.info())