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


def unpickler(file):            # unpickles saved variables
        with open(file, 'rb') as f:
            return pickle.load(f)

def goals_ranker(df,target_col,meth):       # creates a nested dictionary to rank offensive and defensive stats for teams per season
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

def team_ranker(df,dict_):                  # ranks clubs using the goal_ranker function above
    ranker = []
    for team,yr in zip(df.squad,df.season):
        if team in dict_.keys():
            yr_dict = dict_[team]
            if yr in yr_dict.keys():
                ranker.append(yr_dict[yr])
            else:
                ranker.append(0)
    return ranker

class LeagueDFEDA(pd.DataFrame):      # create a class for cleaning league dfs for similarity calc input
    @property
    def _constructor(self):
        return LeagueDFEDA     

    def pickler(self, output):
        with open(output, 'wb') as f:
            pickle.dump(self,f,pickle.HIGHEST_PROTOCOL)

    def unpickler(self,file):
        with open(file, 'rb') as f:
            return pickle.load(f)

    def new_int_col(self,col,new_col,slice=4):  # cleans string columns and replaces them with int columns
        self[col] = self[col].astype(str)
        self[col] = self[col].str.replace('[','').str.replace("'",'').str.replace(']','').str.replace(',','').str.replace('+','')
        self[new_col] = self[col].apply(lambda x: x[:slice])
        self[new_col] = self[new_col].apply(lambda x: x.replace('','0') if len(x) == 0 else x)
        self[new_col] = self[new_col].astype(int)
        return self

    def col_name_cleaner(self):                 # cleans column names
        self.columns = self.columns.str.lower().str.replace(' ','_')
        return self

    def season_reducer(self,more_than,less_than):       # reduces seasons to be used for comparison
        return self[(self.season > more_than) & (self.season < less_than)]

    def dom_trophy_counter(self):           # counts domestic trophies 
        self['trophies'] = self['lg_finish'].apply(lambda x: 1 if x == 1 else 0)
        return self
    
    def dcup_trophy_counter(self):      #  counts domestic cup trophies
        self['trophies'] = self.apply(lambda row: 1 if row['finish'] == 1 else 0,axis=1)
        return self

    def col_integerizer(self,col_lst):  # turns string columns to int columns
        for col in col_lst:
            self[col] = self[col].astype(str)
            self[col] = self[col].astype(int)
        return self

    def int_trophy_counter(self):   # counts international competition trophies
        self['trophies'] = self['lgrank'].apply(lambda x: 1 if x == 'W' else 0)
        return self

if __name__ == '__main__':
    epl_dom_lg_fp = '/home/josh/Documents/dsi/caps/cap3/Football_Supporter_Recommender/data/pickles/epl_domestic_league_df_full.pickle'     # unpickles saved domestic league stats table
    epl_dom_cp_fp = '/home/josh/Documents/dsi/caps/cap3/Football_Supporter_Recommender/data/pickles/epl_domestic_cup_df_full.pickle'      # unpickles saved domestic cup stats table
    epl_int_cp_fp = '/home/josh/Documents/dsi/caps/cap3/Football_Supporter_Recommender/data/pickles/epl_intnl_cup_df_full.pickle'       # unpickles saved international competition stats
                                                                                                                                        # table
    
    epl_lg_df = unpickler(epl_dom_lg_fp)
    epl_dcup_df = unpickler(epl_dom_cp_fp)
    epl_icup_df = unpickler(epl_int_cp_fp)
    
    icup_df = LeagueDFEDA(epl_icup_df)                              # creates international competition class for eda
    icup_df = icup_df.col_name_cleaner()
    icup_df = icup_df.new_int_col('year','season',4)                # creates a year column as an integer
    icup_df = icup_df.new_int_col('comp','competition',1)           # champions league = 1, europa league = 2
    icup_df = icup_df.new_int_col('attendance','attendance',None)
    icup_df = icup_df.new_int_col('gdiff','gdiff',None)
    icup_df = icup_df.int_trophy_counter()
    icup_df = icup_df.season_reducer(2016,2020)                     # reduces the years used for comparison to 2017 to 2019
    icup_df.drop(['comp','top_team_scorer','goalkeeper','year','notes'],axis=1,inplace=True)    # drops unnecessary columns
    icup_df.pickler('/home/josh/Documents/dsi/caps/cap3/Football_Supporter_Recommender/data/pickles/epl_intnl_cup_df_clean_update.pickle')      # saves the class for the comparison
    
    dcup_df = LeagueDFEDA(epl_dcup_df)                      # creates domestic cup competition class for eda 
    dcup_df = dcup_df.col_name_cleaner()
    dcup_df = dcup_df.new_int_col('year','season',4)        # creates a year column as an integer
    dcup_df = dcup_df.new_int_col('lgrank','finish',-2)
    dcup_df = dcup_df.new_int_col('attendance','attendance',None)
    dcup_df = dcup_df.new_int_col('gdiff','gdiff',None)
    dcup_df['competition'] = dcup_df.comp.apply(lambda x: 1 if x == 'FA Cup' else 2)   # FA Cup = 1, EFL Cup =2 
    dcup_df = dcup_df.dcup_trophy_counter()
    dcup_df = dcup_df.season_reducer(2016,2020)             # reduces the years used for comparison to 2017 to 2019
    dcup_df.drop(['top_team_scorer','goalkeeper','year','lgrank','country','notes','comp'],axis=1,inplace=True)         # drops unnecessary columns
    dcup_df.pickler('/home/josh/Documents/dsi/caps/cap3/Football_Supporter_Recommender/data/pickles/epl_domestic_cup_df_clean_update.pickle')
    
    dom_df = LeagueDFEDA(epl_lg_df)                                  # creates a domestic league competition class for eda
    dom_df = dom_df.col_name_cleaner()  
    dom_df = dom_df.new_int_col('year','season',4)                   # creates a year column as an integer
    dom_df = dom_df.new_int_col('lgrank','lg_finish',-2)
    dom_df = dom_df.new_int_col('comp','dom_comp_lvl',1)             # premier league = 1, championship = 2
    dom_df = dom_df.new_int_col('attendance','attendance',None)      # creates int columns for strings
    dom_df = dom_df.new_int_col('gdiff','gdiff',None)                # creates int columns for strings
    dom_df = dom_df.new_int_col('mp','mp',None)                      # creates int columns for strings
    dom_df = dom_df.new_int_col('w','w',None)                        # creates int columns for strings
    dom_df = dom_df.new_int_col('l','l',None)                        # creates int columns for strings
    dom_df = dom_df.new_int_col('d','d',None)                        # creates int columns for strings
    dom_df = dom_df.new_int_col('pts','pts',None)                    # creates int columns for strings
    dom_df = dom_df.new_int_col('ga','ga',None)                      # creates int columns for strings
    dom_df = dom_df.new_int_col('gf','gf',None)                      # creates int columns for strings
    dom_df = dom_df.dom_trophy_counter()
    dom_df['europe'] = dom_df.apply(lambda row: 1 if (row['lg_finish'] >= 6 & row['dom_comp_lvl'] == 1) else 0,axis=1)     # creates a column to compare to nfl 'playoff' teams
    dom_df = dom_df.season_reducer(2016,2020)                                                                              # reduces the years used for comparison to 2017 to 2019
    dom_df['wins'] = dom_df.apply(lambda row: row['w'] * 0.5 if row['dom_comp_lvl'] != 1 else row['w'],axis=1)             # creates a win column that penalizes win in the championship
    dom_df['losses'] = dom_df.apply(lambda row: row['l'] * 2 if row['dom_comp_lvl'] != 1 else row['l'],axis=1)             # creates a loss column that penalizes losses in the championship
    dom_df['dom_win%'] = dom_df.wins / dom_df.mp                    # creates a win% column to compare to nfl win%
    dom_df['dom_loss%'] = dom_df.losses / dom_df.mp                 # creates a loss% column to compare to nfl loss%
    dom_df['dom_draw%'] = dom_df.d / dom_df.mp                      # creates a draw% column to compare to nfl draw%
    dom_df.country = dom_df.country.str[-3:]                        # reduces country column to only showing the country once
    dom_df.drop(['lgrank','comp','top_team_scorer','goalkeeper','year','notes'],axis=1,inplace=True)    # drops unnecessary columns
    
    gf_ranks_yr = goals_ranker(dom_df,'gf',False)                       # creates offensive rankings for goals scored per season
    ga_ranks_yr = goals_ranker(dom_df,'ga',True)                        # creates defensive rankings for goals allowed per season
    gdiff_ranks_yr = goals_ranker(dom_df,'gdiff',False)                 # creates goal difference rankings per season
    dom_df['gf_ranks'] = team_ranker(dom_df,gf_ranks_yr)                # creates offensive rankings for goals scored per season
    dom_df['ga_ranks'] = team_ranker(dom_df,ga_ranks_yr)                # creates defensive rankings for goals allowed per season
    dom_df['gdiff_ranks'] = team_ranker(dom_df,gdiff_ranks_yr)          # creates goal difference rankings per season
    dom_df.pickler('/home/josh/Documents/dsi/caps/cap3/Football_Supporter_Recommender/data/pickles/epl_domestic_league_df_clean_update.pickle')   # saves the class for comparison
    