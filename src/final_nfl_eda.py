import numpy as np
import pandas as pd
import sqlite3
import datetime as dt
from bs4 import BeautifulSoup as BS
from os.path  import basename
import time
import requests
import csv
import re
import pickle
from final_scrape import pickler, unpickler

def new_int_col(df,col):            # turns string cols to int cols
        df[col] = df[col].astype(str)
        df[col] = df[col].str.replace('[','').str.replace("'",'').str.replace(']','').str.replace(',','').str.replace('+','')
        df[col] = df[col].astype(int)
        return df

if __name__ == '__main__':
    nfl_df = unpickler('/home/josh/Documents/dsi/caps/cap3/Football_Supporter_Recommender/data/pickles/working_nfl_df.pickle')   # unpickles nfl stats table 
    nfl_df.columns = ['league','team','wins','losses','ties','division_finish','playoff_finish','points_for','points_against','point_difference',   # renames nfl_df columns
                      'coaches','top_approx_val_plyr','top_passer','top_rusher','top_receiver','off_rank_pts','off_rank_yards','def_rank_pts',
                      'def_rank_yards','turnover_ratio_rank','pt_diff_rank','yard_diff_rank','tot_teams','margin_of_vict','strgth_sched',
                      'srs','off_srs','def_srs','year']
    nfl_df.drop(['league','coaches','top_approx_val_plyr','top_passer','top_rusher','top_receiver','tot_teams'],axis=1,inplace=True)        # drops unnecessary columns
    nfl_df = new_int_col(nfl_df,'year')     
    nfl_df = nfl_df[(nfl_df.year >= 2017) & (nfl_df.year < 2020)]       # reduces years to 2013 - 2019 for comparison
    nfl_df['playoffs'] = nfl_df.playoff_finish.apply(lambda x: 1 if len(x) > 0 else 0)      # creates a column for teams making the playoffs
    nfl_df.team = nfl_df.team.str.replace('*','').str.replace('Team','Redskins').str.split(' ').str[-1]     # cleans team name column
    nfl_df.division_finish = nfl_df.division_finish.str[:1].astype(int)
    nfl_df = new_int_col(nfl_df,'points_for')
    nfl_df = new_int_col(nfl_df,'points_against')
    nfl_df = new_int_col(nfl_df,'point_difference')
    nfl_df = new_int_col(nfl_df,'off_rank_yards')
    nfl_df = new_int_col(nfl_df,'off_rank_pts')
    nfl_df = new_int_col(nfl_df,'def_rank_pts')
    nfl_df = new_int_col(nfl_df,'def_rank_yards')
    nfl_df = new_int_col(nfl_df,'pt_diff_rank')
    nfl_df = new_int_col(nfl_df,'wins')
    nfl_df = new_int_col(nfl_df,'losses')
    nfl_df = new_int_col(nfl_df,'ties')
    nfl_df = new_int_col(nfl_df,'tds')
    nfl_df = new_int_col(nfl_df,'tds_allowed')
    nfl_df['win%'] = nfl_df.wins / 16           # creates a win% column
    nfl_df['loss%'] = nfl_df.losses / 16        # creates a loss% column
    nfl_df['tie%'] = nfl_df.ties / 16           # creates a tie% column
    playoffs_dict = {'Lost Conf':3,'Lost WC':1,'Lost Div':2,'Lost SB':4,'Won SB':5}     # replaces 'lost x' with a numerical value for round of playoff knocked out with extra point for
                                                                                        # winning the super bowl where x is the round of the playoffs
    nfl_df.playoff_finish = nfl_df.playoff_finish.apply(lambda x: 0 if x == '' else x)          # creates a playoff finish column from the dict above
    for k, v in playoffs_dict.items():
        nfl_df.playoff_finish = nfl_df.playoff_finish.apply(lambda x: v if x == k else x)
    pickler(nfl_df,'/home/josh/Documents/dsi/caps/cap3/Football_Supporter_Recommender/data/pickles/NFL_df_clean_update.pickle')     # saves the data for comparison
    # nfl_df = unpickler('/home/josh/Documents/dsi/caps/cap3/Football_Supporter_Recommender/data/pickles/NFL_df_clean.pickle')
    