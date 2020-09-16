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

def name_location_scrapper(url):
    r = requests.get(url)
    soup = BS(r.content,'html.parser')
    tables = soup.find_all('table')
    table_body = tables[0].find_all('tbody')
    body_tds = table_body[0].find_all('td',attrs={'data-stat':'squad'})
    team_link = []
    team_name = []
    for row in body_tds:
        teams = row.find_all('a')
        for team in teams:
            team_link.append(team['href'])
            team_name.append(team.text)
    return team_link, team_name

def epl_link_cleaner(lst_of_team_urls,team_name):
    team_urls = [x.split('/') for x in lst_of_team_urls]
    team_links = ['https://fbref.com/en/squads/'+x[3]+'/history/'+y+'-Stats-and-History'
                  for x,y in zip(team_urls,team_name)]
    return team_links

def pickler(input, output):
    with open(output, 'wb') as f:
        pickle.dump(input,f,pickle.HIGHEST_PROTOCOL)

def unpickler(file):
    with open(file, 'rb') as f:
        return pickle.load(f)

def team_domestic_league_df_creator(lst):
    url = lst[0]
    r = requests.get(url)
    soup = BS(r.content,'html.parser')
    badge = soup.find_all('img',attrs={'class':'teamlogo'})
    badge_pic = badge[0]['src']
    with open(basename(badge_pic),'wb') as f:
        f.write(requests.get(badge_pic).content)
    tables = soup.find_all('table')
    tabs = tables[:3]
    df_dict = {}
    for table in range(len(tabs)):
        bodys = tabs[table].find_all('tbody')
        heads = tabs[table].find_all('thead')
        for head in heads:
            hds = head.find_all('th')
            cols = [hd.text for hd in hds[1:]]
        rows = bodys[0].find_all('tr')
        data = []
        seasons = []
        for row in rows:
            row_tds = row.find_all('td')
            yrs = row.find_all('th',attrs={'scope':'row'})
            yr = [y.text for y in yrs]
            r = [rtd.text for rtd in row_tds]
            data.append(r)
            seasons.append(yr)
        df = pd.DataFrame(data,columns=cols)
        df['year'] = seasons
        df_dict[table] = df
    pickler(df_dict,'df_dict.pickle')
   
    return df_dict

def team_df_appender(lst,dom_df,icup_df,dcup_df):
    for site in lst[1:]:
        url = site
        r = requests.get(url)
        print(url)
        soup = BS(r.content,'lxml')
        badge = soup.find_all('img',attrs={'class':'teamlogo'})
        badge_pic = badge[0]['src']
        with open(basename(badge_pic),'wb') as f:
            f.write(requests.get(badge_pic).content)
        tables = soup.find_all('table')
        df_dict = {}
        caption_text = []
        for tab in tables:
            cap = tab.select('caption')
            for c in cap:
                caption_text.append(c.get_text(strip=True))
        for tabs,caps in zip(range(len(tables)),caption_text):
            df_dict[caps] = tables[tabs]
        for table_name in df_dict.keys():
            bodys = df_dict[table_name].find_all('tbody')
            heads = df_dict[table_name].find_all('thead')
            for head in heads:
                hds = head.find_all('th')
                cols = [hd.text for hd in hds[1:]]
            rows = bodys[0].find_all('tr')
            seasons = []
            data = []
            for row in rows:
                row_tds = row.find_all('td')
                yrs = row.find_all('th',attrs={'scope':'row'})
                yr = [y.text for y in yrs]
                r = [rtd.text for rtd in row_tds]
                data.append(r)
                seasons.append(yr)
            df = pd.DataFrame(data,columns=cols)
            df['year'] = seasons
            if table_name == 'Domestic Leagues Results Table':
                try:
                    dom_df = pd.concat([dom_df,df],axis=0,join='outer')
                    dom_file = r'/home/josh/Documents/dsi/caps/cap3/Football_Supporter_Recommender/data/pickles/epl_domestic_league_df_working.pickle'
                    pickler(dom_df,dom_file)    
                    # dom_file.close()
                except:
                    print(f'{url} dom_league passed!! Try again')
            elif table_name == 'International Cup Results Table':
                try:
                    icup_df = pd.concat([icup_df,df],axis=0,join='outer')
                    icup_file = r'/home/josh/Documents/dsi/caps/cap3/Football_Supporter_Recommender/data/pickles/epl_intnl_cup_df_working.pickle'
                    pickler(icup_df,icup_file)
                    # icup_file.close()
                except:
                    print(f'{url} icup passed!! Try again')
            elif table_name == 'Domestic Cup Results Table':
                try:
                    dcup_df = pd.concat([dcup_df,df],axis=0,join='outer')
                    dcup_file = r'/home/josh/Documents/dsi/caps/cap3/Football_Supporter_Recommender/data/pickles/epl_domestic_cup_df_working.pickle'
                    pickler(dcup_df,dcup_file)
                    # dcup_file.close()
                except:
                    print(f'{url} dcup passed!! Try again')
        wait = np.random.randint(5,size=1)
        time.sleep(wait)
    
    return dom_df, icup_df, dcup_df

def nfl_team_df_creator(lst):
    url = lst[0]
    r = requests.get(url)
    soup = BS(r.content,'html.parser')
    badge = soup.find_all('img',attrs={'class':'teamlogo'})
    badge_pic = badge[0]['src']
    with open(basename(badge_pic),'wb') as f:
        f.write(requests.get(badge_pic).content)
    tables = soup.find_all('table')
    tbod = soup.find_all('tbody')
    thead = soup.find_all('thead')
    head_rows = thead[0].find_all('tr')
    for hr in head_rows:
        hds = hr.find_all('th')
        cols = [hd.text for hd in hds[1:]]
    trows = tbod[0].find_all('tr')
    data = []
    y_played = []
    for tr in trows[:22]:
        tds = tr.find_all('td')
        yrs = tr.find_all('th',attrs={'scope':'row'})
        yr = [y.text for y in yrs]
        row = [td_.text for td_ in tds]
        data.append(row)
        y_played.append(yr)
    df = pd.DataFrame(data,columns=cols)
    df['year'] = y_played
    
    return df

def nfl_df_appender(df_to_append,lst):
    for site in lst[1:]:
        url = site
        print(url)
        r = requests.get(url)
        soup = BS(r.content,'html.parser')
        badge = soup.find_all('img',attrs={'class':'teamlogo'})
        badge_pic = badge[0]['src']
        with open(basename(badge_pic),'wb') as f:
            f.write(requests.get(badge_pic).content)
        tables = soup.find_all('table')
        tbod = soup.find_all('tbody')
        thead = soup.find_all('thead')
        head_rows = thead[0].find_all('tr')
        for hr in head_rows:
            hds = hr.find_all('th')
            cols = [hd.text for hd in hds[1:]]
        trows = tbod[0].find_all('tr')
        data = []
        y_played = []
        for tr in trows[:22]:
            tds = tr.find_all('td')
            yrs = tr.find_all('th',attrs={'scope':'row'})
            yr = [y.text for y in yrs]
            row = [td_.text for td_ in tds]
            data.append(row)
            y_played.append(yr)
        df = pd.DataFrame(data,columns=cols)
        df['year'] = y_played
        df_to_append = df_to_append.append(df)
        pickler(df_to_append,'/home/josh/Documents/dsi/caps/cap3/Football_Supporter_Recommender/data/pickles/working_nfl_df.pickle')
        wait = np.random.randint(5,size=1)
        time.sleep(wait)
    
    return df_to_append

if __name__ == '__main__':
    # team_link, team_name = name_location_scrapper('https://fbref.com/en/players/')
    # team_links = epl_link_cleaner(team_link,team_name)
    
    # pickler(team_links,'team_links.pickle')
    # team_urls = unpickler('/home/josh/Documents/dsi/caps/cap3/Football_Supporter_Recommender/data//pickles/epl_team_links.pickle')
    # team_urls = [x.replace(' ','-') for x in team_urls]
    # team_start_df = team_domestic_league_df_creator(team_urls)
    # team_starter_df = unpickler('/home/josh/Documents/dsi/caps/cap3/Football_Supporter_Recommender/data/pickles/epl_df_dict.pickle')
    
    # domestic_df = team_starter_df[0]
    # intnl_cup_df = team_starter_df[1]
    # dom_cup_df = team_starter_df[2]
    
    # dom_file = '/home/josh/Documents/dsi/caps/cap3/Football_Supporter_Recommender/data/pickles/epl_domestic_league_df.pickle'
    # d_lg_df = unpickler(dom_file)
    # int_file = '/home/josh/Documents/dsi/caps/cap3/Football_Supporter_Recommender/data/pickles/epl_intnl_cup_df.pickle'
    # i_cp_df = unpickler(int_file)
    # domcup_file = '/home/josh/Documents/dsi/caps/cap3/Football_Supporter_Recommender/data/pickles/epl_domestic_cup_df.pickle'
    # d_cp_df = unpickler(domcup_file)
    # domestic_df, intnl_cup_df, dom_cup_df = team_df_appender(lst=team_urls,dom_df=domestic_df,icup_df=intnl_cup_df,dcup_df=dom_cup_df)
    
    # dom_full = r'/home/josh/Documents/dsi/caps/cap3/Football_Supporter_Recommender/data/pickles/epl_domestic_league_df_full.pickle'
    # icup_full = r'/home/josh/Documents/dsi/caps/cap3/Football_Supporter_Recommender/data/pickles/epl_intnl_cup_df_full.pickle'
    # dcup_full = r'/home/josh/Documents/dsi/caps/cap3/Football_Supporter_Recommender/data/pickles/epl_domestic_cup_df_full.pickle'
    
    # pickler(domestic_df,dom_full)
    # pickler(intnl_cup_df,icup_full)
    # pickler(dom_cup_df,dcup_full)

    nfl_team_urls = unpickler('/home/josh/Documents/dsi/caps/cap3/Football_Supporter_Recommender/data/pickles/NFL_team_links.pickle')
    # nfl_team_urls = nfl_link_cleaner(nfl_team_urls)

    # nfl_start_df = nfl_team_df_creator(nfl_team_urls)
    # pickler(nfl_start_df,'/home/josh/Documents/dsi/caps/cap3/Football_Supporter_Recommender/data/pickles/NFL_start_df.pickle')

    nfl_start_df = unpickler('/home/josh/Documents/dsi/caps/cap3/Football_Supporter_Recommender/data/pickles/NFL_start_df.pickle')
    nfl_df = nfl_df_appender(nfl_start_df,nfl_team_urls)
    pickler(nfl_start_df,'/home/josh/Documents/dsi/caps/cap3/Football_Supporter_Recommender/data/pickles/NFL_df.pickle')


