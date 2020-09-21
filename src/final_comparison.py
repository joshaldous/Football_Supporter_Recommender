import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler as SScale
from sklearn.neighbors import DistanceMetric
from final_epl_eda import LeagueDFEDA
from scipy import stats
from scipy.spatial.distance import euclidean, jaccard, cosine
import time
import requests
import csv
import re
import pickle

def unpickler(file):
        with open(file, 'rb') as f:
            return pickle.load(f)

def pickler(input, output):
    with open(output, 'wb') as f:
        pickle.dump(input,f,pickle.HIGHEST_PROTOCOL)

class SimilarityDF(pd.DataFrame):         # create a class for standardizing and vectorizing team stats
    @property
    def _constructor(self):
        return SimilarityDF 
    
    def vectorizer(self,team_lst,df_type):          # standardizes and vectorizes the input for all epl teams and a single nfl team
        if df_type == 'EPL':
            temp_df = self.set_index(['squad','season'])
        elif df_type == 'NFL':
            temp_df = self.set_index(['team','year'])
        if len(team_lst) > 1:
            stack_start = temp_df.loc[team_lst[0]]
            stack_start_std = SScale().fit_transform(stack_start).ravel()
            stack_start_std = stack_start_std.reshape(1,-1)
            for team in team_lst[1:]:
                team_df = temp_df.loc[team]
                team_df_std = SScale().fit_transform(team_df).ravel()
                team_df_std = team_df_std.reshape(1,-1)
                stack_start_std = np.concatenate((stack_start_std,team_df_std),axis=0)
        else:
            stack_start = temp_df.loc[team_lst]
            stack_start_std = SScale().fit_transform(stack_start).ravel()
            stack_start_std = stack_start_std.reshape(1,-1)
        return stack_start_std

class Distances():                  # create a class to calculate the distances between vectors for recommendations
    def __init__(self,team_vector,league_matrix,weights=None):
        self.team_vector = team_vector
        self.league_matrix = league_matrix
        self.weights = weights
               
    def euclidean_dist_calc(self,weights):              # calculates the euclidean distance 
        weights = self.weights
        mat_shape = self.league_matrix.shape
        if not weights:
            weights = np.ones((1,mat_shape[1]))
        if self.league_matrix.shape[0] > 1:
            euc_dist = euclidean(self.team_vector,np.matrix(self.league_matrix[0]),weights)
            for u in np.matrix(self.league_matrix[1:]):
                euc = euclidean(self.team_vector,u,weights)
                euc_dist = np.hstack((euc_dist,euc))
        else:
            euc_dist = euclidean(self.team_vector,self.league_matrix,weights)
        return euc_dist
    
    def cosine_sim_calc(self):              # calculates the cosine similarity (not used)
        mat_shape = self.league_matrix.shape
        if self.league_matrix.shape[0] > 1:
            cos_start = np.dot(self.team_vector,np.matrix(self.league_matrix[0]).T)/(np.linalg.norm(self.team_vector) *
                               np.linalg.norm(np.matrix(self.league_matrix[0]))) 
            cos_sim = 0.5 + 0.5 * cos_start
            for u in np.matrix(self.league_matrix[1:]):
                cos_cont = np.dot(self.team_vector,u.T)/(np.linalg.norm(self.team_vector) * np.linalg.norm(u))
                cos_append = 0.5 + 0.5 * cos_cont
                cos_sim = np.hstack((cos_sim,cos_append))
        else:
            costheta = np.dot(self.team_vector,self.league_matrix.T)/(np.linalg.norm(self.team_vector) * 
                              np.linalg.norm(self.league_matrix.T))
            cos_sim = 0.5 + 0.5 * costheta
        return cos_sim
    
    def cosine_dist_calc(self,weights):         # calculates the cosine distance
        weights = self.weights
        mat_shape = self.league_matrix.shape
        if weights == None:
            weights = np.ones((self.league_matrix.shape[1]))
        if mat_shape[0] > 1:
            cos_dist = cosine(self.team_vector,np.matrix(self.league_matrix[0]).T,weights)
            for u in np.matrix(self.league_matrix[1:]):
                cos_cont = cosine(self.team_vector,u.T,weights)
                cos_dist = np.hstack((cos_dist,cos_cont))
        else:
            cos_dist = cosine(self.team_vector,self.league_matrix,weights)
        return cos_dist
    
    def jaccard_dist_calc(self,weights):        # calculates the jaccard distance (not used)
        weights = self.weights
        mat_shape = self.league_matrix.shape
        if not weights:
            weights = np.ones((1,mat_shape[1]))
        if mat_shape[0] > 1:
            jac_dist = jaccard(self.team_vector,np.matrix(self.league_matrix[0]),weights)
            for u in np.matrix(self.league_matrix[1:]):
                jac_cont = jaccard(self.team_vector,u,weights)
                jac_dist = np.hstack((jac_dist,jac_cont))
        else:
            jac_dist = jaccard(self.team_vector,self.league_matrix,weights)
        return jac_dist
    
    def top_dists(self,distance_calc,index_,col,number,weights=None):       # creates an output for distance calculation with teams to compare
        weights = self.weights
        if distance_calc == 'euclidean':
            dist = self.euclidean_dist_calc(weights).reshape(-1,1)
        elif distance_calc == 'cosine_dist':
            dist = self.cosine_dist_calc(weights)
        elif distance_calc == 'cosine_sim':
            dist = self.cosine_sim_calc().reshape(-1,1)
        else:
            dist = self.jaccard_dist_calc(weights).reshape(-1,1)
        df = pd.DataFrame(dist,index=index_,columns=[col])
        top = df.sort_values(by=col,ascending=True)
        return top[:number]

if __name__ == '__main__':
    dom_df = unpickler('/home/josh/Documents/dsi/caps/cap3/Football_Supporter_Recommender/data/pickles/epl_domestic_league_df_clean_update.pickle')  # unpickles the epl team data
    nfl_df = unpickler('/home/josh/Documents/dsi/caps/cap3/Football_Supporter_Recommender/data/pickles/NFL_df_clean_update.pickle')     # unpickles the nfl team data
    
    dom_df.drop(['country','mp','w','wins','losses','gdiff','dom_draw%','lg_finish',           # drops columns from the epl data deemed to not be useful
                 'dom_comp_lvl','d','l','pts','attendance'],axis=1,inplace=True)  
    
    nfl_df.drop(['wins','losses','ties','off_rank_yards','def_rank_yards','turnover_ratio_rank','off_rank_pts',       # drops columns from the nfl data that is deemed to not be useful
                 'def_rank_pts','pt_diff_rank','division_finish', 'points_for','yard_diff_rank','margin_of_vict',
                 'point_difference','playoff_finish','strgth_sched','srs','tie%','off_srs','def_srs'],axis=1,inplace=True)  

    nfl_cols = nfl_df.columns.tolist()
    nfl_cols = ['team','tds','tds_allowed','year','playoffs','win%','loss%']    # reorders columns to match the epl data
    nfl_df = nfl_df.reindex(columns = nfl_cols)
    
    pickler(dom_df,'/home/josh/Documents/dsi/caps/cap3/Football_Supporter_Recommender/data/pickles/EPL_to_vector_update.pickle')        # saves epl data to be standardized
    pickler(nfl_df,'/home/josh/Documents/dsi/caps/cap3/Football_Supporter_Recommender/data/pickles/NFL_to_vector_update.pickle')        # saves nfl data to be standardized
    
    epl_vec = SimilarityDF(dom_df).vectorizer(dom_df.squad.unique(),'EPL')          # creates an array of standardized epl team data
    pickler(epl_vec,'/home/josh/Documents/dsi/caps/cap3/Football_Supporter_Recommender/data/pickles/epl_dists/epl_vectorized_update.pickle')    # saves vectorized epl data
    
    # dom_df = unpickler('/home/josh/Documents/dsi/caps/cap3/Football_Supporter_Recommender/data/pickles/EPL_to_vector_update.pickle')
    # nfl_df = unpickler('/home/josh/Documents/dsi/caps/cap3/Football_Supporter_Recommender/data/pickles/NFL_to_vector_update.pickle')
    epl_mat = unpickler('/home/josh/Documents/dsi/caps/cap3/Football_Supporter_Recommender/data/pickles/epl_dists/epl_vectorized_update.pickle')
    
    weights = [0.08,0.05,0.093,0.08,0.03]     # weights used in distance calculations 
    w = weights*3
    
    ###  Below used to tweak comparisons not used in calculations for recommendation ###

    ## CARDINALS ##                
    print('CARDINALS')
    team_vec = SimilarityDF(nfl_df).vectorizer(['Cardinals'],'NFL')
    team = Distances(team_vec,epl_mat,w)
    team_euc_top = team.top_dists(distance_calc='euclidean',index_=dom_df.squad.unique(),col='euc_dist',number=None)
    team_cos_dist_top = team.top_dists(distance_calc='cosine_dist',index_=dom_df.squad.unique(),col='cos_dist',number=None)
    print(team_euc_top[:5])
    print(team_cos_dist_top[:5])
    team_comp_start = [1 if x == y else 0 for x,y in zip(team_euc_top.index,team_cos_dist_top.index)] 
    
    ## FALCONS ##
    print('FALCONS')
    team_vec = SimilarityDF(nfl_df).vectorizer(['Falcons'],'NFL')
    team = Distances(team_vec,epl_mat,w)
    team_euc_top = team.top_dists(distance_calc='euclidean',index_=dom_df.squad.unique(),col='euc_dist',number=None)
    team_cos_dist_top = team.top_dists(distance_calc='cosine_dist',index_=dom_df.squad.unique(),col='cos_dist',number=None)
    print(team_euc_top[:5])
    print(team_cos_dist_top[:5])
    team_comp_cont = [1 if x == y else 0 for x,y in zip(team_euc_top.index,team_cos_dist_top.index)] 
    team_comp_start = np.hstack((team_comp_start,team_comp_cont))
    
    ## RAVENS ##
    print('RAVENS')
    team_vec = SimilarityDF(nfl_df).vectorizer(['Ravens'],'NFL')
    team = Distances(team_vec,epl_mat,w)
    team_euc_top = team.top_dists(distance_calc='euclidean',index_=dom_df.squad.unique(),col='euc_dist',number=None)
    team_cos_dist_top = team.top_dists(distance_calc='cosine_dist',index_=dom_df.squad.unique(),col='cos_dist',number=None)
    print(team_euc_top[:5])
    print(team_cos_dist_top[:5])
    team_comp_cont = [1 if x == y else 0 for x,y in zip(team_euc_top.index,team_cos_dist_top.index)] 
    team_comp_start = np.hstack((team_comp_start,team_comp_cont))
    
    ## BILLS ##
    print('BILLS')
    team_vec = SimilarityDF(nfl_df).vectorizer(['Bills'],'NFL')
    team = Distances(team_vec,epl_mat,w)
    team_euc_top = team.top_dists(distance_calc='euclidean',index_=dom_df.squad.unique(),col='euc_dist',number=None)
    team_cos_dist_top = team.top_dists(distance_calc='cosine_dist',index_=dom_df.squad.unique(),col='cos_dist',number=None)
    print(team_euc_top[:5])
    print(team_cos_dist_top[:5])
    team_comp_cont = [1 if x == y else 0 for x,y in zip(team_euc_top.index,team_cos_dist_top.index)] 
    team_comp_start = np.hstack((team_comp_start,team_comp_cont))
    
    ## PANTHERS ##
    print('PANTHERS')
    team_vec = SimilarityDF(nfl_df).vectorizer(['Panthers'],'NFL')
    team = Distances(team_vec,epl_mat,w)
    team_euc_top = team.top_dists(distance_calc='euclidean',index_=dom_df.squad.unique(),col='euc_dist',number=None)
    team_cos_dist_top = team.top_dists(distance_calc='cosine_dist',index_=dom_df.squad.unique(),col='cos_dist',number=None)
    print(team_euc_top[:5])
    print(team_cos_dist_top[:5])
    team_comp_cont = [1 if x == y else 0 for x,y in zip(team_euc_top.index,team_cos_dist_top.index)] 
    team_comp_start = np.hstack((team_comp_start,team_comp_cont))
    
    ## BEARS ##
    print('BEARS')
    team_vec = SimilarityDF(nfl_df).vectorizer(['Bears'],'NFL')
    team = Distances(team_vec,epl_mat,w)
    team_euc_top = team.top_dists(distance_calc='euclidean',index_=dom_df.squad.unique(),col='euc_dist',number=None)
    team_cos_dist_top = team.top_dists(distance_calc='cosine_dist',index_=dom_df.squad.unique(),col='cos_dist',number=None)
    print(team_euc_top[:5])
    print(team_cos_dist_top[:5])
    team_comp_cont = [1 if x == y else 0 for x,y in zip(team_euc_top.index,team_cos_dist_top.index)] 
    team_comp_start = np.hstack((team_comp_start,team_comp_cont))
    
    ## BENGALS ##
    print('BENGALS')
    team_vec = SimilarityDF(nfl_df).vectorizer(['Bengals'],'NFL')
    team = Distances(team_vec,epl_mat,w)
    team_euc_top = team.top_dists(distance_calc='euclidean',index_=dom_df.squad.unique(),col='euc_dist',number=None)
    team_cos_dist_top = team.top_dists(distance_calc='cosine_dist',index_=dom_df.squad.unique(),col='cos_dist',number=None)
    print(team_euc_top[:5])
    print(team_cos_dist_top[:5])
    team_comp_cont = [1 if x == y else 0 for x,y in zip(team_euc_top.index,team_cos_dist_top.index)] 
    team_comp_start = np.hstack((team_comp_start,team_comp_cont))
    
    ## BROWNS ##
    print('BROWNS')
    team_vec = SimilarityDF(nfl_df).vectorizer(['Browns'],'NFL')
    team = Distances(team_vec,epl_mat,w)
    team_euc_top = team.top_dists(distance_calc='euclidean',index_=dom_df.squad.unique(),col='euc_dist',number=None)
    team_cos_dist_top = team.top_dists(distance_calc='cosine_dist',index_=dom_df.squad.unique(),col='cos_dist',number=None)
    print(team_euc_top[:5])
    print(team_cos_dist_top[:5])
    team_comp_cont = [1 if x == y else 0 for x,y in zip(team_euc_top.index,team_cos_dist_top.index)] 
    team_comp_start = np.hstack((team_comp_start,team_comp_cont))
    
    ## COWBOYS ##
    print('COWBOYS')
    team_vec = SimilarityDF(nfl_df).vectorizer(['Cowboys'],'NFL')
    team = Distances(team_vec,epl_mat,w)
    team_euc_top = team.top_dists(distance_calc='euclidean',index_=dom_df.squad.unique(),col='euc_dist',number=None)
    team_cos_dist_top = team.top_dists(distance_calc='cosine_dist',index_=dom_df.squad.unique(),col='cos_dist',number=None)
    print(team_euc_top[:5])
    print(team_cos_dist_top[:5])
    team_comp_cont = [1 if x == y else 0 for x,y in zip(team_euc_top.index,team_cos_dist_top.index)] 
    team_comp_start = np.hstack((team_comp_start,team_comp_cont))
    
    ## BRONCOS ##
    print('BRONCOS')
    team_vec = SimilarityDF(nfl_df).vectorizer(['Broncos'],'NFL')
    team = Distances(team_vec,epl_mat,w)
    team_euc_top = team.top_dists(distance_calc='euclidean',index_=dom_df.squad.unique(),col='euc_dist',number=None)
    team_cos_dist_top = team.top_dists(distance_calc='cosine_dist',index_=dom_df.squad.unique(),col='cos_dist',number=None)
    print(team_euc_top[:5])
    print(team_cos_dist_top[:5])
    team_comp_cont = [1 if x == y else 0 for x,y in zip(team_euc_top.index,team_cos_dist_top.index)] 
    team_comp_start = np.hstack((team_comp_start,team_comp_cont))
    
    ## LIONS ##
    print('LIONS')
    team_vec = SimilarityDF(nfl_df).vectorizer(['Lions'],'NFL')
    team = Distances(team_vec,epl_mat,w)
    team_euc_top = team.top_dists(distance_calc='euclidean',index_=dom_df.squad.unique(),col='euc_dist',number=None)
    team_cos_dist_top = team.top_dists(distance_calc='cosine_dist',index_=dom_df.squad.unique(),col='cos_dist',number=None)
    print(team_euc_top[:5])
    print(team_cos_dist_top[:5])
    team_comp_cont = [1 if x == y else 0 for x,y in zip(team_euc_top.index,team_cos_dist_top.index)] 
    team_comp_start = np.hstack((team_comp_start,team_comp_cont))
    
    ## PACKERS ##
    print('PACKERS')
    team_vec = SimilarityDF(nfl_df).vectorizer(['Packers'],'NFL')
    team = Distances(team_vec,epl_mat,w)
    team_euc_top = team.top_dists(distance_calc='euclidean',index_=dom_df.squad.unique(),col='euc_dist',number=None)
    team_cos_dist_top = team.top_dists(distance_calc='cosine_dist',index_=dom_df.squad.unique(),col='cos_dist',number=None)
    print(team_euc_top[:5])
    print(team_cos_dist_top[:5])
    team_comp_cont = [1 if x == y else 0 for x,y in zip(team_euc_top.index,team_cos_dist_top.index)] 
    team_comp_start = np.hstack((team_comp_start,team_comp_cont))
    
    ## TEXANS ##
    print('TEXANS')
    team_vec = SimilarityDF(nfl_df).vectorizer(['Texans'],'NFL')
    team = Distances(team_vec,epl_mat,w)
    team_euc_top = team.top_dists(distance_calc='euclidean',index_=dom_df.squad.unique(),col='euc_dist',number=None)
    team_cos_dist_top = team.top_dists(distance_calc='cosine_dist',index_=dom_df.squad.unique(),col='cos_dist',number=None)
    print(team_euc_top[:5])
    print(team_cos_dist_top[:5])
    team_comp_cont = [1 if x == y else 0 for x,y in zip(team_euc_top.index,team_cos_dist_top.index)] 
    team_comp_start = np.hstack((team_comp_start,team_comp_cont))
    
    ## COLTS ##
    print('COLTS')
    team_vec = SimilarityDF(nfl_df).vectorizer(['Colts'],'NFL')
    team = Distances(team_vec,epl_mat,w)
    team_euc_top = team.top_dists(distance_calc='euclidean',index_=dom_df.squad.unique(),col='euc_dist',number=None)
    team_cos_dist_top = team.top_dists(distance_calc='cosine_dist',index_=dom_df.squad.unique(),col='cos_dist',number=None)
    print(team_euc_top[:5])
    print(team_cos_dist_top[:5])
    team_comp_cont = [1 if x == y else 0 for x,y in zip(team_euc_top.index,team_cos_dist_top.index)] 
    team_comp_start = np.hstack((team_comp_start,team_comp_cont))
    
    ## JAGUARS ##
    print('JAGUARS')
    team_vec = SimilarityDF(nfl_df).vectorizer(['Jaguars'],'NFL')
    team = Distances(team_vec,epl_mat,w)
    team_euc_top = team.top_dists(distance_calc='euclidean',index_=dom_df.squad.unique(),col='euc_dist',number=None)
    team_cos_dist_top = team.top_dists(distance_calc='cosine_dist',index_=dom_df.squad.unique(),col='cos_dist',number=None)
    print(team_euc_top[:5])
    print(team_cos_dist_top[:5])
    team_comp_cont = [1 if x == y else 0 for x,y in zip(team_euc_top.index,team_cos_dist_top.index)] 
    team_comp_start = np.hstack((team_comp_start,team_comp_cont))
    
    ## CHIEFS ##
    print('CHIEFS')
    team_vec = SimilarityDF(nfl_df).vectorizer(['Chiefs'],'NFL')
    team = Distances(team_vec,epl_mat,w)
    team_euc_top = team.top_dists(distance_calc='euclidean',index_=dom_df.squad.unique(),col='euc_dist',number=None)
    team_cos_dist_top = team.top_dists(distance_calc='cosine_dist',index_=dom_df.squad.unique(),col='cos_dist',number=None)
    print(team_euc_top[:5])
    print(team_cos_dist_top[:5])
    team_comp_cont = [1 if x == y else 0 for x,y in zip(team_euc_top.index,team_cos_dist_top.index)] 
    team_comp_start = np.hstack((team_comp_start,team_comp_cont))
    
    ## CHARGERS ##
    print('CHARGERS')
    team_vec = SimilarityDF(nfl_df).vectorizer(['Chargers'],'NFL')
    team = Distances(team_vec,epl_mat,w)
    team_euc_top = team.top_dists(distance_calc='euclidean',index_=dom_df.squad.unique(),col='euc_dist',number=None)
    team_cos_dist_top = team.top_dists(distance_calc='cosine_dist',index_=dom_df.squad.unique(),col='cos_dist',number=None)
    print(team_euc_top[:5])
    print(team_cos_dist_top[:5])
    team_comp_cont = [1 if x == y else 0 for x,y in zip(team_euc_top.index,team_cos_dist_top.index)] 
    team_comp_start = np.hstack((team_comp_start,team_comp_cont))
    
    ## RAMS ##
    print('RAMS')
    team_vec = SimilarityDF(nfl_df).vectorizer(['Rams'],'NFL')
    team = Distances(team_vec,epl_mat,w)
    team_euc_top = team.top_dists(distance_calc='euclidean',index_=dom_df.squad.unique(),col='euc_dist',number=None)
    team_cos_dist_top = team.top_dists(distance_calc='cosine_dist',index_=dom_df.squad.unique(),col='cos_dist',number=None)
    print(team_euc_top[:5])
    print(team_cos_dist_top[:5])
    team_comp_cont = [1 if x == y else 0 for x,y in zip(team_euc_top.index,team_cos_dist_top.index)] 
    team_comp_start = np.hstack((team_comp_start,team_comp_cont))
    
    ## DOLPHINS ##
    print('DOLPHINS')
    team_vec = SimilarityDF(nfl_df).vectorizer(['Dolphins'],'NFL')
    team = Distances(team_vec,epl_mat,w)
    team_euc_top = team.top_dists(distance_calc='euclidean',index_=dom_df.squad.unique(),col='euc_dist',number=None)
    team_cos_dist_top = team.top_dists(distance_calc='cosine_dist',index_=dom_df.squad.unique(),col='cos_dist',number=None)
    print(team_euc_top[:5])
    print(team_cos_dist_top[:5])
    team_comp_cont = [1 if x == y else 0 for x,y in zip(team_euc_top.index,team_cos_dist_top.index)] 
    team_comp_start = np.hstack((team_comp_start,team_comp_cont))
    
    ## VIKINGS ##
    print('VIKINGS')
    team_vec = SimilarityDF(nfl_df).vectorizer(['Vikings'],'NFL')
    team = Distances(team_vec,epl_mat,w)
    team_euc_top = team.top_dists(distance_calc='euclidean',index_=dom_df.squad.unique(),col='euc_dist',number=None)
    team_cos_dist_top = team.top_dists(distance_calc='cosine_dist',index_=dom_df.squad.unique(),col='cos_dist',number=None)
    print(team_euc_top[:5])
    print(team_cos_dist_top[:5])
    team_comp_cont = [1 if x == y else 0 for x,y in zip(team_euc_top.index,team_cos_dist_top.index)] 
    team_comp_start = np.hstack((team_comp_start,team_comp_cont))
    
    ## PATRIOTS ##
    print('PATRIOTS')
    team_vec = SimilarityDF(nfl_df).vectorizer(['Patriots'],'NFL')
    team = Distances(team_vec,epl_mat,w)
    team_euc_top = team.top_dists(distance_calc='euclidean',index_=dom_df.squad.unique(),col='euc_dist',number=None)
    team_cos_dist_top = team.top_dists(distance_calc='cosine_dist',index_=dom_df.squad.unique(),col='cos_dist',number=None)
    print(team_euc_top[:5])
    print(team_cos_dist_top[:5])
    team_comp_cont = [1 if x == y else 0 for x,y in zip(team_euc_top.index,team_cos_dist_top.index)] 
    team_comp_start = np.hstack((team_comp_start,team_comp_cont))
    
    ## SAINTS ##
    print('SAINTS')
    team_vec = SimilarityDF(nfl_df).vectorizer(['Saints'],'NFL')
    team = Distances(team_vec,epl_mat,w)
    team_euc_top = team.top_dists(distance_calc='euclidean',index_=dom_df.squad.unique(),col='euc_dist',number=None)
    team_cos_dist_top = team.top_dists(distance_calc='cosine_dist',index_=dom_df.squad.unique(),col='cos_dist',number=None)
    print(team_euc_top[:5])
    print(team_cos_dist_top[:5])
    team_comp_cont = [1 if x == y else 0 for x,y in zip(team_euc_top.index,team_cos_dist_top.index)] 
    team_comp_start = np.hstack((team_comp_start,team_comp_cont))
    
    ## GIANTS ##
    print('GIANTS')
    team_vec = SimilarityDF(nfl_df).vectorizer(['Giants'],'NFL')
    team = Distances(team_vec,epl_mat,w)
    team_euc_top = team.top_dists(distance_calc='euclidean',index_=dom_df.squad.unique(),col='euc_dist',number=None)
    team_cos_dist_top = team.top_dists(distance_calc='cosine_dist',index_=dom_df.squad.unique(),col='cos_dist',number=None)
    print(team_euc_top[:5])
    print(team_cos_dist_top[:5])
    team_comp_cont = [1 if x == y else 0 for x,y in zip(team_euc_top.index,team_cos_dist_top.index)] 
    team_comp_start = np.hstack((team_comp_start,team_comp_cont))
    
    ## JETS ##
    print('JETS')
    team_vec = SimilarityDF(nfl_df).vectorizer(['Jets'],'NFL')
    team = Distances(team_vec,epl_mat,w)
    team_euc_top = team.top_dists(distance_calc='euclidean',index_=dom_df.squad.unique(),col='euc_dist',number=None)
    team_cos_dist_top = team.top_dists(distance_calc='cosine_dist',index_=dom_df.squad.unique(),col='cos_dist',number=None)
    print(team_euc_top[:5])
    print(team_cos_dist_top[:5])
    team_comp_cont = [1 if x == y else 0 for x,y in zip(team_euc_top.index,team_cos_dist_top.index)] 
    team_comp_start = np.hstack((team_comp_start,team_comp_cont))
    
    ## RAIDERS ##
    print('RAIDERS')
    team_vec = SimilarityDF(nfl_df).vectorizer(['Raiders'],'NFL')
    team = Distances(team_vec,epl_mat,w)
    team_euc_top = team.top_dists(distance_calc='euclidean',index_=dom_df.squad.unique(),col='euc_dist',number=None)
    team_cos_dist_top = team.top_dists(distance_calc='cosine_dist',index_=dom_df.squad.unique(),col='cos_dist',number=None)
    print(team_euc_top[:5])
    print(team_cos_dist_top[:5])
    team_comp_cont = [1 if x == y else 0 for x,y in zip(team_euc_top.index,team_cos_dist_top.index)] 
    team_comp_start = np.hstack((team_comp_start,team_comp_cont))
    
    ## EAGLES ##
    print('EAGLES')
    team_vec = SimilarityDF(nfl_df).vectorizer(['Eagles'],'NFL')
    team = Distances(team_vec,epl_mat,w)
    team_euc_top = team.top_dists(distance_calc='euclidean',index_=dom_df.squad.unique(),col='euc_dist',number=None)
    team_cos_dist_top = team.top_dists(distance_calc='cosine_dist',index_=dom_df.squad.unique(),col='cos_dist',number=None)
    print(team_euc_top[:5])
    print(team_cos_dist_top[:5])
    team_comp_cont = [1 if x == y else 0 for x,y in zip(team_euc_top.index,team_cos_dist_top.index)] 
    team_comp_start = np.hstack((team_comp_start,team_comp_cont))
    
    ## STEELERS ##
    print('STEELERS')
    team_vec = SimilarityDF(nfl_df).vectorizer(['Steelers'],'NFL')
    team = Distances(team_vec,epl_mat,w)
    team_euc_top = team.top_dists(distance_calc='euclidean',index_=dom_df.squad.unique(),col='euc_dist',number=None)
    team_cos_dist_top = team.top_dists(distance_calc='cosine_dist',index_=dom_df.squad.unique(),col='cos_dist',number=None)
    print(team_euc_top[:5])
    print(team_cos_dist_top[:5])
    team_comp_cont = [1 if x == y else 0 for x,y in zip(team_euc_top.index,team_cos_dist_top.index)] 
    team_comp_start = np.hstack((team_comp_start,team_comp_cont))
    
    ## 49ERS ##
    print('49ERS')
    team_vec = SimilarityDF(nfl_df).vectorizer(['49ers'],'NFL')
    team = Distances(team_vec,epl_mat,w)
    team_euc_top = team.top_dists(distance_calc='euclidean',index_=dom_df.squad.unique(),col='euc_dist',number=None)
    team_cos_dist_top = team.top_dists(distance_calc='cosine_dist',index_=dom_df.squad.unique(),col='cos_dist',number=None)
    print(team_euc_top[:5])
    print(team_cos_dist_top[:5])
    team_comp_cont = [1 if x == y else 0 for x,y in zip(team_euc_top.index,team_cos_dist_top.index)] 
    team_comp_start = np.hstack((team_comp_start,team_comp_cont))
    
    ## SEAHAWKS ##
    print('SEAHAWKS')
    team_vec = SimilarityDF(nfl_df).vectorizer(['Seahawks'],'NFL')
    team = Distances(team_vec,epl_mat,w)
    team_euc_top = team.top_dists(distance_calc='euclidean',index_=dom_df.squad.unique(),col='euc_dist',number=None)
    team_cos_dist_top = team.top_dists(distance_calc='cosine_dist',index_=dom_df.squad.unique(),col='cos_dist',number=None)
    print(team_euc_top[:5])
    print(team_cos_dist_top[:5])
    team_comp_cont = [1 if x == y else 0 for x,y in zip(team_euc_top.index,team_cos_dist_top.index)] 
    team_comp_start = np.hstack((team_comp_start,team_comp_cont))
    
    ## BUCCANEERS ##
    print('BUCCANEERS')
    team_vec = SimilarityDF(nfl_df).vectorizer(['Buccaneers'],'NFL')
    team = Distances(team_vec,epl_mat,w)
    team_euc_top = team.top_dists(distance_calc='euclidean',index_=dom_df.squad.unique(),col='euc_dist',number=None)
    team_cos_dist_top = team.top_dists(distance_calc='cosine_dist',index_=dom_df.squad.unique(),col='cos_dist',number=None)
    print(team_euc_top[:5])
    print(team_cos_dist_top[:5])
    team_comp_cont = [1 if x == y else 0 for x,y in zip(team_euc_top.index,team_cos_dist_top.index)] 
    team_comp_start = np.hstack((team_comp_start,team_comp_cont))
    
    ## TITANS ##
    print('TITANS')
    team_vec = SimilarityDF(nfl_df).vectorizer(['Titans'],'NFL')
    team = Distances(team_vec,epl_mat,w)
    team_euc_top = team.top_dists(distance_calc='euclidean',index_=dom_df.squad.unique(),col='euc_dist',number=None)
    team_cos_dist_top = team.top_dists(distance_calc='cosine_dist',index_=dom_df.squad.unique(),col='cos_dist',number=None)
    print(team_euc_top[:5])
    print(team_cos_dist_top[:5])
    team_comp_cont = [1 if x == y else 0 for x,y in zip(team_euc_top.index,team_cos_dist_top.index)] 
    team_comp_start = np.hstack((team_comp_start,team_comp_cont))
    
    ## REDSKINS ##
    print('REDSKINS')
    team_vec = SimilarityDF(nfl_df).vectorizer(['Redskins'],'NFL')
    team = Distances(team_vec,epl_mat,w)
    team_euc_top = team.top_dists(distance_calc='euclidean',index_=dom_df.squad.unique(),col='euc_dist',number=None)
    team_cos_dist_top = team.top_dists(distance_calc='cosine_dist',index_=dom_df.squad.unique(),col='cos_dist',number=None)
    print(team_euc_top[:5])
    print(team_cos_dist_top[:5])
    team_comp_cont = [1 if x == y else 0 for x,y in zip(team_euc_top.index,team_cos_dist_top.index)] 
    team_comp_start = np.hstack((team_comp_start,team_comp_cont))
    
    pickler(team_comp_start,'/home/josh/Documents/dsi/caps/cap3/Football_Supporter_Recommender/data/pickles/nfl_dists/team_comp.pickle')