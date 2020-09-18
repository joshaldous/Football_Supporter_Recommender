import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.cluster as cluster
from sklearn.ensemble import RandomForestClassifier as RanForCls
from sklearn.preprocessing import StandardScaler as SScale
from sklearn.neighbors import KNeighborsClassifier as knn
from sklearn.neighbors import DistanceMetric
from sklearn.metrics import silhouette_score as silh_score
from working_eda import LeagueDFEDA
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

# def silh_plot(sample,n_clusters,title=None):
#     plt.style.use('seaborn-dark')
#     fig = plt.figure(figsize=(10,6))
#     ax = fig.add_subplot(111)
#     k = []
#     scores = []
#     for c in range(2,n_clusters+2):
#         k.append(c)
#         clusterer = cluster.KMeans(n_clusters=c)
#         cluster_labels = clusterer.fit_predict(sample)
#         sil_score = silh_score(sample, cluster_labels)
#         print("For n_clusters =", c,
#           "The average silhouette_score is :", sil_score)
#         scores.append(sil_score)
    
#     ax.plot(k,scores)
#     ax.set_xticks(list(range(n_clusters)))
#     ax.set_xlabel('k')
#     ax.set_ylabel('Silhouette Score')
#     plt.title(f'Silhouette Score {title} Plot')
#     plt.savefig(f'/home/josh/Documents/dsi/caps/cap3/Football_Supporter_Recommender/images/silh_score_{title}_{n_clusters}.png')
#     plt.show()

class SimilarityDF(pd.DataFrame):         # create a class for splitting the total df into player position dfs
    @property
    def _constructor(self):
        return SimilarityDF 
    
    def vectorizer(self,team_lst,df_type):
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

class Distances():
    def __init__(self,team_vector,league_matrix,weights=None):
        self.team_vector = team_vector
        self.league_matrix = league_matrix
        self.weights = weights
               
    def euclidean_dist_calc(self,weights):
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
    
    def cosine_sim_calc(self):
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
    
    def cosine_dist_calc(self,weights=None):
        mat_shape = self.league_matrix.shape
        if not weights:
            weights = np.ones((self.league_matrix.shape[1]))
        if mat_shape[0] > 1:
            cos_dist = cosine(self.team_vector,np.matrix(self.league_matrix[0]).T,weights)
            for u in np.matrix(self.league_matrix[1:]):
                # print(f'cos_sci(u): {u.shape}')
                cos_cont = cosine(self.team_vector,u.T,weights)
                cos_dist = np.hstack((cos_dist,cos_cont))
        else:
            cos_dist = cosine(self.team_vector,self.league_matrix,weights)
        return cos_dist
    
    def jaccard_dist_calc(self,weights=None):
        mat_shape = self.league_matrix.shape
        if not weights:
            weights = np.ones((1,mat_shape[1]))
        if mat_shape[0] > 1:
            jac_dist = jaccard(self.team_vector,np.matrix(self.league_matrix[0]),weights)
            for u in np.matrix(self.league_matrix[1:]):
                # print(f'jac(u): {u.shape}')
                jac_cont = jaccard(self.team_vector,u,weights)
                jac_dist = np.hstack((jac_dist,jac_cont))
        else:
            jac_dist = jaccard(self.team_vector,self.league_matrix,weights)
        return jac_dist
    
    def top_dists(self,distance_calc,index_,col,number,weights=None):
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
    dom_df = unpickler('/home/josh/Documents/dsi/caps/cap3/Football_Supporter_Recommender/data/pickles/epl_domestic_league_df_clean_update.pickle')
    # icup_df = unpickler('/home/josh/Documents/dsi/caps/cap3/Football_Supporter_Recommender/data/pickles/epl_intnl_cup_df_clean.pickle')
    # dcup_df = unpickler('/home/josh/Documents/dsi/caps/cap3/Football_Supporter_Recommender/data/pickles/epl_domestic_cup_df_clean.pickle')
    nfl_df = unpickler('/home/josh/Documents/dsi/caps/cap3/Football_Supporter_Recommender/data/pickles/NFL_df_clean_update.pickle')
    # print(dcup_df.info())
    # icup_df.set_index(['squad','season'],inplace=True)
    # print(icup_df.head())
    # efl_cup_df = dcup_df[(dcup_df.comp == 'EFL Cup') | (dcup_df.comp == 'League Cup')]
    # fa_cup_df = dcup_df[dcup_df.comp == 'FA Cup']
    
    # print(efl_cup_df.shape)
    # print(fa_cup_df.shape)
    
    dom_df.drop(['country','mp','w','wins','losses','gdiff','dom_draw%','lg_finish','dom_comp_lvl','d','l','pts','attendance'],axis=1,inplace=True)
    nfl_df.drop(['wins','losses','ties','off_rank_yards','def_rank_yards','turnover_ratio_rank','division_finish',
                 'yard_diff_rank','margin_of_vict','point_difference','playoff_finish','strgth_sched','srs','tie%','off_srs','def_srs'],axis=1,inplace=True)

    nfl_cols = nfl_df.columns.tolist()
    nfl_cols = ['team','points_for','points_against','year','playoffs',      #'division_finish',,'point_difference'
                'win%','loss%','off_rank_pts','def_rank_pts','pt_diff_rank']
    nfl_df = nfl_df.reindex(columns = nfl_cols)
    
    pickler(dom_df,'/home/josh/Documents/dsi/caps/cap3/Football_Supporter_Recommender/data/pickles/EPL_to_vector_update.pickle')
    pickler(nfl_df,'/home/josh/Documents/dsi/caps/cap3/Football_Supporter_Recommender/data/pickles/NFL_to_vector_update.pickle')    
    
    # dom_df_std = SScale().fit_transform(dom_df.set_index(['squad','season']))
    
    # print(dom_df.columns)
    # print(nfl_df.columns)'division_finish',
    # print(dom_df.head(15))
    # print(nfl_df.head(15))



    # epl_teams = dom_df.squad.unique()
    # epl19 = dom_df.loc[dom_df.season == 2019]
    # epl19_gf = SScale().fit(epl19.gf)       #.reshape(1,-1)
    # epl_colors = ['blue' for x in epl_teams]
    # broncs = nfl_df.loc[nfl_df.team == 'Broncos']
    # broncs.points_for = broncs.points_for.astype(int)
    # broncs.off_rank_pts = broncs.off_rank_pts.astype(int)
    # print(np.mean(broncs.points_for))
    # print(np.mean(broncs.off_rank_pts))

    # broncs19  = nfl_df.loc[nfl_df.year == 2019]
    # broncs19 .drop('year',axis=1,inplace=True)
    # broncs19 .points_for = broncs19 .points_for.astype(int)
    # print(broncs19 .columns)
    # nfl_teams = nfl_df.team.unique()
    # nfl_colors = ['red' for x in range(20)]
    # broncs19.point_against = broncs19 .points_against.astype(int)
    # broncs19.point_difference = broncs19 .point_difference.astype(int)
    # broncs19.division_finish = broncs19 .division_finish.astype(int)
    # broncs19.playoffs = broncs19 .playoffs.astype(int)
    # broncs19['win%'] = broncs19 ['win%'].astype(float)
    # broncs19['loss%'] = broncs19 ['loss%'].astype(float)
    # broncs19['tie%'] = broncs19 ['tie%'].astype(float)
    # broncs19.off_rank_pts = broncs19 .off_rank_pts.astype(int)
    # broncs19.def_rank_pts = broncs19 .def_rank_pts.astype(int)
    # broncs19.pt_diff_rank = broncs19 .pt_diff_rank.astype(int)
    # broncs19_sort = broncs19.sort_values('points_for',axis=0)
    # broncs19_sort_top20  = broncs19_sort.iloc[:20,:]
    # broncs19_gf = SScale().fit(broncs19_sort.points_for)

    # plt.style.use('seaborn-dark')
    # fig = plt.figure(figsize=(14,8))
    # ax = fig.add_subplot(111)
    # ax.scatter(epl19_gf,broncs19_gf.points_for,c=[['blue' for _ in len(epl19_gf)],['red' for _ in len(broncs19_gf.points_for)]])
    
    # ax.set_xlabel('EPL Goals For')
    # ax.set_ylabel('NFL Points Scored')
    # ax.set_title('2019 Offensive Scoring: EPL Goals For vs NFL Points For')
    # plt.show()

    # broncs_vec = np.array(broncs19 )

    # epl19 = dom_df[dom_df.season == 2019]
    # epl19_vec = SimilarityDF(epl19).vectorizer(dom_df.squad.unique(),'EPL')
    
    # broncs19 _dist = Distances(broncs_vec,epl19_vec)
    # broncs19 _top = broncs19 _dist.top_dists(distance_calc='euclidean',index_=dom_df.squad.unique(),col='euc_dist',number=None)
    # pickler(broncs19 _top,'/home/josh/Documents/dsi/caps/cap3/Football_Supporter_Recommender/data/pickles/nfl_dists/broncs19 _euclidean.pickle')
    # broncs19 _cos_top = broncs19 _dist.top_dists(distance_calc='cosine_dist',index_=dom_df.squad.unique(),col='cos_dist',number=None)
    # pickler(broncs19 _cos_top,'/home/josh/Documents/dsi/caps/cap3/Football_Supporter_Recommender/data/pickles/nfl_dists/broncs19 _cos_dist.pickle')
    # print(broncs19 _top)
    # print(broncs19 _cos_top)

    # goals_per_game = np.mean(dom_df.gf)
    # print(goals_per_game/38)
    # nfl_df.points_for = nfl_df.points_for.astype(int)
    # nfl_ppg = np.mean(nfl_df.points_for.astype(int))
    # ppg_high = nfl_ppg + 10
    # ppg_low = nfl_ppg - 10
    # # print(nfl_df.loc[(nfl_df.points_for <= ppg_high) & (nfl_df.points_for >= ppg_low)])
    # avg_tds = [38,36,40,40,43,40,40,42,44,38,45,40,44,42,39,42,38,42,39,41]
    # print(len(avg_tds))
    # print(np.mean(avg_tds)/16)
    # print(dom_df.columns)
    # print(dom_df.head())
    # print(nfl_df.columns)
    # print(nfl_df.head())
    
    
    # print(epl19.head())
    # epl19.drop('season',axis=1,inplace=True)
    # broncs19  = nfl_df[nfl_df.year == 2019]
    # print(broncs19 .head())
    # broncs19 .drop('year',axis=1,inplace=True)
    # epl19.set_index(['squad','season'],inplace=True)
    # broncs19 .set_index(['team','year'],inplace=True)

    
    # print(epl19_vec)
    # cs = broncs19 .loc[broncs19 .team =='Cardinals']
    # np.array(cs.set_index(['team','year'],inplace=True))
    # cards19_vec = SimilarityDF(broncs19 ).vectorizer(['Cardinals'],'NFL')
    # print(cards19_vec)
    # fig = plt.figure(figsize=(14,8))
    # ax = fig.add_subplot(111)
    # ax.scatter(broncs19 ,epl19,color=['red','blue'],labels=['Cardinals','EPL Teams'])
    # plt.show()

    # cards19 = Distances()
    # cardinals = Distances(cs,epl19_vec)
    # card_euc_top = cardinals.top_dists(distance_calc='euclidean',index_=dom_df.squad.unique(),col='euc_dist',number=None)
    # pickler(card_euc_top,'/home/josh/Documents/dsi/caps/cap3/Football_Supporter_Recommender/data/pickles/nfl_dists/cards19_euclidean.pickle')
    # card_cos_dist_top = cardinals.top_dists(distance_calc='cosine_dist',index_=dom_df.squad.unique(),col='cos_dist',number=None)
    # pickler(card_cos_dist_top,'/home/josh/Documents/dsi/caps/cap3/Football_Supporter_Recommender/data/pickles/nfl_dists/cards19_cos_dist.pickle')
    # print(card_cos_dist_top)
    
    # nfl_df_std = SScale().fit_transform(nfl_df.set_index(['team','year']))
    
    # silh_plot(dom_df,30,'EPL')_calccap3/Football_Supporter_Recommender/data/pickles/epl_kmeans_4clust.pickle')
    
    # nfl_kmeans = cluster.KMeans(n_clusters=5).fit(nfl_df_std)
    # nfl_labels = nfl_kmeans.labels_
    # nfl_df['kmeans'] = nfl_labels
    # pickler(nfl_df,'/home/josh/Documents/dsi/caps/cap3/Football_Supporter_Recommender/data/pickles/nfl_kmeans_5clust.pickle')
    
    # nfl_df = unpickler('/home/josh/Documents/dsi/caps/cap3/Football_Supporter_Recommender/data/pickles/nfl_kmeans_5clust.pickle')
    # dom_df = unpickler('/home/josh/Documents/dsi/caps/cap3/Football_Supporter_Recommender/data/pickles/epl_kmeans_4clust.pickle')

    # print(euc)
    # print(cos)
    # print(dom_df.head())
    # print(nfl_df.team.unique())
    # print(dom_df.head())
    # print(nfl_df.head())
    # print(dom_df.loc['Arsenal'])
    # print(nfl_df.loc['Cardinals'])

    # arsenal = dom_df.loc['Arsenal']
    # arsenal_std = SScale().fit_transform(arsenal.set_index(['squad','season'])).ravel()
    # np.array().ravel()
    # print(dom_df.index)

    epl_vec = SimilarityDF(dom_df).vectorizer(dom_df.squad.unique(),'EPL')
    # print(epl_vec.shape)
    # epl_vec = epl_vec.vectorizer(dom_df,dom_df.squad.unique(),'EPL')
    # epl_vec = epl_vec.
    pickler(epl_vec,'/home/josh/Documents/dsi/caps/cap3/Football_Supporter_Recommender/data/pickles/epl_dists/epl_vectorized_update.pickle')
    # print(epl_vec.shape)

    # cardinals = nfl_df.set_index(['team','year']).loc['Cardinals']
    # cardinals_std = SScale().fit_transform(cardinals).ravel()
    # cardinals_std = cardinals_std.reshape(1,-1)
    # np.array().ravel()
    # print(cardinals_std.shape)
    dom_df = unpickler('/home/josh/Documents/dsi/caps/cap3/Football_Supporter_Recommender/data/pickles/EPL_to_vector_update.pickle')
    nfl_df = unpickler('/home/josh/Documents/dsi/caps/cap3/Football_Supporter_Recommender/data/pickles/NFL_to_vector_update.pickle')
    epl_mat = unpickler('/home/josh/Documents/dsi/caps/cap3/Football_Supporter_Recommender/data/pickles/epl_dists/epl_vectorized_update.pickle')
    
    w = [0.125,0.8,0.135,0.125,0.11,0.1,0.175,0.15]

    ## CARDINALS ##
    print('CARDINALS')
    team_vec = SimilarityDF(nfl_df).vectorizer(['Cardinals'],'NFL')
    team = Distances(team_vec,epl_mat,w)
    team_euc_top = team.top_dists(distance_calc='euclidean',index_=dom_df.squad.unique(),col='euc_dist',number=None)
    # pickler(team_euc_top,'/home/josh/Documents/dsi/caps/cap3/Football_Supporter_Recommender/data/pickles/nfl_dists/cards_euclidean_update.pickle')
    team_cos_dist_top = team.top_dists(distance_calc='cosine_dist',index_=dom_df.squad.unique(),col='cos_dist',number=None)
    # pickler(card_cos_dist_top,'/home/josh/Documents/dsi/caps/cap3/Football_Supporter_Recommender/data/pickles/nfl_dists/cards_cos_dist_update.pickle')
    # team_cos_sim_top = team.top_dists(distance_calc='cosine_sim',index_=dom_df.squad.unique(),col='cos_sim',number=None)[::-1]
    # pickler(card_cos_sim_top,'/home/josh/Documents/dsi/caps/cap3/Football_Supporter_Recommender/data/pickles/nfl_dists/cards_cos_sim.pickle')
    # team_jac_top = team.top_dists(distance_calc='jaccard',index_=dom_df.squad.unique(),col='jac_dist',number=None)
    # pickler(team_jac_top,'/home/josh/Documents/dsi/caps/cap3/Football_Supporter_Recommender/data/pickles/nfl_dists/cards_jaccard.pickle')
    print(team_euc_top[:5])
    print(team_cos_dist_top[:5])
    team_comp_start = [1 if x == y else 0 for x,y in zip(team_euc_top.index,team_cos_dist_top.index)] 
    # print(card_cos_sim_top)
    # print(card_jac_top)

    ## FALCONS ##
    print('FALCONS')
    team_vec = SimilarityDF(nfl_df).vectorizer(['Falcons'],'NFL')
    team = Distances(team_vec,epl_mat,w)
    team_euc_top = team.top_dists(distance_calc='euclidean',index_=dom_df.squad.unique(),col='euc_dist',number=None)
    # pickler(team_euc_top,'/home/josh/Documents/dsi/caps/cap3/Football_Supporter_Recommender/data/pickles/nfl_dists/cards_euclidean_update.pickle')
    team_cos_dist_top = team.top_dists(distance_calc='cosine_dist',index_=dom_df.squad.unique(),col='cos_dist',number=None)
    # pickler(card_cos_dist_top,'/home/josh/Documents/dsi/caps/cap3/Football_Supporter_Recommender/data/pickles/nfl_dists/cards_cos_dist_update.pickle')
    # team_cos_sim_top = team.top_dists(distance_calc='cosine_sim',index_=dom_df.squad.unique(),col='cos_sim',number=None)[::-1]
    # pickler(card_cos_sim_top,'/home/josh/Documents/dsi/caps/cap3/Football_Supporter_Recommender/data/pickles/nfl_dists/cards_cos_sim.pickle')
    # team_jac_top = team.top_dists(distance_calc='jaccard',index_=dom_df.squad.unique(),col='jac_dist',number=None)
    # pickler(team_jac_top,'/home/josh/Documents/dsi/caps/cap3/Football_Supporter_Recommender/data/pickles/nfl_dists/cards_jaccard.pickle')
    print(team_euc_top[:5])
    print(team_cos_dist_top[:5])
    team_comp_cont = [1 if x == y else 0 for x,y in zip(team_euc_top.index,team_cos_dist_top.index)] 
    team_comp_start = np.hstack((team_comp_start,team_comp_cont))
    # print(card_cos_sim_top)
    # print(card_jac_top)

    ## RAVENS ##
    print('RAVENS')
    team_vec = SimilarityDF(nfl_df).vectorizer(['Ravens'],'NFL')
    team = Distances(team_vec,epl_mat,w)
    team_euc_top = team.top_dists(distance_calc='euclidean',index_=dom_df.squad.unique(),col='euc_dist',number=None)
    # pickler(team_euc_top,'/home/josh/Documents/dsi/caps/cap3/Football_Supporter_Recommender/data/pickles/nfl_dists/cards_euclidean_update.pickle')
    team_cos_dist_top = team.top_dists(distance_calc='cosine_dist',index_=dom_df.squad.unique(),col='cos_dist',number=None)
    # pickler(card_cos_dist_top,'/home/josh/Documents/dsi/caps/cap3/Football_Supporter_Recommender/data/pickles/nfl_dists/cards_cos_dist_update.pickle')
    # team_cos_sim_top = team.top_dists(distance_calc='cosine_sim',index_=dom_df.squad.unique(),col='cos_sim',number=None)[::-1]
    # pickler(card_cos_sim_top,'/home/josh/Documents/dsi/caps/cap3/Football_Supporter_Recommender/data/pickles/nfl_dists/cards_cos_sim.pickle')
    # team_jac_top = team.top_dists(distance_calc='jaccard',index_=dom_df.squad.unique(),col='jac_dist',number=None)
    # pickler(team_jac_top,'/home/josh/Documents/dsi/caps/cap3/Football_Supporter_Recommender/data/pickles/nfl_dists/cards_jaccard.pickle')
    print(team_euc_top[:5])
    print(team_cos_dist_top[:5])
    team_comp_cont = [1 if x == y else 0 for x,y in zip(team_euc_top.index,team_cos_dist_top.index)] 
    team_comp_start = np.hstack((team_comp_start,team_comp_cont))
    # print(card_cos_sim_top)
    # print(card_jac_top)

    ## BILLS ##
    print('BILLS')
    team_vec = SimilarityDF(nfl_df).vectorizer(['Bills'],'NFL')
    team = Distances(team_vec,epl_mat,w)
    team_euc_top = team.top_dists(distance_calc='euclidean',index_=dom_df.squad.unique(),col='euc_dist',number=None)
    # pickler(team_euc_top,'/home/josh/Documents/dsi/caps/cap3/Football_Supporter_Recommender/data/pickles/nfl_dists/cards_euclidean_update.pickle')
    team_cos_dist_top = team.top_dists(distance_calc='cosine_dist',index_=dom_df.squad.unique(),col='cos_dist',number=None)
    # pickler(card_cos_dist_top,'/home/josh/Documents/dsi/caps/cap3/Football_Supporter_Recommender/data/pickles/nfl_dists/cards_cos_dist_update.pickle')
    # team_cos_sim_top = team.top_dists(distance_calc='cosine_sim',index_=dom_df.squad.unique(),col='cos_sim',number=None)[::-1]
    # pickler(card_cos_sim_top,'/home/josh/Documents/dsi/caps/cap3/Football_Supporter_Recommender/data/pickles/nfl_dists/cards_cos_sim.pickle')
    # team_jac_top = team.top_dists(distance_calc='jaccard',index_=dom_df.squad.unique(),col='jac_dist',number=None)
    # pickler(team_jac_top,'/home/josh/Documents/dsi/caps/cap3/Football_Supporter_Recommender/data/pickles/nfl_dists/cards_jaccard.pickle')
    print(team_euc_top[:5])
    print(team_cos_dist_top[:5])
    team_comp_cont = [1 if x == y else 0 for x,y in zip(team_euc_top.index,team_cos_dist_top.index)] 
    team_comp_start = np.hstack((team_comp_start,team_comp_cont))
    # print(card_cos_sim_top)
    # print(card_jac_top)

    ## PANTHERS ##
    print('PANTHERS')
    team_vec = SimilarityDF(nfl_df).vectorizer(['Panthers'],'NFL')
    team = Distances(team_vec,epl_mat,w)
    team_euc_top = team.top_dists(distance_calc='euclidean',index_=dom_df.squad.unique(),col='euc_dist',number=None)
    # pickler(team_euc_top,'/home/josh/Documents/dsi/caps/cap3/Football_Supporter_Recommender/data/pickles/nfl_dists/cards_euclidean_update.pickle')
    team_cos_dist_top = team.top_dists(distance_calc='cosine_dist',index_=dom_df.squad.unique(),col='cos_dist',number=None)
    # pickler(card_cos_dist_top,'/home/josh/Documents/dsi/caps/cap3/Football_Supporter_Recommender/data/pickles/nfl_dists/cards_cos_dist_update.pickle')
    # team_cos_sim_top = team.top_dists(distance_calc='cosine_sim',index_=dom_df.squad.unique(),col='cos_sim',number=None)[::-1]
    # pickler(card_cos_sim_top,'/home/josh/Documents/dsi/caps/cap3/Football_Supporter_Recommender/data/pickles/nfl_dists/cards_cos_sim.pickle')
    # team_jac_top = team.top_dists(distance_calc='jaccard',index_=dom_df.squad.unique(),col='jac_dist',number=None)
    # pickler(team_jac_top,'/home/josh/Documents/dsi/caps/cap3/Football_Supporter_Recommender/data/pickles/nfl_dists/cards_jaccard.pickle')
    print(team_euc_top[:5])
    print(team_cos_dist_top[:5])
    team_comp_cont = [1 if x == y else 0 for x,y in zip(team_euc_top.index,team_cos_dist_top.index)] 
    team_comp_start = np.hstack((team_comp_start,team_comp_cont))
    # print(card_cos_sim_top)
    # print(card_jac_top)

    ## BEARS ##
    print('BEARS')
    team_vec = SimilarityDF(nfl_df).vectorizer(['Bears'],'NFL')
    team = Distances(team_vec,epl_mat,w)
    team_euc_top = team.top_dists(distance_calc='euclidean',index_=dom_df.squad.unique(),col='euc_dist',number=None)
    # pickler(team_euc_top,'/home/josh/Documents/dsi/caps/cap3/Football_Supporter_Recommender/data/pickles/nfl_dists/cards_euclidean_update.pickle')
    team_cos_dist_top = team.top_dists(distance_calc='cosine_dist',index_=dom_df.squad.unique(),col='cos_dist',number=None)
    # pickler(card_cos_dist_top,'/home/josh/Documents/dsi/caps/cap3/Football_Supporter_Recommender/data/pickles/nfl_dists/cards_cos_dist_update.pickle')
    # team_cos_sim_top = team.top_dists(distance_calc='cosine_sim',index_=dom_df.squad.unique(),col='cos_sim',number=None)[::-1]
    # pickler(card_cos_sim_top,'/home/josh/Documents/dsi/caps/cap3/Football_Supporter_Recommender/data/pickles/nfl_dists/cards_cos_sim.pickle')
    # team_jac_top = team.top_dists(distance_calc='jaccard',index_=dom_df.squad.unique(),col='jac_dist',number=None)
    # pickler(team_jac_top,'/home/josh/Documents/dsi/caps/cap3/Football_Supporter_Recommender/data/pickles/nfl_dists/cards_jaccard.pickle')
    print(team_euc_top[:5])
    print(team_cos_dist_top[:5])
    team_comp_cont = [1 if x == y else 0 for x,y in zip(team_euc_top.index,team_cos_dist_top.index)] 
    team_comp_start = np.hstack((team_comp_start,team_comp_cont))
    # print(card_cos_sim_top)
    # print(card_jac_top)

    ## BENGALS ##
    print('BENGALS')
    team_vec = SimilarityDF(nfl_df).vectorizer(['Bengals'],'NFL')
    team = Distances(team_vec,epl_mat,w)
    team_euc_top = team.top_dists(distance_calc='euclidean',index_=dom_df.squad.unique(),col='euc_dist',number=None)
    # pickler(team_euc_top,'/home/josh/Documents/dsi/caps/cap3/Football_Supporter_Recommender/data/pickles/nfl_dists/cards_euclidean_update.pickle')
    team_cos_dist_top = team.top_dists(distance_calc='cosine_dist',index_=dom_df.squad.unique(),col='cos_dist',number=None)
    # pickler(card_cos_dist_top,'/home/josh/Documents/dsi/caps/cap3/Football_Supporter_Recommender/data/pickles/nfl_dists/cards_cos_dist_update.pickle')
    # team_cos_sim_top = team.top_dists(distance_calc='cosine_sim',index_=dom_df.squad.unique(),col='cos_sim',number=None)[::-1]
    # pickler(card_cos_sim_top,'/home/josh/Documents/dsi/caps/cap3/Football_Supporter_Recommender/data/pickles/nfl_dists/cards_cos_sim.pickle')
    # team_jac_top = team.top_dists(distance_calc='jaccard',index_=dom_df.squad.unique(),col='jac_dist',number=None)
    # pickler(team_jac_top,'/home/josh/Documents/dsi/caps/cap3/Football_Supporter_Recommender/data/pickles/nfl_dists/cards_jaccard.pickle')
    print(team_euc_top[:5])
    print(team_cos_dist_top[:5])
    team_comp_cont = [1 if x == y else 0 for x,y in zip(team_euc_top.index,team_cos_dist_top.index)] 
    team_comp_start = np.hstack((team_comp_start,team_comp_cont))
    # print(card_cos_sim_top)
    # print(card_jac_top)

    ## BROWNS ##
    print('BROWNS')
    team_vec = SimilarityDF(nfl_df).vectorizer(['Browns'],'NFL')
    team = Distances(team_vec,epl_mat,w)
    team_euc_top = team.top_dists(distance_calc='euclidean',index_=dom_df.squad.unique(),col='euc_dist',number=None)
    # pickler(team_euc_top,'/home/josh/Documents/dsi/caps/cap3/Football_Supporter_Recommender/data/pickles/nfl_dists/cards_euclidean_update.pickle')
    team_cos_dist_top = team.top_dists(distance_calc='cosine_dist',index_=dom_df.squad.unique(),col='cos_dist',number=None)
    # pickler(card_cos_dist_top,'/home/josh/Documents/dsi/caps/cap3/Football_Supporter_Recommender/data/pickles/nfl_dists/cards_cos_dist_update.pickle')
    # team_cos_sim_top = team.top_dists(distance_calc='cosine_sim',index_=dom_df.squad.unique(),col='cos_sim',number=None)[::-1]
    # pickler(card_cos_sim_top,'/home/josh/Documents/dsi/caps/cap3/Football_Supporter_Recommender/data/pickles/nfl_dists/cards_cos_sim.pickle')
    # team_jac_top = team.top_dists(distance_calc='jaccard',index_=dom_df.squad.unique(),col='jac_dist',number=None)
    # pickler(team_jac_top,'/home/josh/Documents/dsi/caps/cap3/Football_Supporter_Recommender/data/pickles/nfl_dists/cards_jaccard.pickle')
    print(team_euc_top[:5])
    print(team_cos_dist_top[:5])
    team_comp_cont = [1 if x == y else 0 for x,y in zip(team_euc_top.index,team_cos_dist_top.index)] 
    team_comp_start = np.hstack((team_comp_start,team_comp_cont))
    # print(card_cos_sim_top)
    # print(card_jac_top)

    ## COWBOYS ##
    print('COWBOYS')
    team_vec = SimilarityDF(nfl_df).vectorizer(['Cowboys'],'NFL')
    team = Distances(team_vec,epl_mat,w)
    team_euc_top = team.top_dists(distance_calc='euclidean',index_=dom_df.squad.unique(),col='euc_dist',number=None)
    # pickler(team_euc_top,'/home/josh/Documents/dsi/caps/cap3/Football_Supporter_Recommender/data/pickles/nfl_dists/cards_euclidean_update.pickle')
    team_cos_dist_top = team.top_dists(distance_calc='cosine_dist',index_=dom_df.squad.unique(),col='cos_dist',number=None)
    # pickler(card_cos_dist_top,'/home/josh/Documents/dsi/caps/cap3/Football_Supporter_Recommender/data/pickles/nfl_dists/cards_cos_dist_update.pickle')
    # team_cos_sim_top = team.top_dists(distance_calc='cosine_sim',index_=dom_df.squad.unique(),col='cos_sim',number=None)[::-1]
    # pickler(card_cos_sim_top,'/home/josh/Documents/dsi/caps/cap3/Football_Supporter_Recommender/data/pickles/nfl_dists/cards_cos_sim.pickle')
    # team_jac_top = team.top_dists(distance_calc='jaccard',index_=dom_df.squad.unique(),col='jac_dist',number=None)
    # pickler(team_jac_top,'/home/josh/Documents/dsi/caps/cap3/Football_Supporter_Recommender/data/pickles/nfl_dists/cards_jaccard.pickle')
    print(team_euc_top[:5])
    print(team_cos_dist_top[:5])
    team_comp_cont = [1 if x == y else 0 for x,y in zip(team_euc_top.index,team_cos_dist_top.index)] 
    team_comp_start = np.hstack((team_comp_start,team_comp_cont))
    # print(card_cos_sim_top)
    # print(card_jac_top)

    ## BRONCOS ##
    print('BRONCOS')
    team_vec = SimilarityDF(nfl_df).vectorizer(['Broncos'],'NFL')
    team = Distances(team_vec,epl_mat,w)
    team_euc_top = team.top_dists(distance_calc='euclidean',index_=dom_df.squad.unique(),col='euc_dist',number=None)
    # pickler(team_euc_top,'/home/josh/Documents/dsi/caps/cap3/Football_Supporter_Recommender/data/pickles/nfl_dists/cards_euclidean_update.pickle')
    team_cos_dist_top = team.top_dists(distance_calc='cosine_dist',index_=dom_df.squad.unique(),col='cos_dist',number=None)
    # pickler(card_cos_dist_top,'/home/josh/Documents/dsi/caps/cap3/Football_Supporter_Recommender/data/pickles/nfl_dists/cards_cos_dist_update.pickle')
    # team_cos_sim_top = team.top_dists(distance_calc='cosine_sim',index_=dom_df.squad.unique(),col='cos_sim',number=None)[::-1]
    # pickler(card_cos_sim_top,'/home/josh/Documents/dsi/caps/cap3/Football_Supporter_Recommender/data/pickles/nfl_dists/cards_cos_sim.pickle')
    # team_jac_top = team.top_dists(distance_calc='jaccard',index_=dom_df.squad.unique(),col='jac_dist',number=None)
    # pickler(team_jac_top,'/home/josh/Documents/dsi/caps/cap3/Football_Supporter_Recommender/data/pickles/nfl_dists/cards_jaccard.pickle')
    print(team_euc_top[:5])
    print(team_cos_dist_top[:5])
    team_comp_cont = [1 if x == y else 0 for x,y in zip(team_euc_top.index,team_cos_dist_top.index)] 
    team_comp_start = np.hstack((team_comp_start,team_comp_cont))
    # print(card_cos_sim_top)
    # print(card_jac_top)

    ## LIONS ##
    print('LIONS')
    team_vec = SimilarityDF(nfl_df).vectorizer(['Lions'],'NFL')
    team = Distances(team_vec,epl_mat,w)
    team_euc_top = team.top_dists(distance_calc='euclidean',index_=dom_df.squad.unique(),col='euc_dist',number=None)
    # pickler(team_euc_top,'/home/josh/Documents/dsi/caps/cap3/Football_Supporter_Recommender/data/pickles/nfl_dists/cards_euclidean_update.pickle')
    team_cos_dist_top = team.top_dists(distance_calc='cosine_dist',index_=dom_df.squad.unique(),col='cos_dist',number=None)
    # pickler(card_cos_dist_top,'/home/josh/Documents/dsi/caps/cap3/Football_Supporter_Recommender/data/pickles/nfl_dists/cards_cos_dist_update.pickle')
    # team_cos_sim_top = team.top_dists(distance_calc='cosine_sim',index_=dom_df.squad.unique(),col='cos_sim',number=None)[::-1]
    # pickler(card_cos_sim_top,'/home/josh/Documents/dsi/caps/cap3/Football_Supporter_Recommender/data/pickles/nfl_dists/cards_cos_sim.pickle')
    # team_jac_top = team.top_dists(distance_calc='jaccard',index_=dom_df.squad.unique(),col='jac_dist',number=None)
    # pickler(team_jac_top,'/home/josh/Documents/dsi/caps/cap3/Football_Supporter_Recommender/data/pickles/nfl_dists/cards_jaccard.pickle')
    print(team_euc_top[:5])
    print(team_cos_dist_top[:5])
    team_comp_cont = [1 if x == y else 0 for x,y in zip(team_euc_top.index,team_cos_dist_top.index)] 
    team_comp_start = np.hstack((team_comp_start,team_comp_cont))
    # print(card_cos_sim_top)
    # print(card_jac_top)

    ## PACKERS ##
    print('PACKERS')
    team_vec = SimilarityDF(nfl_df).vectorizer(['Packers'],'NFL')
    team = Distances(team_vec,epl_mat,w)
    team_euc_top = team.top_dists(distance_calc='euclidean',index_=dom_df.squad.unique(),col='euc_dist',number=None)
    # pickler(team_euc_top,'/home/josh/Documents/dsi/caps/cap3/Football_Supporter_Recommender/data/pickles/nfl_dists/cards_euclidean_update.pickle')
    team_cos_dist_top = team.top_dists(distance_calc='cosine_dist',index_=dom_df.squad.unique(),col='cos_dist',number=None)
    # pickler(card_cos_dist_top,'/home/josh/Documents/dsi/caps/cap3/Football_Supporter_Recommender/data/pickles/nfl_dists/cards_cos_dist_update.pickle')
    # team_cos_sim_top = team.top_dists(distance_calc='cosine_sim',index_=dom_df.squad.unique(),col='cos_sim',number=None)[::-1]
    # pickler(card_cos_sim_top,'/home/josh/Documents/dsi/caps/cap3/Football_Supporter_Recommender/data/pickles/nfl_dists/cards_cos_sim.pickle')
    # team_jac_top = team.top_dists(distance_calc='jaccard',index_=dom_df.squad.unique(),col='jac_dist',number=None)
    # pickler(team_jac_top,'/home/josh/Documents/dsi/caps/cap3/Football_Supporter_Recommender/data/pickles/nfl_dists/cards_jaccard.pickle')
    print(team_euc_top[:5])
    print(team_cos_dist_top[:5])
    team_comp_cont = [1 if x == y else 0 for x,y in zip(team_euc_top.index,team_cos_dist_top.index)] 
    team_comp_start = np.hstack((team_comp_start,team_comp_cont))
    # print(card_cos_sim_top)
    # print(card_jac_top)

    ## TEXANS ##
    print('TEXANS')
    team_vec = SimilarityDF(nfl_df).vectorizer(['Texans'],'NFL')
    team = Distances(team_vec,epl_mat,w)
    team_euc_top = team.top_dists(distance_calc='euclidean',index_=dom_df.squad.unique(),col='euc_dist',number=None)
    # pickler(team_euc_top,'/home/josh/Documents/dsi/caps/cap3/Football_Supporter_Recommender/data/pickles/nfl_dists/cards_euclidean_update.pickle')
    team_cos_dist_top = team.top_dists(distance_calc='cosine_dist',index_=dom_df.squad.unique(),col='cos_dist',number=None)
    # pickler(card_cos_dist_top,'/home/josh/Documents/dsi/caps/cap3/Football_Supporter_Recommender/data/pickles/nfl_dists/cards_cos_dist_update.pickle')
    # team_cos_sim_top = team.top_dists(distance_calc='cosine_sim',index_=dom_df.squad.unique(),col='cos_sim',number=None)[::-1]
    # pickler(card_cos_sim_top,'/home/josh/Documents/dsi/caps/cap3/Football_Supporter_Recommender/data/pickles/nfl_dists/cards_cos_sim.pickle')
    # team_jac_top = team.top_dists(distance_calc='jaccard',index_=dom_df.squad.unique(),col='jac_dist',number=None)
    # pickler(team_jac_top,'/home/josh/Documents/dsi/caps/cap3/Football_Supporter_Recommender/data/pickles/nfl_dists/cards_jaccard.pickle')
    print(team_euc_top[:5])
    print(team_cos_dist_top[:5])
    team_comp_cont = [1 if x == y else 0 for x,y in zip(team_euc_top.index,team_cos_dist_top.index)] 
    team_comp_start = np.hstack((team_comp_start,team_comp_cont))
    # print(card_cos_sim_top)
    # print(card_jac_top)
    
    ## COLTS ##
    print('COLTS')
    team_vec = SimilarityDF(nfl_df).vectorizer(['Colts'],'NFL')
    team = Distances(team_vec,epl_mat,w)
    team_euc_top = team.top_dists(distance_calc='euclidean',index_=dom_df.squad.unique(),col='euc_dist',number=None)
    # pickler(team_euc_top,'/home/josh/Documents/dsi/caps/cap3/Football_Supporter_Recommender/data/pickles/nfl_dists/cards_euclidean_update.pickle')
    team_cos_dist_top = team.top_dists(distance_calc='cosine_dist',index_=dom_df.squad.unique(),col='cos_dist',number=None)
    # pickler(card_cos_dist_top,'/home/josh/Documents/dsi/caps/cap3/Football_Supporter_Recommender/data/pickles/nfl_dists/cards_cos_dist_update.pickle')
    # team_cos_sim_top = team.top_dists(distance_calc='cosine_sim',index_=dom_df.squad.unique(),col='cos_sim',number=None)[::-1]
    # pickler(card_cos_sim_top,'/home/josh/Documents/dsi/caps/cap3/Football_Supporter_Recommender/data/pickles/nfl_dists/cards_cos_sim.pickle')
    # team_jac_top = team.top_dists(distance_calc='jaccard',index_=dom_df.squad.unique(),col='jac_dist',number=None)
    # pickler(team_jac_top,'/home/josh/Documents/dsi/caps/cap3/Football_Supporter_Recommender/data/pickles/nfl_dists/cards_jaccard.pickle')
    print(team_euc_top[:5])
    print(team_cos_dist_top[:5])
    team_comp_cont = [1 if x == y else 0 for x,y in zip(team_euc_top.index,team_cos_dist_top.index)] 
    team_comp_start = np.hstack((team_comp_start,team_comp_cont))
    # print(card_cos_sim_top)
    # print(card_jac_top)

    ## JAGUARS ##
    print('JAGUARS')
    team_vec = SimilarityDF(nfl_df).vectorizer(['Jaguars'],'NFL')
    team = Distances(team_vec,epl_mat,w)
    team_euc_top = team.top_dists(distance_calc='euclidean',index_=dom_df.squad.unique(),col='euc_dist',number=None)
    # pickler(team_euc_top,'/home/josh/Documents/dsi/caps/cap3/Football_Supporter_Recommender/data/pickles/nfl_dists/cards_euclidean_update.pickle')
    team_cos_dist_top = team.top_dists(distance_calc='cosine_dist',index_=dom_df.squad.unique(),col='cos_dist',number=None)
    # pickler(card_cos_dist_top,'/home/josh/Documents/dsi/caps/cap3/Football_Supporter_Recommender/data/pickles/nfl_dists/cards_cos_dist_update.pickle')
    # team_cos_sim_top = team.top_dists(distance_calc='cosine_sim',index_=dom_df.squad.unique(),col='cos_sim',number=None)[::-1]
    # pickler(card_cos_sim_top,'/home/josh/Documents/dsi/caps/cap3/Football_Supporter_Recommender/data/pickles/nfl_dists/cards_cos_sim.pickle')
    # team_jac_top = team.top_dists(distance_calc='jaccard',index_=dom_df.squad.unique(),col='jac_dist',number=None)
    # pickler(team_jac_top,'/home/josh/Documents/dsi/caps/cap3/Football_Supporter_Recommender/data/pickles/nfl_dists/cards_jaccard.pickle')
    print(team_euc_top[:5])
    print(team_cos_dist_top[:5])
    team_comp_cont = [1 if x == y else 0 for x,y in zip(team_euc_top.index,team_cos_dist_top.index)] 
    team_comp_start = np.hstack((team_comp_start,team_comp_cont))
    # print(card_cos_sim_top)
    # print(card_jac_top) 

    ## CHIEFS ##
    print('CHIEFS')
    team_vec = SimilarityDF(nfl_df).vectorizer(['Chiefs'],'NFL')
    team = Distances(team_vec,epl_mat,w)
    team_euc_top = team.top_dists(distance_calc='euclidean',index_=dom_df.squad.unique(),col='euc_dist',number=None)
    # pickler(team_euc_top,'/home/josh/Documents/dsi/caps/cap3/Football_Supporter_Recommender/data/pickles/nfl_dists/cards_euclidean_update.pickle')
    team_cos_dist_top = team.top_dists(distance_calc='cosine_dist',index_=dom_df.squad.unique(),col='cos_dist',number=None)
    # pickler(card_cos_dist_top,'/home/josh/Documents/dsi/caps/cap3/Football_Supporter_Recommender/data/pickles/nfl_dists/cards_cos_dist_update.pickle')
    # team_cos_sim_top = team.top_dists(distance_calc='cosine_sim',index_=dom_df.squad.unique(),col='cos_sim',number=None)[::-1]
    # pickler(card_cos_sim_top,'/home/josh/Documents/dsi/caps/cap3/Football_Supporter_Recommender/data/pickles/nfl_dists/cards_cos_sim.pickle')
    # team_jac_top = team.top_dists(distance_calc='jaccard',index_=dom_df.squad.unique(),col='jac_dist',number=None)
    # pickler(team_jac_top,'/home/josh/Documents/dsi/caps/cap3/Football_Supporter_Recommender/data/pickles/nfl_dists/cards_jaccard.pickle')
    print(team_euc_top[:5])
    print(team_cos_dist_top[:5])
    team_comp_cont = [1 if x == y else 0 for x,y in zip(team_euc_top.index,team_cos_dist_top.index)] 
    team_comp_start = np.hstack((team_comp_start,team_comp_cont))
    # print(card_cos_sim_top)
    # print(card_jac_top)

    ## CHARGERS ##
    print('CHARGERS')
    team_vec = SimilarityDF(nfl_df).vectorizer(['Chargers'],'NFL')
    team = Distances(team_vec,epl_mat,w)
    team_euc_top = team.top_dists(distance_calc='euclidean',index_=dom_df.squad.unique(),col='euc_dist',number=None)
    # pickler(team_euc_top,'/home/josh/Documents/dsi/caps/cap3/Football_Supporter_Recommender/data/pickles/nfl_dists/cards_euclidean_update.pickle')
    team_cos_dist_top = team.top_dists(distance_calc='cosine_dist',index_=dom_df.squad.unique(),col='cos_dist',number=None)
    # pickler(card_cos_dist_top,'/home/josh/Documents/dsi/caps/cap3/Football_Supporter_Recommender/data/pickles/nfl_dists/cards_cos_dist_update.pickle')
    # team_cos_sim_top = team.top_dists(distance_calc='cosine_sim',index_=dom_df.squad.unique(),col='cos_sim',number=None)[::-1]
    # pickler(card_cos_sim_top,'/home/josh/Documents/dsi/caps/cap3/Football_Supporter_Recommender/data/pickles/nfl_dists/cards_cos_sim.pickle')
    # team_jac_top = team.top_dists(distance_calc='jaccard',index_=dom_df.squad.unique(),col='jac_dist',number=None)
    # pickler(team_jac_top,'/home/josh/Documents/dsi/caps/cap3/Football_Supporter_Recommender/data/pickles/nfl_dists/cards_jaccard.pickle')
    print(team_euc_top[:5])
    print(team_cos_dist_top[:5])
    team_comp_cont = [1 if x == y else 0 for x,y in zip(team_euc_top.index,team_cos_dist_top.index)] 
    team_comp_start = np.hstack((team_comp_start,team_comp_cont))
    # print(card_cos_sim_top)
    # print(card_jac_top)

    ## RAMS ##
    print('RAMS')
    team_vec = SimilarityDF(nfl_df).vectorizer(['Rams'],'NFL')
    team = Distances(team_vec,epl_mat,w)
    team_euc_top = team.top_dists(distance_calc='euclidean',index_=dom_df.squad.unique(),col='euc_dist',number=None)
    # pickler(team_euc_top,'/home/josh/Documents/dsi/caps/cap3/Football_Supporter_Recommender/data/pickles/nfl_dists/cards_euclidean_update.pickle')
    team_cos_dist_top = team.top_dists(distance_calc='cosine_dist',index_=dom_df.squad.unique(),col='cos_dist',number=None)
    # pickler(card_cos_dist_top,'/home/josh/Documents/dsi/caps/cap3/Football_Supporter_Recommender/data/pickles/nfl_dists/cards_cos_dist_update.pickle')
    # team_cos_sim_top = team.top_dists(distance_calc='cosine_sim',index_=dom_df.squad.unique(),col='cos_sim',number=None)[::-1]
    # pickler(card_cos_sim_top,'/home/josh/Documents/dsi/caps/cap3/Football_Supporter_Recommender/data/pickles/nfl_dists/cards_cos_sim.pickle')
    # team_jac_top = team.top_dists(distance_calc='jaccard',index_=dom_df.squad.unique(),col='jac_dist',number=None)
    # pickler(team_jac_top,'/home/josh/Documents/dsi/caps/cap3/Football_Supporter_Recommender/data/pickles/nfl_dists/cards_jaccard.pickle')
    print(team_euc_top[:5])
    print(team_cos_dist_top[:5])
    team_comp_cont = [1 if x == y else 0 for x,y in zip(team_euc_top.index,team_cos_dist_top.index)] 
    team_comp_start = np.hstack((team_comp_start,team_comp_cont))
    # print(card_cos_sim_top)
    # print(card_jac_top)

    ## DOLPHINS ##
    print('DOLPHINS')
    team_vec = SimilarityDF(nfl_df).vectorizer(['Dolphins'],'NFL')
    team = Distances(team_vec,epl_mat,w)
    team_euc_top = team.top_dists(distance_calc='euclidean',index_=dom_df.squad.unique(),col='euc_dist',number=None)
    # pickler(team_euc_top,'/home/josh/Documents/dsi/caps/cap3/Football_Supporter_Recommender/data/pickles/nfl_dists/cards_euclidean_update.pickle')
    team_cos_dist_top = team.top_dists(distance_calc='cosine_dist',index_=dom_df.squad.unique(),col='cos_dist',number=None)
    # pickler(card_cos_dist_top,'/home/josh/Documents/dsi/caps/cap3/Football_Supporter_Recommender/data/pickles/nfl_dists/cards_cos_dist_update.pickle')
    # team_cos_sim_top = team.top_dists(distance_calc='cosine_sim',index_=dom_df.squad.unique(),col='cos_sim',number=None)[::-1]
    # pickler(card_cos_sim_top,'/home/josh/Documents/dsi/caps/cap3/Football_Supporter_Recommender/data/pickles/nfl_dists/cards_cos_sim.pickle')
    # team_jac_top = team.top_dists(distance_calc='jaccard',index_=dom_df.squad.unique(),col='jac_dist',number=None)
    # pickler(team_jac_top,'/home/josh/Documents/dsi/caps/cap3/Football_Supporter_Recommender/data/pickles/nfl_dists/cards_jaccard.pickle')
    print(team_euc_top[:5])
    print(team_cos_dist_top[:5])
    team_comp_cont = [1 if x == y else 0 for x,y in zip(team_euc_top.index,team_cos_dist_top.index)] 
    team_comp_start = np.hstack((team_comp_start,team_comp_cont))
    # print(card_cos_sim_top)
    # print(card_jac_top)

    ## VIKINGS ##
    print('VIKINGS')
    team_vec = SimilarityDF(nfl_df).vectorizer(['Vikings'],'NFL')
    team = Distances(team_vec,epl_mat,w)
    team_euc_top = team.top_dists(distance_calc='euclidean',index_=dom_df.squad.unique(),col='euc_dist',number=None)
    # pickler(team_euc_top,'/home/josh/Documents/dsi/caps/cap3/Football_Supporter_Recommender/data/pickles/nfl_dists/cards_euclidean_update.pickle')
    team_cos_dist_top = team.top_dists(distance_calc='cosine_dist',index_=dom_df.squad.unique(),col='cos_dist',number=None)
    # pickler(card_cos_dist_top,'/home/josh/Documents/dsi/caps/cap3/Football_Supporter_Recommender/data/pickles/nfl_dists/cards_cos_dist_update.pickle')
    # team_cos_sim_top = team.top_dists(distance_calc='cosine_sim',index_=dom_df.squad.unique(),col='cos_sim',number=None)[::-1]
    # pickler(card_cos_sim_top,'/home/josh/Documents/dsi/caps/cap3/Football_Supporter_Recommender/data/pickles/nfl_dists/cards_cos_sim.pickle')
    # team_jac_top = team.top_dists(distance_calc='jaccard',index_=dom_df.squad.unique(),col='jac_dist',number=None)
    # pickler(team_jac_top,'/home/josh/Documents/dsi/caps/cap3/Football_Supporter_Recommender/data/pickles/nfl_dists/cards_jaccard.pickle')
    print(team_euc_top[:5])
    print(team_cos_dist_top[:5])
    team_comp_cont = [1 if x == y else 0 for x,y in zip(team_euc_top.index,team_cos_dist_top.index)] 
    team_comp_start = np.hstack((team_comp_start,team_comp_cont))
    # print(card_cos_sim_top)
    # print(card_jac_top)

    ## PATRIOTS ##
    print('PATRIOTS')
    team_vec = SimilarityDF(nfl_df).vectorizer(['Patriots'],'NFL')
    team = Distances(team_vec,epl_mat,w)
    team_euc_top = team.top_dists(distance_calc='euclidean',index_=dom_df.squad.unique(),col='euc_dist',number=None)
    # pickler(team_euc_top,'/home/josh/Documents/dsi/caps/cap3/Football_Supporter_Recommender/data/pickles/nfl_dists/cards_euclidean_update.pickle')
    team_cos_dist_top = team.top_dists(distance_calc='cosine_dist',index_=dom_df.squad.unique(),col='cos_dist',number=None)
    # pickler(card_cos_dist_top,'/home/josh/Documents/dsi/caps/cap3/Football_Supporter_Recommender/data/pickles/nfl_dists/cards_cos_dist_update.pickle')
    # team_cos_sim_top = team.top_dists(distance_calc='cosine_sim',index_=dom_df.squad.unique(),col='cos_sim',number=None)[::-1]
    # pickler(card_cos_sim_top,'/home/josh/Documents/dsi/caps/cap3/Football_Supporter_Recommender/data/pickles/nfl_dists/cards_cos_sim.pickle')
    # team_jac_top = team.top_dists(distance_calc='jaccard',index_=dom_df.squad.unique(),col='jac_dist',number=None)
    # pickler(team_jac_top,'/home/josh/Documents/dsi/caps/cap3/Football_Supporter_Recommender/data/pickles/nfl_dists/cards_jaccard.pickle')
    print(team_euc_top[:5])
    print(team_cos_dist_top[:5])
    team_comp_cont = [1 if x == y else 0 for x,y in zip(team_euc_top.index,team_cos_dist_top.index)] 
    team_comp_start = np.hstack((team_comp_start,team_comp_cont))
    # print(card_cos_sim_top)
    # print(card_jac_top)

    ## SAINTS ##
    print('SAINTS')
    team_vec = SimilarityDF(nfl_df).vectorizer(['Saints'],'NFL')
    team = Distances(team_vec,epl_mat,w)
    team_euc_top = team.top_dists(distance_calc='euclidean',index_=dom_df.squad.unique(),col='euc_dist',number=None)
    # pickler(team_euc_top,'/home/josh/Documents/dsi/caps/cap3/Football_Supporter_Recommender/data/pickles/nfl_dists/cards_euclidean_update.pickle')
    team_cos_dist_top = team.top_dists(distance_calc='cosine_dist',index_=dom_df.squad.unique(),col='cos_dist',number=None)
    # pickler(card_cos_dist_top,'/home/josh/Documents/dsi/caps/cap3/Football_Supporter_Recommender/data/pickles/nfl_dists/cards_cos_dist_update.pickle')
    # team_cos_sim_top = team.top_dists(distance_calc='cosine_sim',index_=dom_df.squad.unique(),col='cos_sim',number=None)[::-1]
    # pickler(card_cos_sim_top,'/home/josh/Documents/dsi/caps/cap3/Football_Supporter_Recommender/data/pickles/nfl_dists/cards_cos_sim.pickle')
    # team_jac_top = team.top_dists(distance_calc='jaccard',index_=dom_df.squad.unique(),col='jac_dist',number=None)
    # pickler(team_jac_top,'/home/josh/Documents/dsi/caps/cap3/Football_Supporter_Recommender/data/pickles/nfl_dists/cards_jaccard.pickle')
    print(team_euc_top[:5])
    print(team_cos_dist_top[:5])
    team_comp_cont = [1 if x == y else 0 for x,y in zip(team_euc_top.index,team_cos_dist_top.index)] 
    team_comp_start = np.hstack((team_comp_start,team_comp_cont))
    # print(card_cos_sim_top)
    # print(card_jac_top)

    ## GIANTS ##
    print('GIANTS')
    team_vec = SimilarityDF(nfl_df).vectorizer(['Giants'],'NFL')
    team = Distances(team_vec,epl_mat,w)
    team_euc_top = team.top_dists(distance_calc='euclidean',index_=dom_df.squad.unique(),col='euc_dist',number=None)
    # pickler(team_euc_top,'/home/josh/Documents/dsi/caps/cap3/Football_Supporter_Recommender/data/pickles/nfl_dists/cards_euclidean_update.pickle')
    team_cos_dist_top = team.top_dists(distance_calc='cosine_dist',index_=dom_df.squad.unique(),col='cos_dist',number=None)
    # pickler(card_cos_dist_top,'/home/josh/Documents/dsi/caps/cap3/Football_Supporter_Recommender/data/pickles/nfl_dists/cards_cos_dist_update.pickle')
    # team_cos_sim_top = team.top_dists(distance_calc='cosine_sim',index_=dom_df.squad.unique(),col='cos_sim',number=None)[::-1]
    # pickler(card_cos_sim_top,'/home/josh/Documents/dsi/caps/cap3/Football_Supporter_Recommender/data/pickles/nfl_dists/cards_cos_sim.pickle')
    # team_jac_top = team.top_dists(distance_calc='jaccard',index_=dom_df.squad.unique(),col='jac_dist',number=None)
    # pickler(team_jac_top,'/home/josh/Documents/dsi/caps/cap3/Football_Supporter_Recommender/data/pickles/nfl_dists/cards_jaccard.pickle')
    print(team_euc_top[:5])
    print(team_cos_dist_top[:5])
    team_comp_cont = [1 if x == y else 0 for x,y in zip(team_euc_top.index,team_cos_dist_top.index)] 
    team_comp_start = np.hstack((team_comp_start,team_comp_cont))
    # print(card_cos_sim_top)
    # print(card_jac_top)

    ## JETS ##
    print('JETS')
    team_vec = SimilarityDF(nfl_df).vectorizer(['Jets'],'NFL')
    team = Distances(team_vec,epl_mat,w)
    team_euc_top = team.top_dists(distance_calc='euclidean',index_=dom_df.squad.unique(),col='euc_dist',number=None)
    # pickler(team_euc_top,'/home/josh/Documents/dsi/caps/cap3/Football_Supporter_Recommender/data/pickles/nfl_dists/cards_euclidean_update.pickle')
    team_cos_dist_top = team.top_dists(distance_calc='cosine_dist',index_=dom_df.squad.unique(),col='cos_dist',number=None)
    # pickler(card_cos_dist_top,'/home/josh/Documents/dsi/caps/cap3/Football_Supporter_Recommender/data/pickles/nfl_dists/cards_cos_dist_update.pickle')
    # team_cos_sim_top = team.top_dists(distance_calc='cosine_sim',index_=dom_df.squad.unique(),col='cos_sim',number=None)[::-1]
    # pickler(card_cos_sim_top,'/home/josh/Documents/dsi/caps/cap3/Football_Supporter_Recommender/data/pickles/nfl_dists/cards_cos_sim.pickle')
    # team_jac_top = team.top_dists(distance_calc='jaccard',index_=dom_df.squad.unique(),col='jac_dist',number=None)
    # pickler(team_jac_top,'/home/josh/Documents/dsi/caps/cap3/Football_Supporter_Recommender/data/pickles/nfl_dists/cards_jaccard.pickle')
    print(team_euc_top[:5])
    print(team_cos_dist_top[:5])
    team_comp_cont = [1 if x == y else 0 for x,y in zip(team_euc_top.index,team_cos_dist_top.index)] 
    team_comp_start = np.hstack((team_comp_start,team_comp_cont))
    # print(card_cos_sim_top)
    # print(card_jac_top)

    ## RAIDERS ##
    print('RAIDERS')
    team_vec = SimilarityDF(nfl_df).vectorizer(['Raiders'],'NFL')
    team = Distances(team_vec,epl_mat,w)
    team_euc_top = team.top_dists(distance_calc='euclidean',index_=dom_df.squad.unique(),col='euc_dist',number=None)
    # pickler(team_euc_top,'/home/josh/Documents/dsi/caps/cap3/Football_Supporter_Recommender/data/pickles/nfl_dists/cards_euclidean_update.pickle')
    team_cos_dist_top = team.top_dists(distance_calc='cosine_dist',index_=dom_df.squad.unique(),col='cos_dist',number=None)
    # pickler(card_cos_dist_top,'/home/josh/Documents/dsi/caps/cap3/Football_Supporter_Recommender/data/pickles/nfl_dists/cards_cos_dist_update.pickle')
    # team_cos_sim_top = team.top_dists(distance_calc='cosine_sim',index_=dom_df.squad.unique(),col='cos_sim',number=None)[::-1]
    # pickler(card_cos_sim_top,'/home/josh/Documents/dsi/caps/cap3/Football_Supporter_Recommender/data/pickles/nfl_dists/cards_cos_sim.pickle')
    # team_jac_top = team.top_dists(distance_calc='jaccard',index_=dom_df.squad.unique(),col='jac_dist',number=None)
    # pickler(team_jac_top,'/home/josh/Documents/dsi/caps/cap3/Football_Supporter_Recommender/data/pickles/nfl_dists/cards_jaccard.pickle')
    print(team_euc_top[:5])
    print(team_cos_dist_top[:5])
    team_comp_cont = [1 if x == y else 0 for x,y in zip(team_euc_top.index,team_cos_dist_top.index)] 
    team_comp_start = np.hstack((team_comp_start,team_comp_cont))
    # print(card_cos_sim_top)
    # print(card_jac_top)

    ## EAGLES ##
    print('EAGLES')
    team_vec = SimilarityDF(nfl_df).vectorizer(['Eagles'],'NFL')
    team = Distances(team_vec,epl_mat,w)
    team_euc_top = team.top_dists(distance_calc='euclidean',index_=dom_df.squad.unique(),col='euc_dist',number=None)
    # pickler(team_euc_top,'/home/josh/Documents/dsi/caps/cap3/Football_Supporter_Recommender/data/pickles/nfl_dists/cards_euclidean_update.pickle')
    team_cos_dist_top = team.top_dists(distance_calc='cosine_dist',index_=dom_df.squad.unique(),col='cos_dist',number=None)
    # pickler(card_cos_dist_top,'/home/josh/Documents/dsi/caps/cap3/Football_Supporter_Recommender/data/pickles/nfl_dists/cards_cos_dist_update.pickle')
    # team_cos_sim_top = team.top_dists(distance_calc='cosine_sim',index_=dom_df.squad.unique(),col='cos_sim',number=None)[::-1]
    # pickler(card_cos_sim_top,'/home/josh/Documents/dsi/caps/cap3/Football_Supporter_Recommender/data/pickles/nfl_dists/cards_cos_sim.pickle')
    # team_jac_top = team.top_dists(distance_calc='jaccard',index_=dom_df.squad.unique(),col='jac_dist',number=None)
    # pickler(team_jac_top,'/home/josh/Documents/dsi/caps/cap3/Football_Supporter_Recommender/data/pickles/nfl_dists/cards_jaccard.pickle')
    print(team_euc_top[:5])
    print(team_cos_dist_top[:5])
    team_comp_cont = [1 if x == y else 0 for x,y in zip(team_euc_top.index,team_cos_dist_top.index)] 
    team_comp_start = np.hstack((team_comp_start,team_comp_cont))
    # print(card_cos_sim_top)
    # print(card_jac_top)

    ## STEELERS ##
    print('STEELERS')
    team_vec = SimilarityDF(nfl_df).vectorizer(['Steelers'],'NFL')
    team = Distances(team_vec,epl_mat,w)
    team_euc_top = team.top_dists(distance_calc='euclidean',index_=dom_df.squad.unique(),col='euc_dist',number=None)
    # pickler(team_euc_top,'/home/josh/Documents/dsi/caps/cap3/Football_Supporter_Recommender/data/pickles/nfl_dists/cards_euclidean_update.pickle')
    team_cos_dist_top = team.top_dists(distance_calc='cosine_dist',index_=dom_df.squad.unique(),col='cos_dist',number=None)
    # pickler(card_cos_dist_top,'/home/josh/Documents/dsi/caps/cap3/Football_Supporter_Recommender/data/pickles/nfl_dists/cards_cos_dist_update.pickle')
    # team_cos_sim_top = team.top_dists(distance_calc='cosine_sim',index_=dom_df.squad.unique(),col='cos_sim',number=None)[::-1]
    # pickler(card_cos_sim_top,'/home/josh/Documents/dsi/caps/cap3/Football_Supporter_Recommender/data/pickles/nfl_dists/cards_cos_sim.pickle')
    # team_jac_top = team.top_dists(distance_calc='jaccard',index_=dom_df.squad.unique(),col='jac_dist',number=None)
    # pickler(team_jac_top,'/home/josh/Documents/dsi/caps/cap3/Football_Supporter_Recommender/data/pickles/nfl_dists/cards_jaccard.pickle')
    print(team_euc_top[:5])
    print(team_cos_dist_top[:5])
    team_comp_cont = [1 if x == y else 0 for x,y in zip(team_euc_top.index,team_cos_dist_top.index)] 
    team_comp_start = np.hstack((team_comp_start,team_comp_cont))
    # print(card_cos_sim_top)
    # print(card_jac_top)

    ## 49ERS ##
    print('49ERS')
    team_vec = SimilarityDF(nfl_df).vectorizer(['49ers'],'NFL')
    team = Distances(team_vec,epl_mat,w)
    team_euc_top = team.top_dists(distance_calc='euclidean',index_=dom_df.squad.unique(),col='euc_dist',number=None)
    # pickler(team_euc_top,'/home/josh/Documents/dsi/caps/cap3/Football_Supporter_Recommender/data/pickles/nfl_dists/cards_euclidean_update.pickle')
    team_cos_dist_top = team.top_dists(distance_calc='cosine_dist',index_=dom_df.squad.unique(),col='cos_dist',number=None)
    # pickler(card_cos_dist_top,'/home/josh/Documents/dsi/caps/cap3/Football_Supporter_Recommender/data/pickles/nfl_dists/cards_cos_dist_update.pickle')
    # team_cos_sim_top = team.top_dists(distance_calc='cosine_sim',index_=dom_df.squad.unique(),col='cos_sim',number=None)[::-1]
    # pickler(card_cos_sim_top,'/home/josh/Documents/dsi/caps/cap3/Football_Supporter_Recommender/data/pickles/nfl_dists/cards_cos_sim.pickle')
    # team_jac_top = team.top_dists(distance_calc='jaccard',index_=dom_df.squad.unique(),col='jac_dist',number=None)
    # pickler(team_jac_top,'/home/josh/Documents/dsi/caps/cap3/Football_Supporter_Recommender/data/pickles/nfl_dists/cards_jaccard.pickle')
    print(team_euc_top[:5])
    print(team_cos_dist_top[:5])
    team_comp_cont = [1 if x == y else 0 for x,y in zip(team_euc_top.index,team_cos_dist_top.index)] 
    team_comp_start = np.hstack((team_comp_start,team_comp_cont))
    # print(card_cos_sim_top)
    # print(card_jac_top)

    ## SEAHAWKS ##
    print('SEAHAWKS')
    team_vec = SimilarityDF(nfl_df).vectorizer(['Seahawks'],'NFL')
    team = Distances(team_vec,epl_mat,w)
    team_euc_top = team.top_dists(distance_calc='euclidean',index_=dom_df.squad.unique(),col='euc_dist',number=None)
    # pickler(team_euc_top,'/home/josh/Documents/dsi/caps/cap3/Football_Supporter_Recommender/data/pickles/nfl_dists/cards_euclidean_update.pickle')
    team_cos_dist_top = team.top_dists(distance_calc='cosine_dist',index_=dom_df.squad.unique(),col='cos_dist',number=None)
    # pickler(card_cos_dist_top,'/home/josh/Documents/dsi/caps/cap3/Football_Supporter_Recommender/data/pickles/nfl_dists/cards_cos_dist_update.pickle')
    # team_cos_sim_top = team.top_dists(distance_calc='cosine_sim',index_=dom_df.squad.unique(),col='cos_sim',number=None)[::-1]
    # pickler(card_cos_sim_top,'/home/josh/Documents/dsi/caps/cap3/Football_Supporter_Recommender/data/pickles/nfl_dists/cards_cos_sim.pickle')
    # team_jac_top = team.top_dists(distance_calc='jaccard',index_=dom_df.squad.unique(),col='jac_dist',number=None)
    # pickler(team_jac_top,'/home/josh/Documents/dsi/caps/cap3/Football_Supporter_Recommender/data/pickles/nfl_dists/cards_jaccard.pickle')
    print(team_euc_top[:5])
    print(team_cos_dist_top[:5])
    team_comp_cont = [1 if x == y else 0 for x,y in zip(team_euc_top.index,team_cos_dist_top.index)] 
    team_comp_start = np.hstack((team_comp_start,team_comp_cont))
    # print(card_cos_sim_top)
    # print(card_jac_top)

    ## BUCCANEERS ##
    print('BUCCANEERS')
    team_vec = SimilarityDF(nfl_df).vectorizer(['Buccaneers'],'NFL')
    team = Distances(team_vec,epl_mat,w)
    team_euc_top = team.top_dists(distance_calc='euclidean',index_=dom_df.squad.unique(),col='euc_dist',number=None)
    # pickler(team_euc_top,'/home/josh/Documents/dsi/caps/cap3/Football_Supporter_Recommender/data/pickles/nfl_dists/cards_euclidean_update.pickle')
    team_cos_dist_top = team.top_dists(distance_calc='cosine_dist',index_=dom_df.squad.unique(),col='cos_dist',number=None)
    # pickler(card_cos_dist_top,'/home/josh/Documents/dsi/caps/cap3/Football_Supporter_Recommender/data/pickles/nfl_dists/cards_cos_dist_update.pickle')
    # team_cos_sim_top = team.top_dists(distance_calc='cosine_sim',index_=dom_df.squad.unique(),col='cos_sim',number=None)[::-1]
    # pickler(card_cos_sim_top,'/home/josh/Documents/dsi/caps/cap3/Football_Supporter_Recommender/data/pickles/nfl_dists/cards_cos_sim.pickle')
    # team_jac_top = team.top_dists(distance_calc='jaccard',index_=dom_df.squad.unique(),col='jac_dist',number=None)
    # pickler(team_jac_top,'/home/josh/Documents/dsi/caps/cap3/Football_Supporter_Recommender/data/pickles/nfl_dists/cards_jaccard.pickle')
    print(team_euc_top[:5])
    print(team_cos_dist_top[:5])
    team_comp_cont = [1 if x == y else 0 for x,y in zip(team_euc_top.index,team_cos_dist_top.index)] 
    team_comp_start = np.hstack((team_comp_start,team_comp_cont))
    # print(card_cos_sim_top)
    # print(card_jac_top)

    ## TITANS ##
    print('TITANS')
    team_vec = SimilarityDF(nfl_df).vectorizer(['Titans'],'NFL')
    team = Distances(team_vec,epl_mat,w)
    team_euc_top = team.top_dists(distance_calc='euclidean',index_=dom_df.squad.unique(),col='euc_dist',number=None)
    # pickler(team_euc_top,'/home/josh/Documents/dsi/caps/cap3/Football_Supporter_Recommender/data/pickles/nfl_dists/cards_euclidean_update.pickle')
    team_cos_dist_top = team.top_dists(distance_calc='cosine_dist',index_=dom_df.squad.unique(),col='cos_dist',number=None)
    # pickler(card_cos_dist_top,'/home/josh/Documents/dsi/caps/cap3/Football_Supporter_Recommender/data/pickles/nfl_dists/cards_cos_dist_update.pickle')
    # team_cos_sim_top = team.top_dists(distance_calc='cosine_sim',index_=dom_df.squad.unique(),col='cos_sim',number=None)[::-1]
    # pickler(card_cos_sim_top,'/home/josh/Documents/dsi/caps/cap3/Football_Supporter_Recommender/data/pickles/nfl_dists/cards_cos_sim.pickle')
    # team_jac_top = team.top_dists(distance_calc='jaccard',index_=dom_df.squad.unique(),col='jac_dist',number=None)
    # pickler(team_jac_top,'/home/josh/Documents/dsi/caps/cap3/Football_Supporter_Recommender/data/pickles/nfl_dists/cards_jaccard.pickle')
    print(team_euc_top[:5])
    print(team_cos_dist_top[:5])
    team_comp_cont = [1 if x == y else 0 for x,y in zip(team_euc_top.index,team_cos_dist_top.index)] 
    team_comp_start = np.hstack((team_comp_start,team_comp_cont))
    # print(card_cos_sim_top)
    # print(card_jac_top)

    ## REDSKINS ##
    print('REDSKINS')
    team_vec = SimilarityDF(nfl_df).vectorizer(['Redskins'],'NFL')
    team = Distances(team_vec,epl_mat,w)
    team_euc_top = team.top_dists(distance_calc='euclidean',index_=dom_df.squad.unique(),col='euc_dist',number=None)
    # pickler(team_euc_top,'/home/josh/Documents/dsi/caps/cap3/Football_Supporter_Recommender/data/pickles/nfl_dists/cards_euclidean_update.pickle')
    team_cos_dist_top = team.top_dists(distance_calc='cosine_dist',index_=dom_df.squad.unique(),col='cos_dist',number=None)
    # pickler(card_cos_dist_top,'/home/josh/Documents/dsi/caps/cap3/Football_Supporter_Recommender/data/pickles/nfl_dists/cards_cos_dist_update.pickle')
    # team_cos_sim_top = team.top_dists(distance_calc='cosine_sim',index_=dom_df.squad.unique(),col='cos_sim',number=None)[::-1]
    # pickler(card_cos_sim_top,'/home/josh/Documents/dsi/caps/cap3/Football_Supporter_Recommender/data/pickles/nfl_dists/cards_cos_sim.pickle')
    # team_jac_top = team.top_dists(distance_calc='jaccard',index_=dom_df.squad.unique(),col='jac_dist',number=None)
    # pickler(team_jac_top,'/home/josh/Documents/dsi/caps/cap3/Football_Supporter_Recommender/data/pickles/nfl_dists/cards_jaccard.pickle')
    print(team_euc_top[:5])
    print(team_cos_dist_top[:5])
    team_comp_cont = [1 if x == y else 0 for x,y in zip(team_euc_top.index,team_cos_dist_top.index)] 
    team_comp_start = np.hstack((team_comp_start,team_comp_cont))
    # print(card_cos_sim_top)
    # print(card_jac_top)

    pickler(team_comp_start,'/home/josh/Documents/dsi/caps/cap3/Football_Supporter_Recommender/data/pickles/nfl_dists/team_comp.pickle')
    # broncs_cos = unpickler('/home/josh/Documents/dsi/caps/cap3/Football_Supporter_Recommender/data/pickles/nfl_dists/broncs_cos_dist.pickle')
    # broncs_euc = unpickler('/home/josh/Documents/dsi/caps/cap3/Football_Supporter_Recommender/data/pickles/nfl_dists/broncs_euclidean.pickle')

    # nines_cos = unpickler('/home/josh/Documents/dsi/caps/cap3/Football_Supporter_Recommender/data/pickles/nfl_dists/nines_cos_dist.pickle')
    # nines_cos = unpickler('/home/josh/Documents/dsi/caps/cap3/Football_Supporter_Recommender/data/pickles/nfl_dists/nines_euclidean.pickle')

    # tex_cos = unpickler('/home/josh/Documents/dsi/caps/cap3/Football_Supporter_Recommender/data/pickles/nfl_dists/texs_cos_dist.pickle')
    # tex_cos = unpickler('/home/josh/Documents/dsi/caps/cap3/Football_Supporter_Recommender/data/pickles/nfl_dists/texs_euclidean.pickle')
    
    # print(broncs_cos)
    # print(broncs_euc)