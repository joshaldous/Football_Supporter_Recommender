import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
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
    def __init__(self,team_vector,league_matrix):
        self.team_vector = team_vector
        self.league_matrix = league_matrix
               
    def euclidean_dist_calc(self,weights=None):
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
    dom_df = unpickler('/home/josh/Documents/dsi/caps/cap3/Football_Supporter_Recommender/data/pickles/epl_domestic_league_df_clean.pickle')
    # icup_df = unpickler('/home/josh/Documents/dsi/caps/cap3/Football_Supporter_Recommender/data/pickles/epl_intnl_cup_df_clean.pickle')
    # dcup_df = unpickler('/home/josh/Documents/dsi/caps/cap3/Football_Supporter_Recommender/data/pickles/epl_domestic_cup_df_clean.pickle')
    nfl_df = unpickler('/home/josh/Documents/dsi/caps/cap3/Football_Supporter_Recommender/data/pickles/NFL_df_clean.pickle')
    # print(dcup_df.info())
    # icup_df.set_index(['squad','season'],inplace=True)
    # print(icup_df.head())
    # efl_cup_df = dcup_df[(dcup_df.comp == 'EFL Cup') | (dcup_df.comp == 'League Cup')]
    # fa_cup_df = dcup_df[dcup_df.comp == 'FA Cup']
    
    # print(efl_cup_df.shape)
    # print(fa_cup_df.shape)
    
    dom_df.drop(['country','mp','w','d','l','pts','attendance'],axis=1,inplace=True)
    nfl_df.drop(['wins','losses','ties','off_rank_yards','def_rank_yards','turnover_ratio_rank',
                 'yard_diff_rank','margin_of_vict','strgth_sched','srs','off_srs','def_srs'],axis=1,inplace=True)

    # nfl_cols = nfl_df.columns.tolist()
    nfl_cols = ['team','year','points_for','points_against','point_difference','division_finish','playoffs',
                'playoff_finish','win%','loss%','tie%','off_rank_pts','def_rank_pts','pt_diff_rank']
    nfl_df = nfl_df.reindex(columns = nfl_cols)

    # dom_df.set_index(['squad','season'],inplace=True)
    # dom_df_std = SScale().fit_transform(dom_df.set_index(['squad','season']))
    
    # nfl_df.set_index(['team','year'],inplace=True)
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

    # epl_vec = SimilarityDF(dom_df).vectorizer(dom_df.squad.unique(),'EPL')
    # print(epl_vec.shape)
    # epl_vec = epl_vec.vectorizer(dom_df,dom_df.squad.unique(),'EPL')
    # epl_vec = epl_vec.
    # pickler(epl_vec,'/home/josh/Documents/dsi/caps/cap3/Football_Supporter_Recommender/data/pickles/epl_dists/epl_vectorized.pickle')
    # print(epl_vec.shape)

    # cardinals = nfl_df.set_index(['team','year']).loc['Cardinals']
    # cardinals_std = SScale().fit_transform(cardinals).ravel()
    # cardinals_std = cardinals_std.reshape(1,-1)
    # np.array().ravel()
    # print(cardinals_std.shape)
    epl_mat = unpickler('/home/josh/Documents/dsi/caps/cap3/Football_Supporter_Recommender/data/pickles/epl_dists/epl_vectorized.pickle')
    
    
    ## CARDINALS ##
    # cardinals_vec = SimilarityDF(nfl_df).vectorizer(['Cardinals'],'NFL')
    # cardinals = Distances(cardinals_vec,epl_mat)
    # card_euc_top = cardinals.top_dists(distance_calc='euclidean',index_=dom_df.squad.unique(),col='euc_dist',number=None)
    # pickler(card_euc_top,'/home/josh/Documents/dsi/caps/cap3/Football_Supporter_Recommender/data/pickles/nfl_dists/cards_euclidean.pickle')
    # card_cos_dist_top = cardinals.top_dists(distance_calc='cosine_dist',index_=dom_df.squad.unique(),col='cos_dist',number=None)
    # pickler(card_cos_dist_top,'/home/josh/Documents/dsi/caps/cap3/Football_Supporter_Recommender/data/pickles/nfl_dists/cards_cos_dist.pickle')
    # card_cos_sim_top = cardinals.top_dists(distance_calc='cosine_sim',index_=dom_df.squad.unique(),col='cos_sim',number=None)[::-1]
    # pickler(card_cos_sim_top,'/home/josh/Documents/dsi/caps/cap3/Football_Supporter_Recommender/data/pickles/nfl_dists/cards_cos_sim.pickle')
    # card_jac_top = cardinals.top_dists(distance_calc='jaccard',index_=dom_df.squad.unique(),col='jac_dist',number=None)
    # pickler(card_jac_top,'/home/josh/Documents/dsi/caps/cap3/Football_Supporter_Recommender/data/pickles/nfl_dists/cards_jaccard.pickle')
    # print(card_euc_top)
    # print(card_cos_dist_top)
    # print(card_cos_sim_top)
    # print(card_jac_top)

    ## FALCONS ##
    # falcons_vec = SimilarityDF(nfl_df).vectorizer(['Falcons'],'NFL')
    # falcons = Distances(falcons_vec,epl_mat) 
    # falc_euc_top = falcons.top_dists(distance_calc='euclidean',index_=dom_df.squad.unique(),col='euc_dist',number=None)
    # pickler(falc_euc_top,'/home/josh/Documents/dsi/caps/cap3/Football_Supporter_Recommender/data/pickles/nfl_dists/falcs_euclidean.pickle')
    # falc_cos_dist_top = falcons.top_dists(distance_calc='cosine_dist',index_=dom_df.squad.unique(),col='cos_dist',number=None)
    # pickler(falc_cos_dist_top,'/home/josh/Documents/dsi/caps/cap3/Football_Supporter_Recommender/data/pickles/nfl_dists/falcs_cos_dist.pickle')
    # falc_cos_sim_top = falcons.top_dists(distance_calc='cosine_sim',index_=dom_df.squad.unique(),col='cos_sim',number=None)[::-1]
    # pickler(falc_cos_sim_top,'/home/josh/Documents/dsi/caps/cap3/Football_Supporter_Recommender/data/pickles/nfl_dists/falcs_cos_sim.pickle')
    # falc_jac_top = falcons.top_dists(distance_calc='jaccard',index_=dom_df.squad.unique(),col='jac_dist',number=None)
    # pickler(falc_jac_top,'/home/josh/Documents/dsi/caps/cap3/Football_Supporter_Recommender/data/pickles/nfl_dists/falcs_jaccard.pickle')
    # print(falc_euc_top)
    # print(falc_cos_dist_top)
    # print(falc_cos_sim_top)
    # print(falc_jac_top)

    ## RAVENS ##
    # ravens_vec = SimilarityDF(nfl_df).vectorizer(['Ravens'],'NFL')
    # ravens = Distances(ravens_vec,epl_mat) 
    # rav_euc_top = ravens.top_dists(distance_calc='euclidean',index_=dom_df.squad.unique(),col='euc_dist',number=None)
    # pickler(rav_euc_top,'/home/josh/Documents/dsi/caps/cap3/Football_Supporter_Recommender/data/pickles/nfl_dists/ravs_euclidean.pickle')
    # rav_cos_dist_top = ravens.top_dists(distance_calc='cosine_dist',index_=dom_df.squad.unique(),col='cos_dist',number=None)
    # pickler(rav_cos_dist_top,'/home/josh/Documents/dsi/caps/cap3/Football_Supporter_Recommender/data/pickles/nfl_dists/ravs_cos_dist.pickle')
    # rav_cos_sim_top = ravens.top_dists(distance_calc='cosine_sim',index_=dom_df.squad.unique(),col='cos_sim',number=None)[::-1]
    # pickler(rav_cos_sim_top,'/home/josh/Documents/dsi/caps/cap3/Football_Supporter_Recommender/data/pickles/nfl_dists/ravs_cos_sim.pickle')
    # rav_jac_top = ravens.top_dists(distance_calc='jaccard',index_=dom_df.squad.unique(),col='jac_dist',number=None)
    # pickler(rav_jac_top,'/home/josh/Documents/dsi/caps/cap3/Football_Supporter_Recommender/data/pickles/nfl_dists/ravs_jaccard.pickle')
    # print(rav_euc_top)
    # print(rav_cos_dist_top)
    # print(rav_cos_sim_top)
    # print(rav_jac_top)

    ## BILLS ##
    # bills_vec = SimilarityDF(nfl_df).vectorizer(['Bills'],'NFL')
    # bills = Distances(bills_vec,epl_mat) 
    # bill_euc_top = bills.top_dists(distance_calc='euclidean',index_=dom_df.squad.unique(),col='euc_dist',number=None)
    # pickler(bill_euc_top,'/home/josh/Documents/dsi/caps/cap3/Football_Supporter_Recommender/data/pickles/nfl_dists/bills_euclidean.pickle')
    # bill_cos_dist_top = bills.top_dists(distance_calc='cosine_dist',index_=dom_df.squad.unique(),col='cos_dist',number=None)
    # pickler(bill_cos_dist_top,'/home/josh/Documents/dsi/caps/cap3/Football_Supporter_Recommender/data/pickles/nfl_dists/bills_cos_dist.pickle')
    # bill_cos_sim_top = bills.top_dists(distance_calc='cosine_sim',index_=dom_df.squad.unique(),col='cos_sim',number=None)[::-1]
    # pickler(bill_cos_sim_top,'/home/josh/Documents/dsi/caps/cap3/Football_Supporter_Recommender/data/pickles/nfl_dists/bills_cos_sim.pickle')
    # bill_jac_top = bills.top_dists(distance_calc='jaccard',index_=dom_df.squad.unique(),col='jac_dist',number=None)
    # pickler(bill_jac_top,'/home/josh/Documents/dsi/caps/cap3/Football_Supporter_Recommender/data/pickles/nfl_dists/bills_jaccard.pickle')
    # print(bill_euc_top)
    # print(bill_cos_dist_top)
    # print(bill_cos_sim_top)
    # print(bill_jac_top)

    ## PANTHERS ##
    # panthers_vec = SimilarityDF(nfl_df).vectorizer(['Panthers'],'NFL')
    # panthers = Distances(panthers_vec,epl_mat) 
    # panth_euc_top = panthers.top_dists(distance_calc='euclidean',index_=dom_df.squad.unique(),col='euc_dist',number=None)
    # pickler(panth_euc_top,'/home/josh/Documents/dsi/caps/cap3/Football_Supporter_Recommender/data/pickles/nfl_dists/panths_euclidean.pickle')
    # panth_cos_dist_top = panthers.top_dists(distance_calc='cosine_dist',index_=dom_df.squad.unique(),col='cos_dist',number=None)
    # pickler(panth_cos_dist_top,'/home/josh/Documents/dsi/caps/cap3/Football_Supporter_Recommender/data/pickles/nfl_dists/panths_cos_dist.pickle')
    # panth_cos_sim_top = panthers.top_dists(distance_calc='cosine_sim',index_=dom_df.squad.unique(),col='cos_sim',number=None)[::-1]
    # pickler(panth_cos_sim_top,'/home/josh/Documents/dsi/caps/cap3/Football_Supporter_Recommender/data/pickles/nfl_dists/panths_cos_sim.pickle')
    # panth_jac_top = panthers.top_dists(distance_calc='jaccard',index_=dom_df.squad.unique(),col='jac_dist',number=None)
    # pickler(panth_jac_top,'/home/josh/Documents/dsi/caps/cap3/Football_Supporter_Recommender/data/pickles/nfl_dists/panths_jaccard.pickle')
    # print(panth_euc_top)
    # print(panth_cos_dist_top)
    # print(panth_cos_sim_top)
    # print(panth_jac_top)

    ## BEARS ##
    # print('BEARS')
    # bears_vec = SimilarityDF(nfl_df).vectorizer(['Bears'],'NFL')
    # bears = Distances(bears_vec,epl_mat) 
    # bear_euc_top = bears.top_dists(distance_calc='euclidean',index_=dom_df.squad.unique(),col='euc_dist',number=None)
    # pickler(bear_euc_top,'/home/josh/Documents/dsi/caps/cap3/Football_Supporter_Recommender/data/pickles/nfl_dists/bears_euclidean.pickle')
    # bear_cos_dist_top = bears.top_dists(distance_calc='cosine_dist',index_=dom_df.squad.unique(),col='cos_dist',number=None)
    # pickler(bear_cos_dist_top,'/home/josh/Documents/dsi/caps/cap3/Football_Supporter_Recommender/data/pickles/nfl_dists/bears_cos_dist.pickle')
    # bear_cos_sim_top = bears.top_dists(distance_calc='cosine_sim',index_=dom_df.squad.unique(),col='cos_sim',number=None)[::-1]
    # pickler(bear_cos_sim_top,'/home/josh/Documents/dsi/caps/cap3/Football_Supporter_Recommender/data/pickles/nfl_dists/bears_cos_sim.pickle')
    # bear_jac_top = bears.top_dists(distance_calc='jaccard',index_=dom_df.squad.unique(),col='jac_dist',number=None)
    # pickler(bear_jac_top,'/home/josh/Documents/dsi/caps/cap3/Football_Supporter_Recommender/data/pickles/nfl_dists/bears_jaccard.pickle')
    # print(bear_euc_top)
    # print(bear_cos_dist_top)
    # print(bear_cos_sim_top)
    # print(bear_jac_top)

    ## BENGALS ##
    # print('BENGALS')
    # bengals_vec = SimilarityDF(nfl_df).vectorizer(['Bengals'],'NFL')
    # bengals = Distances(bengals_vec,epl_mat) 
    # beng_euc_top = bengals.top_dists(distance_calc='euclidean',index_=dom_df.squad.unique(),col='euc_dist',number=None)
    # pickler(beng_euc_top,'/home/josh/Documents/dsi/caps/cap3/Football_Supporter_Recommender/data/pickles/nfl_dists/bengs_euclidean.pickle')
    # beng_cos_dist_top = bengals.top_dists(distance_calc='cosine_dist',index_=dom_df.squad.unique(),col='cos_dist',number=None)
    # pickler(beng_cos_dist_top,'/home/josh/Documents/dsi/caps/cap3/Football_Supporter_Recommender/data/pickles/nfl_dists/bengs_cos_dist.pickle')
    # beng_cos_sim_top = bengals.top_dists(distance_calc='cosine_sim',index_=dom_df.squad.unique(),col='cos_sim',number=None)[::-1]
    # pickler(beng_cos_sim_top,'/home/josh/Documents/dsi/caps/cap3/Football_Supporter_Recommender/data/pickles/nfl_dists/bengs_cos_sim.pickle')
    # beng_jac_top = bengals.top_dists(distance_calc='jaccard',index_=dom_df.squad.unique(),col='jac_dist',number=None)
    # pickler(beng_jac_top,'/home/josh/Documents/dsi/caps/cap3/Football_Supporter_Recommender/data/pickles/nfl_dists/bengs_jaccard.pickle')
    # print(beng_euc_top)
    # print(beng_cos_dist_top)
    # print(beng_cos_sim_top)
    # print(beng_jac_top)

    ## BROWNS ##
    # print('BROWNS')
    # browns_vec = SimilarityDF(nfl_df).vectorizer(['Browns'],'NFL')
    # browns = Distances(browns_vec,epl_mat) 
    # brown_euc_top = browns.top_dists(distance_calc='euclidean',index_=dom_df.squad.unique(),col='euc_dist',number=None)
    # pickler(brown_euc_top,'/home/josh/Documents/dsi/caps/cap3/Football_Supporter_Recommender/data/pickles/nfl_dists/browns_euclidean.pickle')
    # brown_cos_dist_top = browns.top_dists(distance_calc='cosine_dist',index_=dom_df.squad.unique(),col='cos_dist',number=None)
    # pickler(brown_cos_dist_top,'/home/josh/Documents/dsi/caps/cap3/Football_Supporter_Recommender/data/pickles/nfl_dists/browns_cos_dist.pickle')
    # brown_cos_sim_top = browns.top_dists(distance_calc='cosine_sim',index_=dom_df.squad.unique(),col='cos_sim',number=None)[::-1]
    # pickler(brown_cos_sim_top,'/home/josh/Documents/dsi/caps/cap3/Football_Supporter_Recommender/data/pickles/nfl_dists/browns_cos_sim.pickle')
    # brown_jac_top = browns.top_dists(distance_calc='jaccard',index_=dom_df.squad.unique(),col='jac_dist',number=None)
    # pickler(brown_jac_top,'/home/josh/Documents/dsi/caps/cap3/Football_Supporter_Recommender/data/pickles/nfl_dists/browns_jaccard.pickle')
    # print(brown_euc_top)
    # print(brown_cos_dist_top)
    # print(brown_cos_sim_top)
    # print(brown_jac_top)

    ## COWBOYS ##
    # print('COWBOYS')
    # cowboys_vec = SimilarityDF(nfl_df).vectorizer(['Cowboys'],'NFL')
    # cowboys = Distances(cowboys_vec,epl_mat) 
    # cow_euc_top = cowboys.top_dists(distance_calc='euclidean',index_=dom_df.squad.unique(),col='euc_dist',number=None)
    # pickler(cow_euc_top,'/home/josh/Documents/dsi/caps/cap3/Football_Supporter_Recommender/data/pickles/nfl_dists/cows_euclidean.pickle')
    # cow_cos_dist_top = cowboys.top_dists(distance_calc='cosine_dist',index_=dom_df.squad.unique(),col='cos_dist',number=None)
    # pickler(cow_cos_dist_top,'/home/josh/Documents/dsi/caps/cap3/Football_Supporter_Recommender/data/pickles/nfl_dists/cows_cos_dist.pickle')
    # cow_cos_sim_top = cowboys.top_dists(distance_calc='cosine_sim',index_=dom_df.squad.unique(),col='cos_sim',number=None)[::-1]
    # pickler(cow_cos_sim_top,'/home/josh/Documents/dsi/caps/cap3/Football_Supporter_Recommender/data/pickles/nfl_dists/cows_cos_sim.pickle')
    # cow_jac_top = cowboys.top_dists(distance_calc='jaccard',index_=dom_df.squad.unique(),col='jac_dist',number=None)
    # pickler(cow_jac_top,'/home/josh/Documents/dsi/caps/cap3/Football_Supporter_Recommender/data/pickles/nfl_dists/cows_jaccard.pickle')
    # print(cow_euc_top)
    # print(cow_cos_dist_top)
    # print(cow_cos_sim_top)
    # print(cow_jac_top)

    ## BRONCOS ##
    # print('BRONCOS')
    # broncos_vec = SimilarityDF(nfl_df).vectorizer(['Broncos'],'NFL')
    # broncos = Distances(broncos_vec,epl_mat) 
    # bronc_euc_top = broncos.top_dists(distance_calc='euclidean',index_=dom_df.squad.unique(),col='euc_dist',number=None)
    # pickler(bronc_euc_top,'/home/josh/Documents/dsi/caps/cap3/Football_Supporter_Recommender/data/pickles/nfl_dists/broncs_euclidean.pickle')
    # bronc_cos_dist_top = broncos.top_dists(distance_calc='cosine_dist',index_=dom_df.squad.unique(),col='cos_dist',number=None)
    # pickler(bronc_cos_dist_top,'/home/josh/Documents/dsi/caps/cap3/Football_Supporter_Recommender/data/pickles/nfl_dists/broncs_cos_dist.pickle')
    # bronc_cos_sim_top = broncos.top_dists(distance_calc='cosine_sim',index_=dom_df.squad.unique(),col='cos_sim',number=None)[::-1]
    # pickler(bronc_cos_sim_top,'/home/josh/Documents/dsi/caps/cap3/Football_Supporter_Recommender/data/pickles/nfl_dists/broncs_cos_sim.pickle')
    # bronc_jac_top = broncos.top_dists(distance_calc='jaccard',index_=dom_df.squad.unique(),col='jac_dist',number=None)
    # pickler(bronc_jac_top,'/home/josh/Documents/dsi/caps/cap3/Football_Supporter_Recommender/data/pickles/nfl_dists/broncs_jaccard.pickle')
    # print(bronc_euc_top)
    # print(bronc_cos_dist_top)
    # print(bronc_cos_sim_top)
    # print(bronc_jac_top)

    ## LIONS ##
    # print('LIONS')
    # lions_vec = SimilarityDF(nfl_df).vectorizer(['Lions'],'NFL')
    # lions = Distances(lions_vec,epl_mat) 
    # lion_euc_top = lions.top_dists(distance_calc='euclidean',index_=dom_df.squad.unique(),col='euc_dist',number=None)
    # pickler(lion_euc_top,'/home/josh/Documents/dsi/caps/cap3/Football_Supporter_Recommender/data/pickles/nfl_dists/lions_euclidean.pickle')
    # lion_cos_dist_top = lions.top_dists(distance_calc='cosine_dist',index_=dom_df.squad.unique(),col='cos_dist',number=None)
    # pickler(lion_cos_dist_top,'/home/josh/Documents/dsi/caps/cap3/Football_Supporter_Recommender/data/pickles/nfl_dists/lions_cos_dist.pickle')
    # lion_cos_sim_top = lions.top_dists(distance_calc='cosine_sim',index_=dom_df.squad.unique(),col='cos_sim',number=None)[::-1]
    # pickler(lion_cos_sim_top,'/home/josh/Documents/dsi/caps/cap3/Football_Supporter_Recommender/data/pickles/nfl_dists/lions_cos_sim.pickle')
    # lion_jac_top = lions.top_dists(distance_calc='jaccard',index_=dom_df.squad.unique(),col='jac_dist',number=None)
    # pickler(lion_jac_top,'/home/josh/Documents/dsi/caps/cap3/Football_Supporter_Recommender/data/pickles/nfl_dists/lions_jaccard.pickle')
    # print(lion_euc_top)
    # print(lion_cos_dist_top)
    # print(lion_cos_sim_top)
    # print(lion_jac_top)

    ## PACKERS ##
    # print('PACKERS')
    # packers_vec = SimilarityDF(nfl_df).vectorizer(['Packers'],'NFL')
    # packers = Distances(packers_vec,epl_mat) 
    # pack_euc_top = packers.top_dists(distance_calc='euclidean',index_=dom_df.squad.unique(),col='euc_dist',number=None)
    # pickler(pack_euc_top,'/home/josh/Documents/dsi/caps/cap3/Football_Supporter_Recommender/data/pickles/nfl_dists/packs_euclidean.pickle')
    # pack_cos_dist_top = packers.top_dists(distance_calc='cosine_dist',index_=dom_df.squad.unique(),col='cos_dist',number=None)
    # pickler(pack_cos_dist_top,'/home/josh/Documents/dsi/caps/cap3/Football_Supporter_Recommender/data/pickles/nfl_dists/packs_cos_dist.pickle')
    # pack_cos_sim_top = packers.top_dists(distance_calc='cosine_sim',index_=dom_df.squad.unique(),col='cos_sim',number=None)[::-1]
    # pickler(pack_cos_sim_top,'/home/josh/Documents/dsi/caps/cap3/Football_Supporter_Recommender/data/pickles/nfl_dists/packs_cos_sim.pickle')
    # pack_jac_top = packers.top_dists(distance_calc='jaccard',index_=dom_df.squad.unique(),col='jac_dist',number=None)
    # pickler(pack_jac_top,'/home/josh/Documents/dsi/caps/cap3/Football_Supporter_Recommender/data/pickles/nfl_dists/packs_jaccard.pickle')
    # print(pack_euc_top)
    # print(pack_cos_dist_top)
    # print(pack_cos_sim_top)
    # print(pack_jac_top)

    ## TEXANS ##
    # texans = SimilarityDF(nfl_df).vectorizer(['Texans'],'NFL')
    # tex_euc = Distances(texans,epl_mat)
    # tex_euc_top = tex_euc.top_dists(distance_calc='euclidean',index_=dom_df.squad.unique(),col='euc_dist',number=None)
    # pickler(tex_euc_top,'/home/josh/Documents/dsi/caps/cap3/Football_Supporter_Recommender/data/pickles/nfl_dists/tex_euc.pickle')
    # print(tex_euc_top)
    # tex_cos = Distances(texans,epl_mat)
    # tex_cos_sim_top = tex_cos.top_dists(distance_calc='cosine_sim',index_=dom_df.squad.unique(),col='cos_sim',number=None)[::-1]
    # pickler(tex_cos_sim_top,'/home/josh/Documents/dsi/caps/cap3/Football_Supporter_Recommender/data/pickles/nfl_dists/tex_cos_sim.pickle')
    # print(tex_cos_sim_top)
    # tex_cos_dist_top = tex_cos.top_dists(distance_calc='cosine_dist',index_=dom_df.squad.unique(),col='cos_dists',number=None)
    # pickler(tex_cos_dist_top,'/home/josh/Documents/dsi/caps/cap3/Football_Supporter_Recommender/data/pickles/nfl_dists/tex_cos_dist.pickle')
    # print(tex_cos_dist_top)
    # tex_jac = Distances(texans,epl_mat)
    # tex_jac_top = tex_jac.top_dists(distance_calc='jaccard',index_=dom_df.squad.unique(),col='jac_dists',number=None)
    # pickler(tex_jac_top,'/home/josh/Documents/dsi/caps/cap3/Football_Supporter_Recommender/data/pickles/nfl_dists/tex_jac.pickle')
    # print(tex_jac_top)
    
    ## COLTS ##
    # print('COLTS')
    # colts_vec = SimilarityDF(nfl_df).vectorizer(['Colts'],'NFL')
    # colts = Distances(colts_vec,epl_mat) 
    # colt_euc_top = colts.top_dists(distance_calc='euclidean',index_=dom_df.squad.unique(),col='euc_dist',number=None)
    # pickler(colt_euc_top,'/home/josh/Documents/dsi/caps/cap3/Football_Supporter_Recommender/data/pickles/nfl_dists/colts_euclidean.pickle')
    # colt_cos_dist_top = colts.top_dists(distance_calc='cosine_dist',index_=dom_df.squad.unique(),col='cos_dist',number=None)
    # pickler(colt_cos_dist_top,'/home/josh/Documents/dsi/caps/cap3/Football_Supporter_Recommender/data/pickles/nfl_dists/colts_cos_dist.pickle')
    # colt_cos_sim_top = colts.top_dists(distance_calc='cosine_sim',index_=dom_df.squad.unique(),col='cos_sim',number=None)[::-1]
    # pickler(colt_cos_sim_top,'/home/josh/Documents/dsi/caps/cap3/Football_Supporter_Recommender/data/pickles/nfl_dists/colts_cos_sim.pickle')
    # colt_jac_top = colts.top_dists(distance_calc='jaccard',index_=dom_df.squad.unique(),col='jac_dist',number=None)
    # pickler(colt_jac_top,'/home/josh/Documents/dsi/caps/cap3/Football_Supporter_Recommender/data/pickles/nfl_dists/colts_jaccard.pickle')
    # print(colt_euc_top)
    # print(colt_cos_dist_top)
    # print(colt_cos_sim_top)
    # print(colt_jac_top)

    ## JAGUARS ##
    # print('JAGUARS')
    # jaguars_vec = SimilarityDF(nfl_df).vectorizer(['Jaguars'],'NFL')
    # jaguars = Distances(jaguars_vec,epl_mat) 
    # jag_euc_top = jaguars.top_dists(distance_calc='euclidean',index_=dom_df.squad.unique(),col='euc_dist',number=None)
    # pickler(jag_euc_top,'/home/josh/Documents/dsi/caps/cap3/Football_Supporter_Recommender/data/pickles/nfl_dists/jags_euclidean.pickle')
    # jag_cos_dist_top = jaguars.top_dists(distance_calc='cosine_dist',index_=dom_df.squad.unique(),col='cos_dist',number=None)
    # pickler(jag_cos_dist_top,'/home/josh/Documents/dsi/caps/cap3/Football_Supporter_Recommender/data/pickles/nfl_dists/jags_cos_dist.pickle')
    # jag_cos_sim_top = jaguars.top_dists(distance_calc='cosine_sim',index_=dom_df.squad.unique(),col='cos_sim',number=None)[::-1]
    # pickler(jag_cos_sim_top,'/home/josh/Documents/dsi/caps/cap3/Football_Supporter_Recommender/data/pickles/nfl_dists/jags_cos_sim.pickle')
    # jag_jac_top = jaguars.top_dists(distance_calc='jaccard',index_=dom_df.squad.unique(),col='jac_dist',number=None)
    # pickler(jag_jac_top,'/home/josh/Documents/dsi/caps/cap3/Football_Supporter_Recommender/data/pickles/nfl_dists/jags_jaccard.pickle')
    # print(jag_euc_top)
    # print(jag_cos_dist_top)
    # print(jag_cos_sim_top)
    # print(jag_jac_top) 

    ## CHIEFS ##
    # print('CHIEFS')
    # chiefs_vec = SimilarityDF(nfl_df).vectorizer(['Chiefs'],'NFL')
    # chiefs = Distances(chiefs_vec,epl_mat) 
    # chf_euc_top = chiefs.top_dists(distance_calc='euclidean',index_=dom_df.squad.unique(),col='euc_dist',number=None)
    # pickler(chf_euc_top,'/home/josh/Documents/dsi/caps/cap3/Football_Supporter_Recommender/data/pickles/nfl_dists/chfs_euclidean.pickle')
    # chf_cos_dist_top = chiefs.top_dists(distance_calc='cosine_dist',index_=dom_df.squad.unique(),col='cos_dist',number=None)
    # pickler(chf_cos_dist_top,'/home/josh/Documents/dsi/caps/cap3/Football_Supporter_Recommender/data/pickles/nfl_dists/chfs_cos_dist.pickle')
    # chf_cos_sim_top = chiefs.top_dists(distance_calc='cosine_sim',index_=dom_df.squad.unique(),col='cos_sim',number=None)[::-1]
    # pickler(chf_cos_sim_top,'/home/josh/Documents/dsi/caps/cap3/Football_Supporter_Recommender/data/pickles/nfl_dists/chfs_cos_sim.pickle')
    # chf_jac_top = chiefs.top_dists(distance_calc='jaccard',index_=dom_df.squad.unique(),col='jac_dist',number=None)
    # pickler(chf_jac_top,'/home/josh/Documents/dsi/caps/cap3/Football_Supporter_Recommender/data/pickles/nfl_dists/chfs_jaccard.pickle')
    # print(chf_euc_top)
    # print(chf_cos_dist_top)
    # print(chf_cos_sim_top)
    # print(chf_jac_top)

    ## CHARGERS ##
    # print('CHARGERS')
    # chargers_vec = SimilarityDF(nfl_df).vectorizer(['Chargers'],'NFL')
    # chargers = Distances(chargers_vec,epl_mat) 
    # chrg_euc_top = chargers.top_dists(distance_calc='euclidean',index_=dom_df.squad.unique(),col='euc_dist',number=None)
    # pickler(chrg_euc_top,'/home/josh/Documents/dsi/caps/cap3/Football_Supporter_Recommender/data/pickles/nfl_dists/chrgs_euclidean.pickle')
    # chrg_cos_dist_top = chargers.top_dists(distance_calc='cosine_dist',index_=dom_df.squad.unique(),col='cos_dist',number=None)
    # pickler(chrg_cos_dist_top,'/home/josh/Documents/dsi/caps/cap3/Football_Supporter_Recommender/data/pickles/nfl_dists/chrgs_cos_dist.pickle')
    # chrg_cos_sim_top = chargers.top_dists(distance_calc='cosine_sim',index_=dom_df.squad.unique(),col='cos_sim',number=None)[::-1]
    # pickler(chrg_cos_sim_top,'/home/josh/Documents/dsi/caps/cap3/Football_Supporter_Recommender/data/pickles/nfl_dists/chrgs_cos_sim.pickle')
    # chrg_jac_top = chargers.top_dists(distance_calc='jaccard',index_=dom_df.squad.unique(),col='jac_dist',number=None)
    # pickler(chrg_jac_top,'/home/josh/Documents/dsi/caps/cap3/Football_Supporter_Recommender/data/pickles/nfl_dists/chrgs_jaccard.pickle')
    # print(chrg_euc_top)
    # print(chrg_cos_dist_top)
    # print(chrg_cos_sim_top)
    # print(chrg_jac_top)

    ## RAMS ##
    # print('RAMS')
    # rams_vec = SimilarityDF(nfl_df).vectorizer(['Rams'],'NFL')
    # rams = Distances(rams_vec,epl_mat) 
    # ram_euc_top = rams.top_dists(distance_calc='euclidean',index_=dom_df.squad.unique(),col='euc_dist',number=None)
    # pickler(ram_euc_top,'/home/josh/Documents/dsi/caps/cap3/Football_Supporter_Recommender/data/pickles/nfl_dists/rams_euclidean.pickle')
    # ram_cos_dist_top = rams.top_dists(distance_calc='cosine_dist',index_=dom_df.squad.unique(),col='cos_dist',number=None)
    # pickler(ram_cos_dist_top,'/home/josh/Documents/dsi/caps/cap3/Football_Supporter_Recommender/data/pickles/nfl_dists/rams_cos_dist.pickle')
    # ram_cos_sim_top = rams.top_dists(distance_calc='cosine_sim',index_=dom_df.squad.unique(),col='cos_sim',number=None)[::-1]
    # pickler(ram_cos_sim_top,'/home/josh/Documents/dsi/caps/cap3/Football_Supporter_Recommender/data/pickles/nfl_dists/rams_cos_sim.pickle')
    # ram_jac_top = rams.top_dists(distance_calc='jaccard',index_=dom_df.squad.unique(),col='jac_dist',number=None)
    # pickler(ram_jac_top,'/home/josh/Documents/dsi/caps/cap3/Football_Supporter_Recommender/data/pickles/nfl_dists/rams_jaccard.pickle')
    # print(ram_euc_top)
    # print(ram_cos_dist_top)
    # print(ram_cos_sim_top)
    # print(ram_jac_top)

    ## DOLPHINS ##
    # print('DOLPHINS')
    # dolphins_vec = SimilarityDF(nfl_df).vectorizer(['Dolphins'],'NFL')
    # dolphins = Distances(dolphins_vec,epl_mat) 
    # dolph_euc_top = dolphins.top_dists(distance_calc='euclidean',index_=dom_df.squad.unique(),col='euc_dist',number=None)
    # pickler(dolph_euc_top,'/home/josh/Documents/dsi/caps/cap3/Football_Supporter_Recommender/data/pickles/nfl_dists/dolphs_euclidean.pickle')
    # dolph_cos_dist_top = dolphins.top_dists(distance_calc='cosine_dist',index_=dom_df.squad.unique(),col='cos_dist',number=None)
    # pickler(dolph_cos_dist_top,'/home/josh/Documents/dsi/caps/cap3/Football_Supporter_Recommender/data/pickles/nfl_dists/dolphs_cos_dist.pickle')
    # dolph_cos_sim_top = dolphins.top_dists(distance_calc='cosine_sim',index_=dom_df.squad.unique(),col='cos_sim',number=None)[::-1]
    # pickler(dolph_cos_sim_top,'/home/josh/Documents/dsi/caps/cap3/Football_Supporter_Recommender/data/pickles/nfl_dists/dolphs_cos_sim.pickle')
    # dolph_jac_top = dolphins.top_dists(distance_calc='jaccard',index_=dom_df.squad.unique(),col='jac_dist',number=None)
    # pickler(dolph_jac_top,'/home/josh/Documents/dsi/caps/cap3/Football_Supporter_Recommender/data/pickles/nfl_dists/dolphs_jaccard.pickle')
    # print(dolph_euc_top)
    # print(dolph_cos_dist_top)
    # print(dolph_cos_sim_top)
    # print(dolph_jac_top)

    ## VIKINGS ##
    # print('VIKINGS')
    # vikings_vec = SimilarityDF(nfl_df).vectorizer(['Vikings'],'NFL')
    # vikings = Distances(vikings_vec,epl_mat) 
    # vik_euc_top = vikings.top_dists(distance_calc='euclidean',index_=dom_df.squad.unique(),col='euc_dist',number=None)
    # pickler(vik_euc_top,'/home/josh/Documents/dsi/caps/cap3/Football_Supporter_Recommender/data/pickles/nfl_dists/viks_euclidean.pickle')
    # vik_cos_dist_top = vikings.top_dists(distance_calc='cosine_dist',index_=dom_df.squad.unique(),col='cos_dist',number=None)
    # pickler(vik_cos_dist_top,'/home/josh/Documents/dsi/caps/cap3/Football_Supporter_Recommender/data/pickles/nfl_dists/viks_cos_dist.pickle')
    # vik_cos_sim_top = vikings.top_dists(distance_calc='cosine_sim',index_=dom_df.squad.unique(),col='cos_sim',number=None)[::-1]
    # pickler(vik_cos_sim_top,'/home/josh/Documents/dsi/caps/cap3/Football_Supporter_Recommender/data/pickles/nfl_dists/viks_cos_sim.pickle')
    # vik_jac_top = vikings.top_dists(distance_calc='jaccard',index_=dom_df.squad.unique(),col='jac_dist',number=None)
    # pickler(vik_jac_top,'/home/josh/Documents/dsi/caps/cap3/Football_Supporter_Recommender/data/pickles/nfl_dists/viks_jaccard.pickle')
    # print(vik_euc_top)
    # print(vik_cos_dist_top)
    # print(vik_cos_sim_top)
    # print(vik_jac_top)

    ## PATRIOTS ##
    # print('PATRIOTS')
    # patriots_vec = SimilarityDF(nfl_df).vectorizer(['Patriots'],'NFL')
    # patriots = Distances(patriots_vec,epl_mat) 
    # pat_euc_top = patriots.top_dists(distance_calc='euclidean',index_=dom_df.squad.unique(),col='euc_dist',number=None)
    # pickler(pat_euc_top,'/home/josh/Documents/dsi/caps/cap3/Football_Supporter_Recommender/data/pickles/nfl_dists/pats_euclidean.pickle')
    # pat_cos_dist_top = patriots.top_dists(distance_calc='cosine_dist',index_=dom_df.squad.unique(),col='cos_dist',number=None)
    # pickler(pat_cos_dist_top,'/home/josh/Documents/dsi/caps/cap3/Football_Supporter_Recommender/data/pickles/nfl_dists/pats_cos_dist.pickle')
    # pat_cos_sim_top = patriots.top_dists(distance_calc='cosine_sim',index_=dom_df.squad.unique(),col='cos_sim',number=None)[::-1]
    # pickler(pat_cos_sim_top,'/home/josh/Documents/dsi/caps/cap3/Football_Supporter_Recommender/data/pickles/nfl_dists/pats_cos_sim.pickle')
    # pat_jac_top = patriots.top_dists(distance_calc='jaccard',index_=dom_df.squad.unique(),col='jac_dist',number=None)
    # pickler(pat_jac_top,'/home/josh/Documents/dsi/caps/cap3/Football_Supporter_Recommender/data/pickles/nfl_dists/pats_jaccard.pickle')
    # print(pat_euc_top)
    # print(pat_cos_dist_top)
    # print(pat_cos_sim_top)
    # print(pat_jac_top)

    ## SAINTS ##
    # print('SAINTS')
    # saints_vec = SimilarityDF(nfl_df).vectorizer(['Saints'],'NFL')
    # saints = Distances(saints_vec,epl_mat) 
    # saint_euc_top = saints.top_dists(distance_calc='euclidean',index_=dom_df.squad.unique(),col='euc_dist',number=None)
    # pickler(saint_euc_top,'/home/josh/Documents/dsi/caps/cap3/Football_Supporter_Recommender/data/pickles/nfl_dists/saints_euclidean.pickle')
    # saint_cos_dist_top = saints.top_dists(distance_calc='cosine_dist',index_=dom_df.squad.unique(),col='cos_dist',number=None)
    # pickler(saint_cos_dist_top,'/home/josh/Documents/dsi/caps/cap3/Football_Supporter_Recommender/data/pickles/nfl_dists/saints_cos_dist.pickle')
    # saint_cos_sim_top = saints.top_dists(distance_calc='cosine_sim',index_=dom_df.squad.unique(),col='cos_sim',number=None)[::-1]
    # pickler(saint_cos_sim_top,'/home/josh/Documents/dsi/caps/cap3/Football_Supporter_Recommender/data/pickles/nfl_dists/saints_cos_sim.pickle')
    # saint_jac_top = saints.top_dists(distance_calc='jaccard',index_=dom_df.squad.unique(),col='jac_dist',number=None)
    # pickler(saint_jac_top,'/home/josh/Documents/dsi/caps/cap3/Football_Supporter_Recommender/data/pickles/nfl_dists/saints_jaccard.pickle')
    # print(saint_euc_top)
    # print(saint_cos_dist_top)
    # print(saint_cos_sim_top)
    # print(saint_jac_top)

    ## GIANTS ##
    # print('GIANTS')
    # giants_vec = SimilarityDF(nfl_df).vectorizer(['Giants'],'NFL')
    # giants = Distances(giants_vec,epl_mat) 
    # giant_euc_top = giants.top_dists(distance_calc='euclidean',index_=dom_df.squad.unique(),col='euc_dist',number=None)
    # pickler(giant_euc_top,'/home/josh/Documents/dsi/caps/cap3/Football_Supporter_Recommender/data/pickles/nfl_dists/giants_euclidean.pickle')
    # giant_cos_dist_top = giants.top_dists(distance_calc='cosine_dist',index_=dom_df.squad.unique(),col='cos_dist',number=None)
    # pickler(giant_cos_dist_top,'/home/josh/Documents/dsi/caps/cap3/Football_Supporter_Recommender/data/pickles/nfl_dists/giants_cos_dist.pickle')
    # giant_cos_sim_top = giants.top_dists(distance_calc='cosine_sim',index_=dom_df.squad.unique(),col='cos_sim',number=None)[::-1]
    # pickler(giant_cos_sim_top,'/home/josh/Documents/dsi/caps/cap3/Football_Supporter_Recommender/data/pickles/nfl_dists/giants_cos_sim.pickle')
    # giant_jac_top = giants.top_dists(distance_calc='jaccard',index_=dom_df.squad.unique(),col='jac_dist',number=None)
    # pickler(giant_jac_top,'/home/josh/Documents/dsi/caps/cap3/Football_Supporter_Recommender/data/pickles/nfl_dists/giants_jaccard.pickle')
    # print(giant_euc_top)
    # print(giant_cos_dist_top)
    # print(giant_cos_sim_top)
    # print(giant_jac_top)

    ## JETS ##
    # print('JETS')
    # jets_vec = SimilarityDF(nfl_df).vectorizer(['Jets'],'NFL')
    # jets = Distances(jets_vec,epl_mat) 
    # jets_euc_top = jets.top_dists(distance_calc='euclidean',index_=dom_df.squad.unique(),col='euc_dist',number=None)
    # pickler(jets_euc_top,'/home/josh/Documents/dsi/caps/cap3/Football_Supporter_Recommender/data/pickles/nfl_dists/jetss_euclidean.pickle')
    # jets_cos_dist_top = jets.top_dists(distance_calc='cosine_dist',index_=dom_df.squad.unique(),col='cos_dist',number=None)
    # pickler(jets_cos_dist_top,'/home/josh/Documents/dsi/caps/cap3/Football_Supporter_Recommender/data/pickles/nfl_dists/jetss_cos_dist.pickle')
    # jets_cos_sim_top = jets.top_dists(distance_calc='cosine_sim',index_=dom_df.squad.unique(),col='cos_sim',number=None)[::-1]
    # pickler(jets_cos_sim_top,'/home/josh/Documents/dsi/caps/cap3/Football_Supporter_Recommender/data/pickles/nfl_dists/jetss_cos_sim.pickle')
    # jets_jac_top = jets.top_dists(distance_calc='jaccard',index_=dom_df.squad.unique(),col='jac_dist',number=None)
    # pickler(jets_jac_top,'/home/josh/Documents/dsi/caps/cap3/Football_Supporter_Recommender/data/pickles/nfl_dists/jetss_jaccard.pickle')
    # print(jets_euc_top)
    # print(jets_cos_dist_top)
    # print(jets_cos_sim_top)
    # print(jets_jac_top)

    ## RAIDERS ##
    # print('RAIDERS')
    # raiders_vec = SimilarityDF(nfl_df).vectorizer(['Raiders'],'NFL')
    # raiders = Distances(raiders_vec,epl_mat) 
    # raid_euc_top = raiders.top_dists(distance_calc='euclidean',index_=dom_df.squad.unique(),col='euc_dist',number=None)
    # pickler(raid_euc_top,'/home/josh/Documents/dsi/caps/cap3/Football_Supporter_Recommender/data/pickles/nfl_dists/raids_euclidean.pickle')
    # raid_cos_dist_top = raiders.top_dists(distance_calc='cosine_dist',index_=dom_df.squad.unique(),col='cos_dist',number=None)
    # pickler(raid_cos_dist_top,'/home/josh/Documents/dsi/caps/cap3/Football_Supporter_Recommender/data/pickles/nfl_dists/raids_cos_dist.pickle')
    # raid_cos_sim_top = raiders.top_dists(distance_calc='cosine_sim',index_=dom_df.squad.unique(),col='cos_sim',number=None)[::-1]
    # pickler(raid_cos_sim_top,'/home/josh/Documents/dsi/caps/cap3/Football_Supporter_Recommender/data/pickles/nfl_dists/raids_cos_sim.pickle')
    # raid_jac_top = raiders.top_dists(distance_calc='jaccard',index_=dom_df.squad.unique(),col='jac_dist',number=None)
    # pickler(raid_jac_top,'/home/josh/Documents/dsi/caps/cap3/Football_Supporter_Recommender/data/pickles/nfl_dists/raids_jaccard.pickle')
    # print(raid_euc_top)
    # print(raid_cos_dist_top)
    # print(raid_cos_sim_top)
    # print(raid_jac_top)

    ## EAGLES ##
    # print('EAGLES')
    # eagles_vec = SimilarityDF(nfl_df).vectorizer(['Eagles'],'NFL')
    # eagles = Distances(eagles_vec,epl_mat) 
    # eag_euc_top = eagles.top_dists(distance_calc='euclidean',index_=dom_df.squad.unique(),col='euc_dist',number=None)
    # pickler(eag_euc_top,'/home/josh/Documents/dsi/caps/cap3/Football_Supporter_Recommender/data/pickles/nfl_dists/eags_euclidean.pickle')
    # eag_cos_dist_top = eagles.top_dists(distance_calc='cosine_dist',index_=dom_df.squad.unique(),col='cos_dist',number=None)
    # pickler(eag_cos_dist_top,'/home/josh/Documents/dsi/caps/cap3/Football_Supporter_Recommender/data/pickles/nfl_dists/eags_cos_dist.pickle')
    # eag_cos_sim_top = eagles.top_dists(distance_calc='cosine_sim',index_=dom_df.squad.unique(),col='cos_sim',number=None)[::-1]
    # pickler(eag_cos_sim_top,'/home/josh/Documents/dsi/caps/cap3/Football_Supporter_Recommender/data/pickles/nfl_dists/eags_cos_sim.pickle')
    # eag_jac_top = eagles.top_dists(distance_calc='jaccard',index_=dom_df.squad.unique(),col='jac_dist',number=None)
    # pickler(eag_jac_top,'/home/josh/Documents/dsi/caps/cap3/Football_Supporter_Recommender/data/pickles/nfl_dists/eags_jaccard.pickle')
    # print(eag_euc_top)
    # print(eag_cos_dist_top)
    # print(eag_cos_sim_top)
    # print(eag_jac_top)

    ## STEELERS ##
    # print('STEELERS')
    # steelers_vec = SimilarityDF(nfl_df).vectorizer(['Steelers'],'NFL')
    # steelers = Distances(steelers_vec,epl_mat) 
    # steel_euc_top = steelers.top_dists(distance_calc='euclidean',index_=dom_df.squad.unique(),col='euc_dist',number=None)
    # pickler(steel_euc_top,'/home/josh/Documents/dsi/caps/cap3/Football_Supporter_Recommender/data/pickles/nfl_dists/steels_euclidean.pickle')
    # steel_cos_dist_top = steelers.top_dists(distance_calc='cosine_dist',index_=dom_df.squad.unique(),col='cos_dist',number=None)
    # pickler(steel_cos_dist_top,'/home/josh/Documents/dsi/caps/cap3/Football_Supporter_Recommender/data/pickles/nfl_dists/steels_cos_dist.pickle')
    # steel_cos_sim_top = steelers.top_dists(distance_calc='cosine_sim',index_=dom_df.squad.unique(),col='cos_sim',number=None)[::-1]
    # pickler(steel_cos_sim_top,'/home/josh/Documents/dsi/caps/cap3/Football_Supporter_Recommender/data/pickles/nfl_dists/steels_cos_sim.pickle')
    # steel_jac_top = steelers.top_dists(distance_calc='jaccard',index_=dom_df.squad.unique(),col='jac_dist',number=None)
    # pickler(steel_jac_top,'/home/josh/Documents/dsi/caps/cap3/Football_Supporter_Recommender/data/pickles/nfl_dists/steels_jaccard.pickle')
    # print(steel_euc_top)
    # print(steel_cos_dist_top)
    # print(steel_cos_sim_top)
    # print(steel_jac_top)

    ## 49ERS ##
    # print('49ERS')
    # niners_vec = SimilarityDF(nfl_df).vectorizer(['49ers'],'NFL')
    # niners = Distances(niners_vec,epl_mat) 
    # nine_euc_top = niners.top_dists(distance_calc='euclidean',index_=dom_df.squad.unique(),col='euc_dist',number=None)
    # pickler(nine_euc_top,'/home/josh/Documents/dsi/caps/cap3/Football_Supporter_Recommender/data/pickles/nfl_dists/nines_euclidean.pickle')
    # nine_cos_dist_top = niners.top_dists(distance_calc='cosine_dist',index_=dom_df.squad.unique(),col='cos_dist',number=None)
    # pickler(nine_cos_dist_top,'/home/josh/Documents/dsi/caps/cap3/Football_Supporter_Recommender/data/pickles/nfl_dists/nines_cos_dist.pickle')
    # nine_cos_sim_top = niners.top_dists(distance_calc='cosine_sim',index_=dom_df.squad.unique(),col='cos_sim',number=None)[::-1]
    # pickler(nine_cos_sim_top,'/home/josh/Documents/dsi/caps/cap3/Football_Supporter_Recommender/data/pickles/nfl_dists/nines_cos_sim.pickle')
    # nine_jac_top = niners.top_dists(distance_calc='jaccard',index_=dom_df.squad.unique(),col='jac_dist',number=None)
    # pickler(nine_jac_top,'/home/josh/Documents/dsi/caps/cap3/Football_Supporter_Recommender/data/pickles/nfl_dists/nines_jaccard.pickle')
    # print(nine_euc_top)
    # print(nine_cos_dist_top)
    # print(nine_cos_sim_top)
    # print(nine_jac_top)

    ## SEAHAWKS ##
    # print('SEAHAWKS')
    # seahawks_vec = SimilarityDF(nfl_df).vectorizer(['Seahawks'],'NFL')
    # seahawks = Distances(seahawks_vec,epl_mat) 
    # shawks_euc_top = seahawks.top_dists(distance_calc='euclidean',index_=dom_df.squad.unique(),col='euc_dist',number=None)
    # pickler(shawks_euc_top,'/home/josh/Documents/dsi/caps/cap3/Football_Supporter_Recommender/data/pickles/nfl_dists/shawkss_euclidean.pickle')
    # shawks_cos_dist_top = seahawks.top_dists(distance_calc='cosine_dist',index_=dom_df.squad.unique(),col='cos_dist',number=None)
    # pickler(shawks_cos_dist_top,'/home/josh/Documents/dsi/caps/cap3/Football_Supporter_Recommender/data/pickles/nfl_dists/shawkss_cos_dist.pickle')
    # shawks_cos_sim_top = seahawks.top_dists(distance_calc='cosine_sim',index_=dom_df.squad.unique(),col='cos_sim',number=None)[::-1]
    # pickler(shawks_cos_sim_top,'/home/josh/Documents/dsi/caps/cap3/Football_Supporter_Recommender/data/pickles/nfl_dists/shawkss_cos_sim.pickle')
    # shawks_jac_top = seahawks.top_dists(distance_calc='jaccard',index_=dom_df.squad.unique(),col='jac_dist',number=None)
    # pickler(shawks_jac_top,'/home/josh/Documents/dsi/caps/cap3/Football_Supporter_Recommender/data/pickles/nfl_dists/shawkss_jaccard.pickle')
    # print(shawks_euc_top)
    # print(shawks_cos_dist_top)
    # print(shawks_cos_sim_top)
    # print(shawks_jac_top)

    ## BUCCANEERS ##
    # print('BUCCANEERS')
    # buccaneers_vec = SimilarityDF(nfl_df).vectorizer(['Buccaneers'],'NFL')
    # buccaneers = Distances(buccaneers_vec,epl_mat) 
    # bucc_euc_top = buccaneers.top_dists(distance_calc='euclidean',index_=dom_df.squad.unique(),col='euc_dist',number=None)
    # pickler(bucc_euc_top,'/home/josh/Documents/dsi/caps/cap3/Football_Supporter_Recommender/data/pickles/nfl_dists/buccs_euclidean.pickle')
    # bucc_cos_dist_top = buccaneers.top_dists(distance_calc='cosine_dist',index_=dom_df.squad.unique(),col='cos_dist',number=None)
    # pickler(bucc_cos_dist_top,'/home/josh/Documents/dsi/caps/cap3/Football_Supporter_Recommender/data/pickles/nfl_dists/buccs_cos_dist.pickle')
    # bucc_cos_sim_top = buccaneers.top_dists(distance_calc='cosine_sim',index_=dom_df.squad.unique(),col='cos_sim',number=None)[::-1]
    # pickler(bucc_cos_sim_top,'/home/josh/Documents/dsi/caps/cap3/Football_Supporter_Recommender/data/pickles/nfl_dists/buccs_cos_sim.pickle')
    # bucc_jac_top = buccaneers.top_dists(distance_calc='jaccard',index_=dom_df.squad.unique(),col='jac_dist',number=None)
    # pickler(bucc_jac_top,'/home/josh/Documents/dsi/caps/cap3/Football_Supporter_Recommender/data/pickles/nfl_dists/buccs_jaccard.pickle')
    # print(bucc_euc_top)
    # print(bucc_cos_dist_top)
    # print(bucc_cos_sim_top)
    # print(bucc_jac_top)

    ## TITANS ##
    # print('TITANS')
    # titans_vec = SimilarityDF(nfl_df).vectorizer(['Titans'],'NFL')
    # titans = Distances(titans_vec,epl_mat) 
    # tit_euc_top = titans.top_dists(distance_calc='euclidean',index_=dom_df.squad.unique(),col='euc_dist',number=None)
    # pickler(tit_euc_top,'/home/josh/Documents/dsi/caps/cap3/Football_Supporter_Recommender/data/pickles/nfl_dists/tits_euclidean.pickle')
    # tit_cos_dist_top = titans.top_dists(distance_calc='cosine_dist',index_=dom_df.squad.unique(),col='cos_dist',number=None)
    # pickler(tit_cos_dist_top,'/home/josh/Documents/dsi/caps/cap3/Football_Supporter_Recommender/data/pickles/nfl_dists/tits_cos_dist.pickle')
    # tit_cos_sim_top = titans.top_dists(distance_calc='cosine_sim',index_=dom_df.squad.unique(),col='cos_sim',number=None)[::-1]
    # pickler(tit_cos_sim_top,'/home/josh/Documents/dsi/caps/cap3/Football_Supporter_Recommender/data/pickles/nfl_dists/tits_cos_sim.pickle')
    # tit_jac_top = titans.top_dists(distance_calc='jaccard',index_=dom_df.squad.unique(),col='jac_dist',number=None)
    # pickler(tit_jac_top,'/home/josh/Documents/dsi/caps/cap3/Football_Supporter_Recommender/data/pickles/nfl_dists/tits_jaccard.pickle')
    # print(tit_euc_top)
    # print(tit_cos_dist_top)
    # print(tit_cos_sim_top)
    # print(tit_jac_top)

    ## REDSKINS ##
    # print('REDSKINS')
    # redskins_vec = SimilarityDF(nfl_df).vectorizer(['Redskins'],'NFL')
    # redskins = Distances(redskins_vec,epl_mat) 
    # red_euc_top = redskins.top_dists(distance_calc='euclidean',index_=dom_df.squad.unique(),col='euc_dist',number=None)
    # pickler(red_euc_top,'/home/josh/Documents/dsi/caps/cap3/Football_Supporter_Recommender/data/pickles/nfl_dists/reds_euclidean.pickle')
    # red_cos_dist_top = redskins.top_dists(distance_calc='cosine_dist',index_=dom_df.squad.unique(),col='cos_dist',number=None)
    # pickler(red_cos_dist_top,'/home/josh/Documents/dsi/caps/cap3/Football_Supporter_Recommender/data/pickles/nfl_dists/reds_cos_dist.pickle')
    # red_cos_sim_top = redskins.top_dists(distance_calc='cosine_sim',index_=dom_df.squad.unique(),col='cos_sim',number=None)[::-1]
    # pickler(red_cos_sim_top,'/home/josh/Documents/dsi/caps/cap3/Football_Supporter_Recommender/data/pickles/nfl_dists/reds_cos_sim.pickle')
    # red_jac_top = redskins.top_dists(distance_calc='jaccard',index_=dom_df.squad.unique(),col='jac_dist',number=None)
    # pickler(red_jac_top,'/home/josh/Documents/dsi/caps/cap3/Football_Supporter_Recommender/data/pickles/nfl_dists/reds_jaccard.pickle')
    # print(red_euc_top)
    # print(red_cos_dist_top)
    # print(red_cos_sim_top)
    # print(red_jac_top)