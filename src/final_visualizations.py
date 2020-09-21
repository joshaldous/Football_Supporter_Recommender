import numpy as np
import pandas as pd
from final_comparison import unpickler
from final_epl_eda import LeagueDFEDA
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.offsetbox import (TextArea, DrawingArea, OffsetImage,
                                  AnnotationBbox)
from matplotlib.cbook import get_sample_data

img_dict = {'Arsenal':'/home/josh/Documents/dsi/caps/cap3/Football_Supporter_Recommender/src/web_app/static/epl/18bb7c10.png','Aston Villa':'/home/josh/Documents/dsi/caps/cap3/Football_Supporter_Recommender/src/web_app/static/epl/8602292d.png','Bournemouth':'/home/josh/Documents/dsi/caps/cap3/Football_Supporter_Recommender/src/web_app/static/epl/4ba7cbea.png',
            'Brighton':'/home/josh/Documents/dsi/caps/cap3/Football_Supporter_Recommender/src/web_app/static/epl/d07537b9.png','Burnley':'/home/josh/Documents/dsi/caps/cap3/Football_Supporter_Recommender/src/web_app/static/epl/943e8050.png','Chelsea':'/home/josh/Documents/dsi/caps/cap3/Football_Supporter_Recommender/src/web_app/static/epl/cff3d9bb.png',
            'Crystal Palace':'/home/josh/Documents/dsi/caps/cap3/Football_Supporter_Recommender/src/web_app/static/epl/47c64c55.png','Everton':'/home/josh/Documents/dsi/caps/cap3/Football_Supporter_Recommender/src/web_app/static/epl/d3fd31cc.png','Leicester City':'/home/josh/Documents/dsi/caps/cap3/Football_Supporter_Recommender/src/web_app/static/epl/a2d435b3.png',
            'Liverpool':'/home/josh/Documents/dsi/caps/cap3/Football_Supporter_Recommender/src/web_app/static/epl/822bd0ba.png','Manchester City':'/home/josh/Documents/dsi/caps/cap3/Football_Supporter_Recommender/src/web_app/static/epl/b8fd03ef.png','Manchester Utd':'/home/josh/Documents/dsi/caps/cap3/Football_Supporter_Recommender/src/web_app/static/epl/19538871.png',
            'Newcastle Utd':'/home/josh/Documents/dsi/caps/cap3/Football_Supporter_Recommender/src/web_app/static/epl/b2b47a98.png','Norwich City':'/home/josh/Documents/dsi/caps/cap3/Football_Supporter_Recommender/src/web_app/static/epl/1c781004.png','Sheffield Utd':'/home/josh/Documents/dsi/caps/cap3/Football_Supporter_Recommender/src/web_app/static/epl/1df6b87e.png',
            'Southampton':'/home/josh/Documents/dsi/caps/cap3/Football_Supporter_Recommender/src/web_app/static/epl/33c895d4.png','Tottenham':'/home/josh/Documents/dsi/caps/cap3/Football_Supporter_Recommender/src/web_app/static/epl/361ca564.png','Watford':'/home/josh/Documents/dsi/caps/cap3/Football_Supporter_Recommender/src/web_app/static/epl/2abfe087.png',
            'West Ham':'/home/josh/Documents/dsi/caps/cap3/Football_Supporter_Recommender/src/web_app/static/epl/7c21e445.png','Wolves':'/home/josh/Documents/dsi/caps/cap3/Football_Supporter_Recommender/src/web_app/static/epl/8cec06e1.png'}

def scat_plt(df,column,title,save=None):
    plt.style.use('seaborn-dark')
    df1 = df.reset_index()
    holder = [1,1]
    holder = holder * 10
    coord = [0.006,0.006] 
    xcoord = [0.02,0.02]    
    epl = [x for x in df1.iloc[:,0]]
    cos = [x for x in df[column]]
    z = list(range(20))
    
    fig = plt.figure(figsize=(14,2))
    ax = fig.add_subplot(111,facecolor='lightslategrey')
    ax.set_ylim(bottom=0.98,top=1.02)

    ax.scatter(df,holder,c=z[::-1],cmap='plasma')
    ax.set_yticklabels([])
    
    img1 = get_sample_data(img_dict[epl[0]],asfileobj=False)
    img2 = get_sample_data(img_dict[epl[-1]],asfileobj=False)
    
    arr_img1 = plt.imread(img1,format='png')
    arr_img2 = plt.imread(img2,format='png')
    
    imagebox1 = OffsetImage(arr_img1,zoom=0.5)
    imagebox2 = OffsetImage(arr_img2,zoom=0.5)
    
    imagebox1.image.axes = ax
    imagebox2.image.axes = ax
    
    xy1 = ((cos[0]-xcoord[0],1+coord[0]))
    xy2 = ((cos[-1]-xcoord[-1],1+coord[-1]))
    
    ab1 = AnnotationBbox(imagebox1,xy1,xybox=(20.,20.),xycoords = 'data',boxcoords='offset points',pad=0.3)
    ab2 = AnnotationBbox(imagebox2,xy2,xybox=(20.,20.),xycoords = 'data',boxcoords='offset points',pad=0.3)
    
    ax.add_artist(ab1)
    ax.add_artist(ab2)

    plt.xticks(fontsize=24)
    plt.xlabel('EPL Cosine Distances',fontsize=28)
    plt.ylabel('Texans',fontsize=28)
    plt.tight_layout()
 
    plt.title(f'{title} Texans vs EPL Cosine Distances Scatter',fontsize=28)
    if save:
        plt.savefig(F'/home/josh/Documents/dsi/caps/cap3/Football_Supporter_Recommender/images/{save}.jpeg')
    plt.show()

def bar_plt(nfl_max,nfl_med,nfl_min,epl_max,epl_med,epl_min,title,save):
    plt.style.use('seaborn-dark')
    fig = plt.figure(figsize=(14,8))
    barWidth = 0.4
 
    nfl = [nfl_max,nfl_med,nfl_min]
    epl = [epl_max,epl_med,epl_min]

 
    b1 = np.arange(len(nfl))
    b2 = [x + barWidth for x in b1]
 
    plt.bar(b1, nfl, color='steelblue', width=barWidth, label='NFL')
    plt.bar(b2, epl, color='firebrick', width=barWidth, label='EPL')

    for i, v in enumerate(nfl):
        plt.text(i-.05,v-.3, str(round(v,2)), color='white',fontsize=24,fontweight='bold')
    for i, v in enumerate(epl):
        plt.text(i+0.33,v-.3, str(round(v,2)), color='white',fontsize=24,fontweight='bold')

    plt.xlabel('Max, Median and Min Scores per Game 2017-2019',fontsize=24)
    plt.ylabel('TDs and Goals per Game',fontsize=24)

    plt.yticks(fontsize=24)
    plt.xticks([r + barWidth/2 for r in range(len(nfl))], ['Max', 'Median', 'Min'],fontsize=24)
    
    plt.title(title,fontsize=28)
    plt.legend(fontsize=28)
    plt.savefig(f'/home/josh/Documents/dsi/caps/cap3/Football_Supporter_Recommender/images/{save}.jpeg')
    plt.show()









if __name__ == '__main__':
    texan_initial = unpickler('/home/josh/Documents/dsi/caps/cap3/Football_Supporter_Recommender/data/pickles/nfl_dists/tex_cos_dist.pickle')
    texan_final = unpickler('/home/josh/Documents/dsi/caps/cap3/Football_Supporter_Recommender/data/pickles/nfl_dists/tex_cos_dist_update2.pickle')

    scat_plt(texan_initial,'cos_dists','Initial','texan_test_initial')
    scat_plt(texan_final,'cos_dist','Finalized','texan_test_final')

    epl = unpickler('/home/josh/Documents/dsi/caps/cap3/Football_Supporter_Recommender/data/pickles/epl_domestic_league_df_clean_update.pickle')
    nfl = unpickler('/home/josh/Documents/dsi/caps/cap3/Football_Supporter_Recommender/data/pickles/NFL_df_clean_update.pickle')

    epl_max = epl.loc[epl.gf == np.max(epl.gf),'gf'].values[0]
    nfl_max = nfl.loc[nfl.tds == np.max(nfl.tds),'tds'].values[0]

    epl_med = epl.loc[epl.gf == 52,'gf'].values[0]
    nfl_med = nfl.loc[nfl.tds == np.median(nfl.tds),'tds'].values[0]

    epl_min = epl.loc[epl.gf == np.min(epl.gf),'gf'].values[0]
    nfl_min = nfl.loc[nfl.tds == np.min(nfl.tds),'tds'].values[0]

    epl_max_avg = epl_max / 38
    nfl_max_avg = nfl_max / 16

    epl_med_avg = epl_med / 38
    nfl_med_avg = nfl_med / 16

    epl_min_avg = epl_min / 38
    nfl_min_avg = nfl_min / 16

    bar_plt(nfl_max_avg,nfl_med_avg,nfl_min_avg,epl_max_avg,epl_med_avg,epl_min_avg,
            'Average TDs and Goals Scored per Game','scoring_per_game_bar')
