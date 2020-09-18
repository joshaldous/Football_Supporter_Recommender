import flask
import sys
sys.path.append('/home/josh/Documents/dsi/caps/cap3/Football_Supporter_Recommender/src/')
import working_mod 
import working_eda
from working_mod import pickler, unpickler, SimilarityDF, Distances
from working_eda import LeagueDFEDA
import pickle

app = flask.Flask(__name__)
PORT = 8105
REGISTER_URL = "http://localhost:5000/index"
DATA = []
TIMESTAMP = []

img_dict = {'Arsenal':'epl/18bb7c10.png','Aston Villa':'epl/8602292d.png','Bournemouth':'epl/4ba7cbea.png',
            'Brighton':'epl/d07537b9.png','Burnley':'epl/943e8050.png','Chelsea':'epl/cff3d9bb.png',
            'Crystal Palace':'epl/47c64c55.png','Everton':'epl/d3fd31cc.png','Leicester City':'epl/a2d435b3.png',
            'Liverpool':'epl/822bd0ba.png','Manchester City':'epl/b8fd03ef.png','Manchester Utd':'epl/19538871.png',
            'Newcastle Utd':'epl/b2b47a98.png','Norwich City':'epl/1c781004.png','Sheffield Utd':'epl/1df6b87e.png',
            'Southampton':'epl/33c895d4.png','Tottenham':'epl/361ca564.png','Watford':'epl/2abfe087.png',
            'West Ham':'epl/7c21e445.png','Wolves':'epl/8cec06e1.png'}

url_dict = {'Arsenal':'https://www.arsenal.com','Aston Villa':'https://avfc.co.uk','Bournemouth':'https://afcb.co.uk',
            'Brighton':'https://brightonandhovealbion.com','Burnley':'https://burnleyfootballclub.com','Chelsea':'https://chelseafc.com',
            'Crystal Palace':'https://cpfc.co.uk','Everton':'https://evertonfc.com','Leicester City':'https://lcfc.com',
            'Liverpool':'https://liverpoolfc.com','Manchester City':'https://mancity.com','Manchester Utd':'https://manutd.com',
            'Newcastle Utd':'https://nufc.co.uk','Norwich City':'https://canaries.co.uk','Sheffield Utd':'https://sufc.co.uk',
            'Southampton':'https://southamptonfc.com','Tottenham':'https://tottenhamhotspur.com','Watford':'https://watfordfc.com',
            'West Ham':'https://whufc.com','Wolves':'https://wolves.co.uk'}

goal_dict = {'Arsenal':'https://www.youtube.com/watch?v=yO31QsfmnRE','Aston Villa':'https://www.youtube.com/watch?v=2YMJPhN3wnc',
             'Bournemouth':'https://www.youtube.com/watch?v=c9VjOPnCDC0','Brighton':'https://www.youtube.com/watch?v=8-jvCuZ7Sc8',
             'Burnley':'https://www.youtube.com/watch?v=mcfqxZIb6do','Chelsea':'https://www.youtube.com/watch?v=RXeT0RKb1X0',
            'Crystal Palace':'https://www.youtube.com/watch?v=URWKcozh4T0','Everton':'https://www.youtube.com/watch?v=1mE-OvnWoaI',
            'Leicester City':'https://www.youtube.com/watch?v=HMijNgNnqHc','Liverpool':'https://www.youtube.com/watch?v=BO0uy4CivMM',
            'Manchester City':'https://www.youtube.com/watch?v=z61dIkxmPWw','Manchester Utd':'https://www.youtube.com/watch?v=-Q3tBEysdsQ',
            'Newcastle Utd':'https://www.youtube.com/watch?v=kLhcPacf3Ww','Norwich City':'https://www.youtube.com/watch?v=ymsMqP0f2bk',
            'Sheffield Utd':'https://www.youtube.com/watch?v=U9Tjk-AsLps','Southampton':'https://www.youtube.com/watch?v=24U454qJwLg',
            'Tottenham':'https://www.youtube.com/watch?v=f8MxdAkf4xY','Watford':'https://www.youtube.com/watch?v=988yKC-odmM',
            'West Ham':'https://www.youtube.com/watch?v=hxMg3wswzZU','Wolves':'https://www.youtube.com/watch?v=ICyVBpnSOfQ'}

@app.route('/', methods = ['GET','POST'])

def index():
    return flask.render_template('index.html')
    
    
@app.route('/predict', methods = ['GET','POST'])
def predict ():
    epl_df = unpickler('/home/josh/Documents/dsi/caps/cap3/Football_Supporter_Recommender/data/pickles/epl_domestic_league_df_clean.pickle')
    epl_mat = unpickler('/home/josh/Documents/dsi/caps/cap3/Football_Supporter_Recommender/data/pickles/epl_dists/epl_vectorized.pickle')
    nfl_df = unpickler('/home/josh/Documents/dsi/caps/cap3/Football_Supporter_Recommender/data/pickles/NFL_to_vector.pickle')
    team = flask.request.form['team']
    print(team)
    team_vec = SimilarityDF(nfl_df).vectorizer([team],'NFL')
    selected = Distances(team_vec,epl_mat)
    team_euc_top = selected.top_dists(distance_calc='euclidean',index_=epl_df.squad.unique(),col='euc_dist',number=3)
    team_cos_dist_top = selected.top_dists(distance_calc='cosine_dist',index_=epl_df.squad.unique(),col='cos_dist',number=3)
    # team_cos_sim_top = selected.top_dists(distance_calc='cosine_sim',index_=epl_df.squad.unique(),col='cos_sim',number=3)[::-1]
    # team_jac_top = selected.top_dists(distance_calc='jaccard',index_=epl_df.squad.unique(),col='jac_dist',number=3)
    # return flask.render_template('index_album.html')
    print(team_euc_top)
    idx = [x for x in team_euc_top.index]
    team1 = idx[0]
    print(team1)
    team2 = idx[1]
    team3 = idx[2]
    team1_img = img_dict[team1]
    team1_info = url_dict[team1]
    team1_goals = goal_dict[team1]
    team2_img = img_dict[team2]
    team2_info = url_dict[team2]
    team2_goals = goal_dict[team2]
    team3_img = img_dict[team3]
    team3_info = url_dict[team3]
    team3_goals = goal_dict[team3]
    return flask.render_template('index_album.html', team1=team1, team1_img=team1_img, team1_info=team1_info, team1_goals=team1_goals,
                                 team2=team2, team2_img=team2_img, team2_info=team2_info, team2_goals=team2_goals,
                                 team3=team3, team3_img=team3_img, team3_info=team3_info, team3_goals=team3_goals)
    
@app.route('/info', methods = ['GET','POST'])
def info():
    team_url = flask.request.form['team']
    print(team_url)
    url = flask.request.form[team_url]
    print(url)
    return flask.redirect(url,code=302)
    
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=PORT, debug=True)