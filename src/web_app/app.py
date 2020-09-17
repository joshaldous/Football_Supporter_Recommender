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

img_dict = {'Arsenal':'18bb7c10.png','Aston Villa':'8602292d.png','Bournemouth':'4ba7cbea.png',
            'Brighton':'d07537b9.png','Burnley':'943e8050.png','Chelsea':'cff3d9bb.png',
            'Crystal Palace':'47c64c55.png','Everton':'d3fd31cc.png','Leicester City':'a2d435b3.png',
            'Liverpool':'822bd0ba.png','Manchester City':'b8fd03ef.png','Manchester United':'19538871.png',
            'Newcastle':'b2b47a98.png','Norwich City':'1c781004.png','Sheffield United':'1df6b87e.png',
            'Southampton':'33c895d4.png','Tottenham':'361ca564.png','Watford':'2abfe087.png',
            'West Ham':'7c21e445.png','Wolves':'8cec06e1.png'}

url_dict = {'Arsenal':'www.arsenal.com','Aston Villa':'avfc.co.uk','Bournemouth':'afcb.co.uk',
            'Brighton':'brightonandhovealbion.com','Burnley':'burnleyfootballclub.com','Chelsea':'chelseafc.com',
            'Crystal Palace':'cpfc.co.uk','Everton':'evertonfc.com','Leicester City':'lcfc.com',
            'Liverpool':'liverpoolfc.com','Manchester City':'mancity.com','Manchester United':'manutd.com',
            'Newcastle':'nufc.co.uk','Norwich City':'canaries.co.uk','Sheffield United':'sufc.co.uk',
            'Southampton':'southamptonfc.com','Tottenham':'tottenhamhotspur.com','Watford':'watfordfc.com',
            'West Ham':'whufc.com','Wolves':'wolves.co.uk'}

@app.route('/', methods = ['GET','POST'])

def index():
    # print(flask.request.method)
    # team = '''
    #     <form action="/predict" method="POST">
    #     <input type='submit' value='team'>
    #     </form>
    #     '''
    return flask.render_template('index.html')
    
    
@app.route('/predict', methods = ['GET','POST'])
def predict ():
    epl_df = unpickler('/home/josh/Documents/dsi/caps/cap3/Football_Supporter_Recommender/data/pickles/epl_domestic_league_df_clean.pickle')
    epl_mat = unpickler('/home/josh/Documents/dsi/caps/cap3/Football_Supporter_Recommender/data/pickles/epl_dists/epl_vectorized.pickle')
    nfl_df = unpickler('/home/josh/Documents/dsi/caps/cap3/Football_Supporter_Recommender/data/pickles/NFL_to_vector.pickle')
    # team = flask.request.form.value()
    # print(flask.request.get_json())
    # if flask.request.method == 'POST':
    #     print(flask.request.form)
    # team = 'Cardinals'
    # print(team)
    # return team
    team = flask.request.form['team']
    # print(team)
    # team = str(flask.request.form[team])
    # print(team)
    team_vec = SimilarityDF(nfl_df).vectorizer([team],'NFL')
    selected = Distances(team_vec,epl_mat)
    team_euc_top = selected.top_dists(distance_calc='euclidean',index_=epl_df.squad.unique(),col='euc_dist',number=3)
    team_cos_dist_top = selected.top_dists(distance_calc='cosine_dist',index_=epl_df.squad.unique(),col='cos_dist',number=3)
    # team_cos_sim_top = selected.top_dists(distance_calc='cosine_sim',index_=epl_df.squad.unique(),col='cos_sim',number=3)[::-1]
    # team_jac_top = selected.top_dists(distance_calc='jaccard',index_=epl_df.squad.unique(),col='jac_dist',number=3)
    # return flask.render_template('index_album.html')
    # print(team_euc_top)
    team1 = team_euc_top[0]
    team2 = team_euc_top[1]
    team3 = team_euc_top[2]
    team1_img = img_dict[team_euc_top[0]]
    team2_img = img_dict[team_euc_top[1]]
    team3_img = img_dict[team_euc_top[2]]
    return flask.render_template('index_album.html', team1=team1_img, team1_img=team1_img, team2=team2, team2_img=team2_img,
                                 team3=team3, team3_img=team3_img)
    
    
    
    # tables=[team_euc_top[0,0].to_html(classes='Top Recommendation'),
    # team_euc_top[1:4,1:4].to_html(classes='Other Recommendation')],
    # titles = ['You Should Support', 'You Should Consider'])

@app.route('/info')
def info():
    team1 = flask.request.form['team1']
    team2 = flask.request.form['team2']
    team3 = flask.request.form['team3']
    if team1:
        team = team_euc_top[:1]
        url = url_dict[team]
        return flask.redirect(url)
    if team2:
        team = team_euc_top[:1]
        url = url_dict[team]
        return flask.redirect(url)
    if team3:
        team = team_euc_top[:1]
        url = url_dict[team]
        return flask.redirect(url)






if __name__ == '__main__':
    app.run(host='0.0.0.0', port=PORT, debug=True)