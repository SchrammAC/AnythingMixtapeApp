import requests
import os
import pandas as pd
from flask import Flask, render_template, request, redirect

#from bokeh.plotting import figure
#from bokeh.embed import components
from spotipy.oauth2 import SpotifyClientCredentials
import spotipy
import dill
import spotipy.util as util
import spotipy.oauth2 as oauth2
import configparser
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.neighbors import NearestNeighbors

app = Flask(__name__)



def save_pkl(df, filename):
    with open('static/data/'+filename+'.pkl','wb') as fobj:
        dill.dump(df,fobj)
    
def load_pkl(filename):
    with open('static/data/'+filename+'.pkl','rb') as fobj:
        df = dill.load(fobj)
    return df

def load_credentials():
    config = configparser.ConfigParser()
    config.read('static/keys/tam_creds.nogit')
    client_id = config.get('SPOTIFY', 'CLIENT_ID')
    client_secret = config.get('SPOTIFY', 'CLIENT_SECRET')
    
#    auth = oauth2.SpotifyClientCredentials(
#        client_id=client_id,
#        client_secret=client_secret
#    )
    return client_id, client_secret

def get_token(username,client_id,client_secret):
    scope = 'playlist-modify-public'
    token = util.prompt_for_user_token(username,scope,client_id=client_id,client_secret=client_secret,redirect_uri='theanythingmixtape://returnafterlogin')
    return token

def create_spotify_playlist(token,username,playlist_name,playlist_description,playlist_ids):
    if token:
        sp = spotipy.Spotify(auth=token)
        sp.trace = False
        playlist = sp.user_playlist_create(username, playlist_name, public=True, description=playlist_description)
#        pprint.pprint(playlist)
        _ = sp.user_playlist_add_tracks(username, playlist['id'], playlist_ids)
#        print(results)
        return playlist['id']
    else:
        print("Can't get token for", username)
    
def spotify_from_trackids(username, playlist_name, playlist_description, playlist_ids):    
    client_id, client_secret = load_credentials()
    token = get_token(username,client_id,client_secret)
    playlist_id = create_spotify_playlist(token,username,playlist_name,playlist_description,playlist_ids)
    return playlist_id
 
def classify_text(model, text):    
    print(text)
    print(model.predict_proba(text))
    best_genre = model.predict(text)
    print(best_genre)
    return best_genre

def create_playlist(genre, data, input_text, vocab, num_tracks=10):
    genre_data = data.loc[data['genre']==genre[0]] #Select genre tracks
    tfidf_matrix = vectorize_it(genre_data, input_text, vocab)
    indices = get_closest_indices(tfidf_matrix,num_tracks)
    track_list = fetch_track_ids(genre_data,indices)
    return track_list
    
    
    
def vectorize_it(genre_data, input_text, vocab):
    input_df = pd.DataFrame(data=[['', input_text, '', '']], columns=['genre', 'lyrics', 'orig_index', 'track_id'])
    genre_data = genre_data.append(input_df, ignore_index=True)
    
    ctvect = CountVectorizer(vocabulary=vocab)
    tfidf_trans = TfidfTransformer()
    
    text_vect = ctvect.fit_transform(genre_data.lyrics)
    
    return tfidf_trans.fit_transform(text_vect)

    
def get_closest_indices(tfidf_matrix,num_tracks):
    nbrs = NearestNeighbors(n_neighbors=num_tracks+1).fit(tfidf_matrix)
    distances, indices = nbrs.kneighbors(tfidf_matrix[-1,:])
    indices = indices.flatten()[1:]
    return indices

def fetch_track_ids(genre_data,indices):
    similar_tracks = pd.Series(indices).map(genre_data.reset_index()['track_id'])
    return similar_tracks


def genre_prediction(input_text, num_songs):
    data = load_pkl('labeled_df')
    genre_clf = load_pkl('cnb_classifier')
    
    total_vocab = genre_clf.named_steps['ctvect'].vocabulary_
    
    best_genre = classify_text(genre_clf, [input_text])
    
    track_id_list = create_playlist(best_genre, data, input_text, total_vocab, num_songs)
    
    return track_id_list

def handle_spotify(playlist_vars):
        track_ids = genre_prediction(playlist_vars['input_text'], 10)
        playlist_id = spotify_from_trackids(playlist_vars['username'], playlist_vars['playlist_name'], 'Your Mixtape' , track_ids)
        return playlist_id



@app.route('/')
def homepage():
    return redirect('/home')

@app.route('/home', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/themodel')
def themodel():
  return render_template('themodel.html')
    
@app.route('/themixtape', methods=['GET','POST'])
def themixtape():
    playlist_vars = {}
    if request.method == 'GET':
        return render_template('themixtape.html')
    else:
        # request was a POST
        playlist_vars['username'] = request.form['username']
        playlist_vars['input_text'] = request.form['text_description']
        playlist_vars['playlist_name'] = request.form['playlist_name']
        
        playlist_id = handle_spotify(playlist_vars)
       
        return render_template("yourmixtape.html", playlist_id=playlist_id,
                       name = playlist_vars['playlist_name'])
    
@app.route('/yourmixtape', methods=['GET','POST'])
def yourmixtape():
    if request.method == 'GET':
        return render_template('yourmixtape.html')
    else:
        # request was a POST
#        playlist_vars['symbol'] = request.form['stocksym']
       
        return redirect('/home')
    



#@app.route('/stock', methods=['GET','POST'])
#def stockplot():
#    
#    api_url = 'http://www.quandl.com/api/v1/datasets/WIKI/%s.json?api_key=1gRwKdFTfvXCUXfxwa_n' % (app.vars['symbol'])
#    
#    session = requests.Session()
#    session.mount('http://', requests.adapters.HTTPAdapter(max_retries=3))
#    raw_data = session.get(api_url)
#    json_data=raw_data.json()
#    stock_data = pd.DataFrame(data=json_data.get("data"), columns=json_data.get("column_names"))
#    
#    stock_data['Date'] =  pd.to_datetime(stock_data['Date'], yearfirst=True)
#
#    p=figure(plot_width=400, plot_height=250, x_axis_type="datetime")
#    p.xaxis.axis_label = 'Date'
#    p.yaxis.axis_label = 'Closing Price'
#    p.line(stock_data.Date,stock_data.Close)
#    p.toolbar.logo = None
#    
#    script,div=components(p)
#    plots=[]
#    plots.append([script,div])
#
#    return render_template("chart.html", ticker=app.vars['symbol'],
#                       the_div=div, the_script=script)


if __name__ == "__main__":
#    port = int(os.environ.get("PORT", 33507))
#    app.run(host='0.0.0.0', port=port)
    app.run(debug=False)
