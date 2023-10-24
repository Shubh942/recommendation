import pickle as pkl
import json
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
import random
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

new_df = pd.read_pickle('movies.pkl') 
similarity = pkl.load(open('similarity.pkl','rb'))

@app.route('/predict', methods=['POST'])
def predict():
    movie = request.get_json()['movie']
    movie_index = -1
    print(new_df[new_df['title'] == movie])
    if(new_df[new_df['title'] == movie].isnull().values.any() == True):
        movie_index = new_df[new_df['title'] == movie].index[0]
    else:
        movie_index=random.randint(1, 1000)
    distances = similarity[movie_index]
    movie_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]
    recommended_movies = [new_df.iloc[i[0]].title for i in movie_list]
    
    return jsonify({'movies': recommended_movies})

if __name__ == "__main__":
    app.run()
