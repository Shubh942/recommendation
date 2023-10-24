import pickle as pkl
import json
import pandas as pd
import difflib
from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

movies_data = pd.read_pickle('data.pkl') 
similarity = pkl.load(open('similarity1.pkl','rb'))

@app.route('/predict', methods=['POST'])
def predict():
    keyword = request.get_json()['keyword']
    list_of_all_keywords = movies_data['keywords'].tolist()
    close_matches=[]
    for keywords in list_of_all_keywords:
      for key in keywords:
        if(key==keyword):
          close_matches.append(keywords)
          break
    close_match = close_matches[0]
    index_of_the_movie=-1
    for i in range(len(movies_data['keywords'])):
      if(movies_data['keywords'][i] == close_match):
        index_of_the_movie=i

    similarity_score = list(enumerate(similarity[index_of_the_movie]))

    sorted_similar_movies = sorted(similarity_score, key=lambda x: x[1], reverse=True)
    recommended_movies = []
    for movie in sorted_similar_movies:
        index = movie[0]
        title_from_index = movies_data[movies_data.index == index]['title'].values[0]
        recommended_movies.append(title_from_index)
    
    return jsonify({'movies': recommended_movies[:100]})

if __name__ == "__main__":
    app.run()
