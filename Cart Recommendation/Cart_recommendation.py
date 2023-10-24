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

import json
for i in range(len(movies_data['keywords'])):
  movies_data['keywords'][i]=json.loads(movies_data['keywords'][i])
  movies_data['keywords'][i] = [d['name'] for d in movies_data['keywords'][i]]

@app.route('/predict', methods=['POST'])
def predict():
    keyword = request.get_json()['keyword']
    list_of_all_keywords = movies_data['keywords'].tolist()
    find_close_match=[]
    for idx, keywords in enumerate(list_of_all_keywords):
        close_matches = difflib.get_close_matches(keyword, list_of_all_keywords)
        if close_matches:
            for match in close_matches:
                find_close_match.append(keywords)
    close_match = find_close_match[0]
    index_of_the_movie=0
    for i in range(len(movies_data['keywords'])):
      if(movies_data['keywords'][i] == close_match):
        index_of_the_movie=i

    similarity_score = list(enumerate(similarity[index_of_the_movie]))

    sorted_similar_movies = sorted(similarity_score, key=lambda x: x[1], reverse=True)

    print('Movies suggested for you: \n')

    i = 1
    recommended_movies = []
    for movie in sorted_similar_movies:
        index = movie[0]
        title_from_index = movies_data[movies_data.index == index]['title'].values[0]
        if i < 30:
            recommended_movies.append(title_from_index)
    
    return jsonify({'movies': recommended_movies})

if __name__ == "__main__":
    app.run()
