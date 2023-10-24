import pickle as pkl
import json
import pandas as pd
import random
from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

movies_cleaned_df = pd.read_pickle('movies_cleaned_df.pkl')
tfv_matrix = pkl.load(open('tfv_matrix.pkl','rb'))

from sklearn.metrics.pairwise import sigmoid_kernel

# Compute the sigmoid kernel
sig = sigmoid_kernel(tfv_matrix, tfv_matrix)

indices=pkl.load(open('indices.pkl','rb'))

@app.route('/predict', methods=['POST'])
def predict():
    title = request.get_json()['movie']
    for i in range(len(movies_cleaned_df)):
        if(movies_cleaned_df['original_title'][i] == title):
            idx = indices[title]
            break
        else:
            idx=random.randint(1, 1000)

    # Get the pairwsie similarity scores 
    sig_scores = list(enumerate(sig[idx]))

    # Sort the movies 
    sig_scores = sorted(sig_scores, key=lambda x: x[1], reverse=True)

    # Scores of the 10 most similar movies
    sig_scores = sig_scores[1:11]

    # Movie indices
    movie_indices = [i[0] for i in sig_scores]

    # Top 10 most similar movies
    recommended_movies= movies_cleaned_df['original_title'].iloc[movie_indices]
    
    return jsonify({'movies': recommended_movies.tolist()})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080)
