import pickle as pkl
import json
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

movies_cleaned_df = pd.read_pickle('movies_cleaned_df.pkl')

from sklearn.metrics.pairwise import sigmoid_kernel

# Compute the sigmoid kernel
sig = sigmoid_kernel(tfv_matrix, tfv_matrix)
# Reverse mapping of indices and movie titles
indices = pd.Series(movies_cleaned_df.index, index=movies_cleaned_df['original_title']).drop_duplicates()
indices
new_df = pd.read_pickle('movies.pkl') 
similarity = pkl.load(open('similarity.pkl','rb'))

@app.route('/predict', methods=['POST'])
def predict():
    movie = request.get_json()['movie']
    movie_index = new_df[new_df['title'] == movie].index[0]
    distances = similarity[movie_index]
    movie_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]
    recommended_movies = [new_df.iloc[i[0]].title for i in movie_list]
    
    return jsonify({'movies': recommended_movies})

if __name__ == "__main__":
    app.run()
