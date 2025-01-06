from flask import Flask, render_template, request
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# flask app
app = Flask(__name__)

# Load your dataset
df = pd.read_csv('clustered_df.csv')

numerical_features = [
    "valence", "danceability", "energy", "tempo",
    "acousticness", "liveness", "speechiness", "instrumentalness"
]

def recommend_songs(song_name, df, num_recommendations=5):
    # Get the cluster of the input song
    song_cluster = df[df["name"] == song_name]["Cluster"].values[0]
    same_cluster_songs = df[df["Cluster"] == song_cluster]
    song_index = same_cluster_songs[same_cluster_songs["name"] == song_name].index[0]
    cluster_features = same_cluster_songs[numerical_features]
    similarity = cosine_similarity(cluster_features, cluster_features)
    similar_songs = np.argsort(similarity[song_index])[-(num_recommendations + 1):-1][::-1]
    recommendations = same_cluster_songs.iloc[similar_songs][["name", "year", "artists"]]
    return recommendations

# route
@app.route("/")
def index():
    return render_template('index.html')

@app.route("/recommend", methods=["GET", "POST"])
def recommend():
    recommendations = []
    if request.method == "POST":
        song_name = request.form.get("song_name")
        try:
            recommendations = recommend_songs(song_name, df).to_dict(orient="records")
        except Exception as e:
            recommendations = [{"name": "Error", "artists": "Invalid song name", "year": ""}]
    return render_template("index.html", recommendations=recommendations)

if __name__ == "__main__":
    app.run(debug=True)
