from fastapi import FastAPI
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import os
import random

app = FastAPI(title="AI Playlist Cluster Recommender")

if os.path.exists("dataset.csv"):
    df_pool = pd.read_csv("dataset.csv").dropna(subset=['danceability', 'energy', 'tempo'])
    df_pool['track_name_clean'] = df_pool['track_name'].str.lower().str.strip()
else:
    df_pool = None

@app.post("/recommend-clustered")
def recommend_clustered_songs():
    if df_pool is None:
        return {"status": "ERROR", "message": "dataset.csv not found in folder!"}
    if not os.path.exists("my_playlist.txt"):
        return {"status": "ERROR", "message": "my_playlist.txt not found in folder!"}

    with open("my_playlist.txt", "r") as f:
        user_songs = [line.strip().lower() for line in f if line.strip()]

    matched_tracks = df_pool[df_pool['track_name_clean'].isin(user_songs)].copy()

    if len(matched_tracks) < 3:
        return {
            "status": "ERROR", 
            "message": f"Found only {len(matched_tracks)} matching songs. We need at least 3 to build clusters!",
            "hint": "Check the exact spelling of titles in your text file."
        }

    features_keys = ['danceability', 'energy', 'loudness', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']
    
    X_user = matched_tracks[features_keys].values
    
    n_clusters = min(3, len(X_user))
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    matched_tracks['cluster_label'] = kmeans.fit_predict(X_user)

    selected_cluster = random.randint(0, n_clusters - 1)
    cluster_tracks = matched_tracks[matched_tracks['cluster_label'] == selected_cluster]
    
    cluster_center = cluster_tracks[features_keys].mean().values

    pool_features = df_pool[features_keys].values
    distances = np.linalg.norm(pool_features - cluster_center, axis=1)
    df_pool['distance'] = distances
    
    filtered_pool = df_pool[~df_pool['track_name_clean'].isin(user_songs)]
    top_10_recommendations = filtered_pool.sort_values(by='distance').head(10)

    recommendations = []
    for _, row in top_10_recommendations.iterrows():
        recommendations.append({
            "track_name": row['track_name'],
            "artist": row['artists'],
            "album": row['album_name'],
            "match_confidence": round(float(100 - (row['distance'] * 10)), 2)
        })

    return {
        "status": "SUCCESS",
        "total_playlist_songs_mapped": len(matched_tracks),
        "vibe_sections_detected": n_clusters,
        "section_selected_for_recommendations": selected_cluster,
        "sample_songs_from_this_vibe": list(cluster_tracks['track_name'].head(3)),
        "targeted_taste_averages": dict(zip(features_keys, np.round(cluster_center, 3))),
        "top_10_recommended_tracks": recommendations
    }