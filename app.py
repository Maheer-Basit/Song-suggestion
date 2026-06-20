from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import os
import random
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("MLOpsPipeline")

app = FastAPI(
    title="AI Playlist Cluster Recommender",
    description="Production-grade unsupervised ML music recommendation engine."
)

class PlaylistRequest(BaseModel):
    songs: List[str] = Field(
        ..., 
        min_items=3, 
        description="List of at least 3 song titles to build the taste profile.",
        example=["Thunderstruck", "Stronger", "Till I Collapse"]
    )

@app.get("/health", tags=["Infrastructure"])
def health_check():
    dataset_exists = os.path.exists("dataset.csv")
    return {
        "status": "HEALTHY" if dataset_exists else "DEGRADED",
        "database_connected": dataset_exists,
        "engine": "FastAPI + Scikit-Learn"
    }

DATASET_PATH = "dataset.csv"
if os.path.exists(DATASET_PATH):
    logger.info("Loading Kaggle database universe into memory...")
    # Read only what we need to optimize performance and memory footprint
    df_pool = pd.read_csv(DATASET_PATH).dropna(subset=['danceability', 'energy', 'tempo'])
    df_pool['track_name_clean'] = df_pool['track_name'].str.lower().str.strip()
    logger.info(f"Database successfully mapped: {len(df_pool)} tracks loaded.")
else:
    logger.error(f"CRITICAL: {DATASET_PATH} missing from project root folder!")
    df_pool = None

@app.post("/recommend-clustered", tags=["Machine Learning Inference"])
def recommend_clustered_songs(payload: PlaylistRequest):
    if df_pool is None:
        raise HTTPException(status_code=500, detail="Database file 'dataset.csv' is missing on server.")

    user_songs = [song.strip().lower() for song in payload.songs if song.strip()]

    matched_tracks = df_pool[df_pool['track_name_clean'].isin(user_songs)].copy()

    if len(matched_tracks) < 3:
        raise HTTPException(
            status_code=422, 
            detail=f"Only matched {len(matched_tracks)} out of your provided tracks. Need at least 3 recognized songs to calculate vibe clusters."
        )

    features_keys = ['danceability', 'energy', 'loudness', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']
    
    X_user = matched_tracks[features_keys].values
    n_clusters = min(3, len(X_user))
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    matched_tracks['cluster_label'] = kmeans.fit_predict(X_user)

    selected_cluster = int(matched_tracks.groupby('cluster_label')['energy'].mean().idxmax())
    cluster_tracks = matched_tracks[matched_tracks['cluster_label'] == selected_cluster]
    cluster_center = cluster_tracks[features_keys].mean().values

    pool_features = df_pool[features_keys].values
    distances = np.linalg.norm(pool_features - cluster_center, axis=1)
    
    recommendation_pool = df_pool[~df_pool['track_name_clean'].isin(user_songs)].copy()
    recommendation_pool['distance'] = distances[~df_pool['track_name_clean'].isin(user_songs)]
    
    top_10_recommendations = recommendation_pool.sort_values(by='distance').head(10)

    recommendations = []
    for _, row in top_10_recommendations.iterrows():
        similarity_index = round(float(max(0, 100 - (row['distance'] * 8))), 2)
        recommendations.append({
            "track_name": row['track_name'],
            "artist": row['artists'],
            "album": row['album_name'],
            "similarity_index": similarity_index
        })

    logger.info(f"Successfully generated 10 recommendations for cluster profile {selected_cluster}.")
    return {
        "status": "SUCCESS",
        "songs_matched_count": len(matched_tracks),
        "target_cluster_index": selected_cluster,
        "sample_cluster_tracks": list(cluster_tracks['track_name'].head(3)),
        "cluster_center_vector": dict(zip(features_keys, np.round(cluster_center, 3))),
        "recommendations": recommendations
    }