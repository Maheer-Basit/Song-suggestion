from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import os
import logging

# Setup Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("DataPipeline")

app = FastAPI(
    title="AI Playlist Cluster Recommender",
    version="2.0.0"
)

class PlaylistRequest(BaseModel):
    songs: List[str] = Field(
        ..., 
        min_length=3, 
        max_length=50, # Defensive limit to protect server memory
        description="List of 3 to 50 song titles to analyze."
    )

FEATURE_KEYS = ['danceability', 'energy', 'loudness', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']

DATASET_PATH = "dataset.csv"
if os.path.exists(DATASET_PATH):
    logger.info("Initializing Data Quality Pipeline...")
    
    raw_df = pd.read_csv(DATASET_PATH).dropna(subset=FEATURE_KEYS)
    
    df_pool = raw_df.drop_duplicates(subset=['track_name', 'artists']).copy()
    
    df_pool['track_name_clean'] = df_pool['track_name'].str.lower().str.strip()
    
    df_scaled = df_pool.copy()
    for col in FEATURE_KEYS:
        min_val = df_pool[col].min()
        max_val = df_pool[col].max()
        if max_val - min_val > 0:
            df_scaled[col] = (df_pool[col] - min_val) / (max_val - min_val)
            
    logger.info(f"Pipeline Online. Sanitized {len(df_pool)} unique tracks.")
else:
    logger.error("CRITICAL ERROR: dataset.csv not found!")
    df_pool = None
    df_scaled = None

@app.get("/health", tags=["Infrastructure"])
def health_check():
    return {
        "status": "HEALTHY" if df_pool is not None else "CRITICAL_DATA_MISSING",
        "total_catalog_tracks": len(df_pool) if df_pool is not None else 0
    }

@app.post("/recommend-clustered", tags=["Machine Learning Inference"])
def recommend_clustered_songs(payload: PlaylistRequest):
    if df_pool is None or df_scaled is None:
        raise HTTPException(status_code=500, detail="Database uninitialized.")

    user_songs = list(set([song.strip().lower() for song in payload.songs if song.strip()]))

    matched_scaled = df_scaled[df_scaled['track_name_clean'].isin(user_songs)].copy()

    if len(matched_scaled) < 3:
        raise HTTPException(
            status_code=422, 
            detail=f"Only matched {len(matched_scaled)} songs. Please ensure precise title spelling."
        )

    X_user = matched_scaled[FEATURE_KEYS].values
    n_clusters = min(3, len(X_user))
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    matched_scaled['cluster_label'] = kmeans.fit_predict(X_user)

    selected_cluster = int(matched_scaled.groupby('cluster_label')['danceability'].mean().idxmax())
    cluster_tracks = matched_scaled[matched_scaled['cluster_label'] == selected_cluster]
    cluster_center = cluster_tracks[FEATURE_KEYS].mean().values

    pool_features_scaled = df_scaled[FEATURE_KEYS].values
    distances = np.linalg.norm(pool_features_scaled - cluster_center, axis=1)
    
    recommendation_pool = df_pool[~df_pool['track_name_clean'].isin(user_songs)].copy()
    recommendation_pool['distance'] = distances[~df_pool['track_name_clean'].isin(user_songs)]
    
    top_10 = recommendation_pool.sort_values(by='distance').head(10)

    recommendations = []
    for _, row in top_10.iterrows():
        similarity_index = round(float(max(0, 100 - (row['distance'] * 100))), 2)
        recommendations.append({
            "track_name": row['track_name'],
            "artist": row['artists'],
            "album": row['album_name'],
            "tempo_bpm": round(float(row['tempo']), 1),
            "similarity_index": similarity_index
        })

    return {
        "status": "SUCCESS",
        "target_cluster_id": selected_cluster,
        "cluster_center_normalized": dict(zip(FEATURE_KEYS, np.round(cluster_center, 3))),
        "recommendations": recommendations
    }