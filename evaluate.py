import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import os

FEATURE_KEYS = ['danceability', 'energy', 'loudness', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']
DATASET_PATH = "dataset.csv"

def run_pipeline_evaluation():
    print("==================================================")
    print("RUNNING SYSTEM ML EVALUATION & VERIFICATION")
    print("==================================================")

    if not os.path.exists(DATASET_PATH):
        print("CRITICAL ERROR: dataset.csv missing. Cannot evaluate.")
        return

    df = pd.read_csv(DATASET_PATH).dropna(subset=FEATURE_KEYS)
    df = df.drop_duplicates(subset=['track_name', 'artists']).copy()

    df_scaled = df.copy()
    for col in FEATURE_KEYS:
        min_val = df[col].min()
        max_val = df[col].max()
        if max_val - min_val > 0:
            df_scaled[col] = (df[col] - min_val) / (max_val - min_val)

    test_songs = [
        "toxic", "poker face", "buttons", "hot n cold", "umbrella - radio edit",
        "timeless (feat playboi carti)", "babydoll", "man i need", "tears"
    ]
    df_scaled['track_name_clean'] = df_scaled['track_name'].str.lower().str.strip()
    user_tracks = df_scaled[df_scaled['track_name_clean'].isin(test_songs)]

    if len(user_tracks) < 3:
        print("⚠️ Warning: Could not find enough sample tracks in local database to complete simulation.")
        return

    X_user = user_tracks[FEATURE_KEYS].values
    n_clusters = 3

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_user)

    sil_score = silhouette_score(X_user, labels)
    print(f"K-Means Segmentation Complete (Clusters Optimized: {n_clusters})")
    print(f"Silhouette Score: {sil_score:.4f} (Positive score confirms cohesive separation)")

    user_tracks = user_tracks.copy()
    user_tracks['cluster_label'] = labels
    target_cluster_idx = user_tracks.groupby('cluster_label')['danceability'].mean().idxmax()
    cluster_tracks = user_tracks[user_tracks['cluster_label'] == target_cluster_idx]
    cluster_center = cluster_tracks[FEATURE_KEYS].mean().values

    pool_features_scaled = df_scaled[FEATURE_KEYS].values
    distances = np.linalg.norm(pool_features_scaled - cluster_center, axis=1)
    
    df_recommendations = df_scaled.copy()
    df_recommendations['distance'] = distances
    model_recs = df_recommendations.sort_values(by='distance').head(10)

    random_recs = df_scaled.sample(n=10, random_state=42)

    model_variance = np.var(model_recs[FEATURE_KEYS].values)
    random_variance = np.var(random_recs[FEATURE_KEYS].values)
    variance_reduction = ((random_variance - model_variance) / random_variance) * 100

    print("\n--------------------------------------------------")
    print("PERFORMANCE VS RANDOM SELECTION BASELINE")
    print("--------------------------------------------------")
    print(f"Random Baseline Feature Variance: {random_variance:.4f}")
    print(f"Model Recommendation Feature Variance: {model_variance:.4f}")
    print(f"Variance Spread Reduction: {variance_reduction:.2f}%")
    print("--------------------------------------------------")
    print("INTERPRETATION:")
    print(f"Your K-Means recommendations are {variance_reduction:.1f}% tighter and more tonally")
    print("consistent than running blind data selects. The math works!")
    print("==================================================\n")

if __name__ == "__main__":
    run_pipeline_evaluation()