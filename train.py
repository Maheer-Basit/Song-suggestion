import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import joblib

print(">>> Loading Spotify dataset (114,000+ tracks)...")
df = pd.read_csv("dataset.csv")

features = ['danceability', 'energy', 'loudness', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']
target = 'popularity'

df = df.dropna(subset=features + [target])

X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(">>> Training the Machine Learning model (this might take a minute)...")
model = RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

score = model.score(X_test, y_test)
print(f">>> Model trained successfully! R2 Accuracy Score: {score:.4f}")

joblib.dump(model, "spotify_model.pkl")
print(">>> Saved trained model to 'spotify_model.pkl'")