# Song Suggestion

An API-driven music recommendation project that uses audio features from a Spotify-style dataset to cluster a user playlist and return songs with similar sonic profiles.

This project includes:

- A FastAPI service for health checks and recommendations
- K-Means clustering over track audio features
- Dataset cleaning and normalization for inference
- A separate evaluation script for checking recommendation quality
- Pytest coverage and GitHub Actions CI
- Docker support for running the app in a container

## Why this project is useful

The system demonstrates a full machine learning workflow rather than only a notebook demo. It loads a catalog, cleans the data, finds matching tracks from a user playlist, clusters them, and recommends other tracks close to the chosen cluster center.

## Tech Stack

- Python 3.11
- FastAPI
- Pandas
- NumPy
- scikit-learn
- Pytest
- Uvicorn
- Docker

## How it works

1. The app loads `dataset.csv` and removes incomplete or duplicate rows.
2. Audio features are scaled to a 0 to 1 range for distance calculations.
3. The user submits 3 to 50 song titles through the `/recommend-clustered` endpoint.
4. Matching tracks are clustered with K-Means.
5. The most cohesive cluster is selected and used as the recommendation anchor.
6. The app returns the closest 10 tracks from the catalog, excluding the user’s own songs.

## API Endpoints

### `GET /health`

Returns a simple system status response and the number of tracks loaded into the catalog.

Example response:

```json
{
	"status": "HEALTHY",
	"total_catalog_tracks": 114000
}
```

### `POST /recommend-clustered`

Request body:

```json
{
	"songs": ["Toxic", "Poker Face", "Buttons", "Hot N Cold"]
}
```

Example response structure:

```json
{
	"status": "SUCCESS",
	"target_cluster_id": 1,
	"cluster_center_normalized": {
		"danceability": 0.82,
		"energy": 0.77,
		"loudness": 0.44
	},
	"recommendations": [
		{
			"track_name": "Just Dance",
			"artist": "Lady Gaga",
			"album": "The Fame",
			"tempo_bpm": 119.0,
			"similarity_index": 93.42
		}
	]
}
```

## Local Setup

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## Run the API

```bash
uvicorn app:app --reload
```

Then open:

- `http://127.0.0.1:8000/health`
- `http://127.0.0.1:8000/docs`

## Run Tests

```bash
pytest -q
```

## Evaluate the Pipeline

```bash
python evaluate.py
```

The evaluation script compares the recommendation set against a simple random baseline and prints a silhouette score for the clustered sample.

## Docker

```bash
docker build -t song-suggestion .
docker run -p 8000:8000 song-suggestion
```

## Notes

- The app expects `dataset.csv` to be present in the project root.
- Song title matching is case-insensitive, but spelling still needs to be reasonably precise.
- The current approach is designed for explainability and simplicity rather than state-of-the-art ranking performance.

## Project Status

This is a solid second-year portfolio project because it shows real engineering range: data cleaning, ML inference, API design, containerization, automated testing, and CI.

## Future Improvements

- Add a small frontend or Streamlit demo
- Expose artist and genre filters
- Persist the model or precomputed clusters instead of fitting on request
- Add richer evaluation metrics and sample visualizations
