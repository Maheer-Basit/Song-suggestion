import pytest
from fastapi.testclient import TestClient
from app import app, df_pool

# Initialize the FastAPI TestClient framework
client = TestClient(app)

def test_health_endpoint():
    """
    ARRANGE & ACT: Query the infrastructure system probe route
    """
    response = client.get("/health")
    
    """
    ASSERT: Check that the server responds with a perfect HTTP 200 OK 
    and verifies the catalog metrics are populated.
    """
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert data["status"] == "HEALTHY"
    assert "total_catalog_tracks" in data
    assert data["total_catalog_tracks"] > 0


def test_successful_recommendation():
    """
    ARRANGE: Construct a valid pop/club track playlist payload
    """
    payload = {
        "songs": ["Toxic", "Poker Face", "Buttons", "Hot N Cold"]
    }
    
    """
    ACT: Trigger the machine learning cluster vector calculations
    """
    response = client.post("/recommend-clustered", json=payload)
    
    """
    ASSERT: Verify the pipeline creates successful clusters and scores top 10 tracks
    """
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "SUCCESS"
    assert "target_cluster_id" in data
    assert "recommendations" in data
    assert len(data["recommendations"]) == 10
    
    # Check that individual recommendations contain required catalog parameters
    first_rec = data["recommendations"][0]
    assert "track_name" in first_rec
    assert "artist" in first_rec
    assert "similarity_index" in first_rec


def test_invalid_payload_too_few_songs():
    """
    ARRANGE: Build a bad payload with only 1 song (violating min_items=3 rule)
    """
    payload = {"songs": ["Toxic"]}
    
    """
    ACT & ASSERT: Verify our Pydantic data layer catches this and returns a 422 Unprocessable Content error
    """
    response = client.post("/recommend-clustered", json=payload)
    assert response.status_code == 422


def test_unknown_songs_error():
    """
    ARRANGE: Pass song titles that absolutely do not exist in the Kaggle dataset pool
    """
    payload = {
        "songs": ["Fake Song Name A", "Fake Song Name B", "Fake Song Name C"]
    }
    
    """
    ACT & ASSERT: Check that the system returns a defensive validation error instead of crashing
    """
    response = client.post("/recommend-clustered", json=payload)
    assert response.status_code == 422
    assert "Please ensure precise title spelling" in response.json()["detail"]