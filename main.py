import pytest
from fastapi.testclient import TestClient
from main import app
import os
from datetime import datetime

client = TestClient(app)

def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()
    assert "endpoints" in response.json()
    assert "documentation" in response.json()

def test_create_message():
    # Test creating a message without image
    response = client.post(
        "/messages",
        data={"content": "Hello, this is a test message", "model": "gpt-3.5-turbo"}
    )
    assert response.status_code == 200
    data = response.json()
    assert "id" in data
    assert data["content"] == "Hello, this is a test message"
    assert data["model"] == "gpt-3.5-turbo"
    assert "response" in data
    assert "created_at" in data
    assert data["status"] == "completed"

def test_create_message_with_image():
    # Create a dummy image file for testing
    with open("test_image.jpg", "wb") as f:
        f.write(b"dummy image content")
    
    try:
        with open("test_image.jpg", "rb") as f:
            response = client.post(
                "/messages",
                data={"content": "Describe this image", "model": "gpt-4-vision-preview"},
                files={"image": ("test_image.jpg", f, "image/jpeg")}
            )
        assert response.status_code == 200
        data = response.json()
        assert "id" in data
        assert data["content"] == "Describe this image"
        assert "response" in data
    finally:
        # Clean up the test image file
        if os.path.exists("test_image.jpg"):
            os.remove("test_image.jpg")

def test_list_jobs():
    # First create a job
    client.post(
        "/messages",
        data={"content": "Test job for listing", "model": "gpt-3.5-turbo"}
    )
    
    # Then test listing jobs
    response = client.get("/jobs")
    assert response.status_code == 200
    assert isinstance(response.json(), list)
    assert len(response.json()) > 0

def test_get_job_by_id():
    # First create a job
    create_response = client.post(
        "/messages",
        data={"content": "Test job for retrieval", "model": "gpt-3.5-turbo"}
    )
    job_id = create_response.json()["id"]
    
    # Then test retrieving the specific job
    response = client.get(f"/jobs/{job_id}")
    assert response.status_code == 200
    data = response.json()
    assert data["id"] == job_id
    assert data["content"] == "Test job for retrieval"

def test_get_nonexistent_job():
    response = client.get("/jobs/nonexistent-id")
    assert response.status_code == 404

def test_get_models():
    response = client.get("/models")
    assert response.status_code == 200
    models = response.json()
    assert isinstance(models, list)
    
    # Check if common models are present
    model_ids = [model["id"] for model in models]
    assert any("gpt-4" in model_id for model_id in model_ids)
    assert any("gpt-3.5-turbo" in model_id for model_id in model_ids)

def test_create_message_invalid_model():
    response = client.post(
        "/messages",
        data={"content": "Test message", "model": "invalid-model"}
    )
    assert response.status_code == 500  # Should fail due to invalid model 
