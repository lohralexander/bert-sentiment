import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import sys
import os
from pathlib import Path

# Add the parent directory to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import the main module
import app as main

# Create a test client
client = TestClient(main.app)


# Mock pytest-specific fixture
@pytest.fixture(autouse=True)
def setup_and_teardown():
    """Setup test environment before each test and clean up after."""
    # Setup: mock all required components
    main.tokenizer = MagicMock()
    main.model = MagicMock()
    main.sentiment_pipeline = MagicMock()
    main.sentiment_pipeline.return_value = [{"label": "POSITIVE", "score": 0.9}]

    yield

    # Teardown (optional, as pytest will handle this)
    pass


# Test the health check endpoint
def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


# Test the health check failure
def test_health_check_failure():
    # Temporarily set components to None
    original_tokenizer = main.tokenizer
    original_model = main.model
    original_pipeline = main.sentiment_pipeline

    main.tokenizer = None
    main.model = None
    main.sentiment_pipeline = None

    try:
        response = client.get("/health")
        assert response.status_code == 503
        assert response.json()["detail"] == "Model not loaded"
    finally:
        # Restore components
        main.tokenizer = original_tokenizer
        main.model = original_model
        main.sentiment_pipeline = original_pipeline


# Test the root endpoint
def test_root():
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert "usage" in data
    assert data["message"] == "Sentiment Analysis API"
    assert "POST /predict" in data["usage"]


# Test the predict endpoint with standard request
def test_predict_sentiment():
    test_text = "I love this product!"
    response = client.post(
        "/predict",
        json={"text": test_text, "return_all_scores": False}
    )
    assert response.status_code == 200
    result = response.json()
    assert result["text"] == test_text
    assert result["label"] == "POSITIVE"
    assert result["score"] == 0.9
    assert result["all_scores"] is None


# Test predict with return_all_scores=True
def test_predict_sentiment_with_all_scores():
    test_text = "I love this product!"

    # Configure mock for return_all_scores=True
    with patch.object(main, 'sentiment_pipeline') as mock_pipeline:
        mock_pipeline.return_value = [[
            {"label": "POSITIVE", "score": 0.9},
            {"label": "NEGATIVE", "score": 0.1}
        ]]

        response = client.post(
            "/predict",
            json={"text": test_text, "return_all_scores": True}
        )

        assert response.status_code == 200
        result = response.json()
        assert result["text"] == test_text
        assert result["label"] == "POSITIVE"
        assert result["score"] == 0.9
        assert len(result["all_scores"]) == 2
        assert result["all_scores"][0]["label"] == "POSITIVE"
        assert result["all_scores"][1]["label"] == "NEGATIVE"


# Test error handling when model not loaded
def test_predict_sentiment_model_not_loaded():
    # Temporarily set sentiment_pipeline to None
    original = main.sentiment_pipeline
    main.sentiment_pipeline = None

    try:
        test_text = "I love this product!"
        response = client.post(
            "/predict",
            json={"text": test_text}
        )
        assert response.status_code == 500
        assert "detail" in response.json()
        assert "Prediction error" in response.json()["detail"]
    finally:
        # Restore sentiment_pipeline
        main.sentiment_pipeline = original


# Test predict_with_pipeline function directly
def test_predict_with_pipeline_function():
    with patch.object(main, 'sentiment_pipeline') as mock_pipeline:
        mock_pipeline.return_value = [{"label": "POSITIVE", "score": 0.9}]

        result = main.predict_with_pipeline("I love this product!")

        assert result["text"] == "I love this product!"
        assert result["label"] == "POSITIVE"
        assert result["score"] == 0.9
        assert result["all_scores"] is None

        # Verify the mock was called correctly
        mock_pipeline.assert_called_once_with("I love this product!", return_all_scores=False)


# Test predict_with_pipeline with all_scores
def test_predict_with_pipeline_all_scores():
    with patch.object(main, 'sentiment_pipeline') as mock_pipeline:
        mock_pipeline.return_value = [[
            {"label": "POSITIVE", "score": 0.9},
            {"label": "NEGATIVE", "score": 0.1}
        ]]

        result = main.predict_with_pipeline("I love this product!", return_all_scores=True)

        assert result["text"] == "I love this product!"
        assert result["label"] == "POSITIVE"
        assert result["score"] == 0.9
        assert len(result["all_scores"]) == 2

        # Verify the mock was called correctly
        mock_pipeline.assert_called_once_with("I love this product!", return_all_scores=True)


# Test error handling in predict_with_pipeline
def test_predict_with_pipeline_error():
    with patch.object(main, 'sentiment_pipeline') as mock_pipeline:
        mock_pipeline.side_effect = Exception("Test error")

        with pytest.raises(Exception) as excinfo:
            main.predict_with_pipeline("I love this product!")

        assert "Prediction error" in str(excinfo.value)


# Test empty input text
def test_predict_empty_text():
    response = client.post(
        "/predict",
        json={"text": "", "return_all_scores": False}
    )
    assert response.status_code == 200  # The API doesn't handle empty text specially


# Test with very long input
def test_predict_long_text():
    long_text = "This is a very long text. " * 100  # 2600 characters

    with patch.object(main, 'sentiment_pipeline') as mock_pipeline:
        mock_pipeline.return_value = [{"label": "NEUTRAL", "score": 0.6}]

        response = client.post(
            "/predict",
            json={"text": long_text, "return_all_scores": False}
        )

        assert response.status_code == 200
        result = response.json()
        assert result["text"] == long_text
        assert result["label"] == "NEUTRAL"
        assert result["score"] == 0.6