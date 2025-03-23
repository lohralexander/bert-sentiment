import logging
import time
from contextlib import asynccontextmanager
from typing import Dict, List, Any, Optional

from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel, Field, field_validator
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from transformers.pipelines import Pipeline
from typing_extensions import AsyncGenerator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Global variables for model components
model_name = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer: Optional[AutoTokenizer] = None
model: Optional[AutoModelForSequenceClassification] = None
sentiment_pipeline: Optional[Pipeline] = None


# Define lifespan context manager for startup/shutdown events
@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    # Startup: Load model and tokenizer
    global tokenizer, model, sentiment_pipeline
    try:
        logger.info("Loading model and tokenizer...")
        start_time = time.time()
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        sentiment_pipeline = pipeline(
            "sentiment-analysis",
            model=model,
            tokenizer=tokenizer,
            device=-1  # CPU, use device=0 for GPU
        )
        elapsed = time.time() - start_time
        logger.info(f"Model loaded successfully in {elapsed:.2f} seconds!")
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise RuntimeError(f"Failed to load model: {str(e)}")

    yield  # This is where the FastAPI app runs

    # Shutdown: Clean up resources
    logger.info("Shutting down and cleaning up resources...")
    tokenizer = None
    model = None
    sentiment_pipeline = None


# Initialize FastAPI app with lifespan
app = FastAPI(
    title="Sentiment Analysis API",
    description="API for sentiment analysis using BERT model from Hugging Face",
    version="1.0.0",
    lifespan=lifespan
)


# Define request model
class PredictRequest(BaseModel):
    text: str = Field(..., description="Text to analyze")
    return_all_scores: Optional[bool] = Field(False, description="Whether to return all scores or just the best one")

    @field_validator('text')
    @classmethod
    def text_must_not_be_empty(cls, v: str) -> str:
        if v is None:
            raise ValueError('Text cannot be None')

        if not isinstance(v, str):
            raise ValueError('Text must be a string')

        if not v.strip():
            raise ValueError('Text cannot be empty')

        return v.strip()  # Strip whitespace from input


# Define response model
class PredictResponse(BaseModel):
    label: str
    confidence: float


# Check if model is loaded
def ensure_model_loaded():
    if tokenizer is None or model is None or sentiment_pipeline is None:
        logger.error("Model components not loaded")
        raise HTTPException(status_code=503, detail="Model not loaded")
    return True


# Predict using the pipeline
def predict_with_pipeline(text: str) -> Dict[str, Any]:
    try:
        # Extra validation as a safeguard
        if not text or not isinstance(text, str):
            logger.error(f"Invalid input text: {text!r}")
            raise ValueError("Text must be a non-empty string")

        if not sentiment_pipeline:
            logger.error("Cannot predict, sentiment_pipeline is not loaded")
            raise RuntimeError("Model not loaded")

        logger.info(f"Processing text (length: {len(text)})")
        start_time = time.time()

        # Get prediction
        result = sentiment_pipeline(
            text
        )

        elapsed = time.time() - start_time
        logger.info(f"Prediction completed in {elapsed:.4f} seconds")

        # Format response

        response = {
            "label": result[0]["label"],
            "confidence": round(result[0]["score"], 2),
        }

        return response

    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


# Rate limiter (simple in-memory implementation)
class RateLimiter:
    def __init__(self, requests_per_minute=60):
        self.requests_per_minute = requests_per_minute
        self.request_timestamps = []

    def check_rate_limit(self):
        """Check if the request is within rate limits."""
        current_time = time.time()

        # Remove timestamps older than 1 minute
        self.request_timestamps = [ts for ts in self.request_timestamps
                                   if current_time - ts < 60]

        # Check if we're over the limit
        if len(self.request_timestamps) >= self.requests_per_minute:
            raise HTTPException(
                status_code=429,
                detail="Rate limit exceeded. Please try again later."
            )

        # Add current timestamp
        self.request_timestamps.append(current_time)
        return True


# Create rate limiter instance
rate_limiter = RateLimiter()


# Define endpoint
@app.post("/predict", response_model=PredictResponse)
async def predict_sentiment_endpoint(request: PredictRequest, _: bool = Depends(ensure_model_loaded),
                                     __: bool = Depends(rate_limiter.check_rate_limit)):
    """
    Predict the sentiment of the input text using the Hugging Face pipeline.

    Parameters:
    - text: The text to analyze
    - return_all_scores: Whether to return scores for all labels or just the best one

    Returns:
    - text: The input text
    - label: The predicted sentiment label
    - score: The confidence score for the predicted label
    - all_scores: (Optional) Scores for all possible labels
    """
    result = predict_with_pipeline(request.text)
    return result


# Health check endpoint
@app.get("/health")
async def health_check(_: bool = Depends(ensure_model_loaded)):
    """Health check endpoint to verify the API is running."""
    return {"status": "ok"}


# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Sentiment Analysis API",
        "usage": "POST /predict with JSON body containing 'text' field",
        "documentation": "/docs or /redoc for API documentation"
    }


# Add middleware for request logging
@app.middleware("http")
async def log_requests(request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    logger.info(f"{request.method} {request.url.path} - Status: {response.status_code} - Time: {process_time:.4f}s")
    return response


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
