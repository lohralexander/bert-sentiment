import logging
import time
from contextlib import asynccontextmanager
from typing import Dict, Any

from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel, Field, field_validator
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from transformers.pipelines import Pipeline
from typing_extensions import AsyncGenerator

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

model_name = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer: AutoTokenizer
model: AutoModelForSequenceClassification
sentiment_pipeline: Pipeline


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
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
            device=-1  # -1 = CPU, 0 = GPU
        )
        elapsed = time.time() - start_time
        logger.info(f"Model loaded successfully in {elapsed:.2f} seconds!")
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise RuntimeError(f"Failed to load model: {str(e)}")

    yield

    logger.info("Shutting down and cleaning up resources...")
    tokenizer = None
    model = None
    sentiment_pipeline = None


app = FastAPI(
    title="Sentiment Analysis API",
    description="API for sentiment analysis using BERT model from Hugging Face",
    version="1.0.0",
    lifespan=lifespan
)


class PredictRequest(BaseModel):
    text: str = Field(..., description="Text to analyze")

    @field_validator("text", mode="before")
    @classmethod
    def text_must_not_be_empty(cls, v: str) -> str:
        if v is None:
            raise ValueError('Text cannot be None')

        if not isinstance(v, str):
            raise ValueError('Text must be a string')

        if not v.strip():
            raise ValueError('Text cannot be empty')

        return v.strip()  # Strip whitespace from input


class PredictResponse(BaseModel):
    label: str
    confidence: float


def ensure_model_loaded():
    if tokenizer is None or model is None or sentiment_pipeline is None:
        logger.error("Model components not loaded")
        raise HTTPException(status_code=503, detail="Model not loaded")
    return True


def predict_with_pipeline(text: str) -> Dict[str, Any]:
    try:
        if not text or not isinstance(text, str):
            logger.error(f"Invalid input text: {text!r}")
            raise ValueError("Text must be a non-empty string")

        if not sentiment_pipeline:
            logger.error("Cannot predict, sentiment_pipeline is not loaded")
            raise RuntimeError("Model not loaded")

        logger.info(f"Processing text (length: {len(text)})")
        start_time = time.time()

        result = sentiment_pipeline(
            text
        )

        elapsed = time.time() - start_time
        logger.info(f"Prediction completed in {elapsed:.4f} seconds")

        response = {
            "label": result[0]["label"],
            "confidence": round(result[0]["score"], 2),
        }

        return response

    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


# NOTE: In-memory rate limiter – does not persist across workers or processes.
class RateLimiter:
    def __init__(self, requests_per_minute=60):
        self.requests_per_minute = requests_per_minute
        self.request_timestamps = []

    def check_rate_limit(self):
        """Check if the request is within rate limits."""
        current_time = time.time()

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


rate_limiter = RateLimiter()


@app.post("/predict", response_model=PredictResponse)
async def predict_sentiment_endpoint(request: PredictRequest, _: bool = Depends(ensure_model_loaded),
                                     __: bool = Depends(rate_limiter.check_rate_limit)):
    """
    Predict the sentiment of the input text using the Hugging Face pipeline.

    Parameters:
    - text: The text to analyze

    Returns:
    - label: The predicted sentiment label
    - confidence: The confidence score for the predicted label
    """
    result = predict_with_pipeline(request.text)
    return result


@app.get("/health")
async def health_check(_: bool = Depends(ensure_model_loaded)):
    """Health check endpoint to verify the API is running."""
    return {"status": "ok"}


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Sentiment Analysis API",
        "usage": "POST /predict with JSON body containing 'text' field",
        "documentation": "/docs or /redoc for API documentation"
    }


@app.middleware("http")
async def log_requests(request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    logger.info(f"{request.method} {request.url.path} - Status: {response.status_code} - Time: {process_time:.4f}s")
    return response
