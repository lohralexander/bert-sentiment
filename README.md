# Sentiment Analysis API

A FastAPI-based web service for performing sentiment analysis using Hugging Face's `distilbert-base-uncased-finetuned-sst-2-english` model. Built for both development and deployment via Docker.

## 🚀 Features

- 🔍 Sentiment prediction (positive/negative) using BERT
- 🔁 In-memory rate limiting (60 requests/minute)
- 🧪 Health check endpoint
- ⚡ FastAPI + Uvicorn
- 🐳 Dockerized for easy deployment

## 📦 Installation (Local)

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/sentiment-analysis-api.git
   cd sentiment-analysis-api
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the API:**
   ```bash
   uvicorn main:app --host 0.0.0.0 --port 8000
   ```

4. **Access the API:**
   - Swagger UI: http://localhost:8000/docs
   - Health check: http://localhost:8000/health

## 🧪 How to Use the API

### POST /predict

**Request body:**
```json
{
  "text": "I love this!"
}
```

**Response:**
```json
{
  "label": "POSITIVE",
  "confidence": 0.99
}
```

## 🐳 Docker Instructions

1. **Build the Docker image**
   ```bash
   docker build -t sentiment-analysis-api .
   ```

2. **Run the container**
   ```bash
   docker run -p 8000:8000 sentiment-analysis-api
   ```

## 🐳 Using Docker Compose

The `docker-compose.yml` file handles everything for you:
```bash
docker compose up --build
```

This will:
- Build the image
- Start the container
- Expose port 8000
- Attach a named volume to persist Hugging Face model cache (`model-cache`)
- Set up a health check to `/health`

## 🔄 API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | Root endpoint with usage info |
| GET | `/health` | Health check |
| POST | `/predict` | Sentiment analysis prediction |
| GET | `/docs` | Swagger documentation |
| GET | `/redoc` | ReDoc documentation |

## 📁 Project Structure

```
.
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── main.py
└── README.md
```

## ⚠️ Notes

- The rate limiter is in-memory and not shared across multiple containers or processes.
- This app uses transformers, which downloads models from Hugging Face on the first run.
- The Docker Compose setup includes a named volume `model-cache` to persist those downloads.

## 📄 License

MIT © Alexander Lohr