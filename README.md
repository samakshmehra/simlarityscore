# Text Similarity API

A FastAPI-based API for computing text similarity scores using Sentence-BERT (SBERT) model.

## Features

- Computes similarity between two texts using advanced token-level and sentence-level embeddings.
- Returns a normalized similarity score between 0 and 1.
- Built with FastAPI for high performance and automatic API documentation.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/samakshmehra/simlarityscore.git
   cd simlarityscore
   ```

2. Create a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Running Locally

Start the server:
```bash
uvicorn routers.routers:app --host 0.0.0.0 --port 8000 --reload
```

The API will be available at `http://0.0.0.0:8000`.

### API Endpoint

**POST /similarity**

Compute similarity between two texts.

**Request Body:**
```json
{
  "text1": "First text here",
  "text2": "Second text here"
}
```

**Response:**
```json
{
  "similarity_score": 0.85
}
```

**Example using curl:**
```bash
curl -X POST "http://0.0.0.0:8000/similarity" \
     -H "Content-Type: application/json" \
     -d '{"text1": "Hello world", "text2": "Hi there"}'
```

### API Documentation

Visit `http://0.0.0.0:8000/docs` for interactive Swagger UI documentation.

## Deployment

### Using Docker

Build and run the Docker container:
```bash
docker build -t text-similarity-api .
docker run -p 8000:8000 text-similarity-api
```

### On Render

This app is configured for deployment on Render. Connect your GitHub repo and deploy.

## Technologies Used

- **FastAPI**: Web framework for building APIs.
- **Transformers**: Hugging Face library for SBERT model.
- **PyTorch**: Deep learning framework.
- **Uvicorn**: ASGI server for FastAPI.
