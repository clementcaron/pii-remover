# PII Remover Proxy

A lightweight, highly scalable, stateless reverse proxy (AI Gateway) designed to anonymize Personally Identifiable Information (PII) in requests to LLM providers and de-anonymize it in responses.

## Features

- Multi-Provider Support: Supports OpenAI, Anthropic, and Google Gemini API formats.
- Robust Anonymization: Uses Microsoft Presidio and spaCy for Named Entity Recognition (NER) and Regex for structured data.
- Ephemeral State: Request-scoped mapping storage with TTL-based cleanup to prevent memory leaks.
- Streaming Support: Handles Server-Sent Events (SSE) with a lookahead buffer to correctly de-anonymize fragmented chunks.
- Security-First: Non-root Docker execution, Bearer token authentication, and sanitized logging.

## Core Architecture

- api/: FastAPI routes and middleware for authentication and request tracking.
- core/: Configuration management and sanitized logging.
- engine/: Anonymization logic using NLP and Regex.
- proxy/: HTTP client management and SSE stream handling.

## Installation

### Prerequisites

- Python 3.12+
- uv (recommended) or pip

### Standard Setup

1. Clone the repository and navigate to the root directory.
2. Create and activate a virtual environment.
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Download the spaCy model (small model for lower footprint):
   ```bash
   python -m spacy download en_core_web_sm
   ```

### Using uv

```bash
uv pip install -r requirements.txt
uv pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.7.1/en_core_web_sm-3.7.1-py3-none-any.whl
```

## Configuration

Copy the example environment file and fill in your API keys:

```bash
cp .env.example .env
```

Key variables:
- OPENAI_API_KEY: Your OpenAI API key.
- ANTHROPIC_API_KEY: Your Anthropic API key.
- GEMINI_API_KEY: Your Google Gemini API key.
- PROXY_AUTH_TOKEN: A secure token you choose to protect your proxy.

## Running the Proxy

```bash
python -m src.main
```

The server will start on http://0.0.0.0:8000.

## Usage Example

Send an authenticated request to the proxy:

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Authorization: Bearer your-proxy-token" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-3.5-turbo",
    "messages": [{"role": "user", "content": "My name is John Doe and my email is john@example.com."}],
    "stream": false
  }'
```

## Docker Deployment

Build the secure production image:

```bash
docker build -t pii-remover -f docker/Dockerfile .
```

Run the container:

```bash
docker run -p 8000:8000 --env-file .env pii-remover
```
