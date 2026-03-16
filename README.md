# PII Remover Proxy

A reverse proxy designed to anonymize Personally Identifiable Information in requests to LLM providers and de-anonymize it in responses.

```text
    USER REQUEST          ANONYMIZATION            LLM CALL
   (Contains PII)        (Token Swapping)       (Clean Payload)
   
    ┌──────────┐         ┌───────────────┐      ┌──────────────┐
    │  CLIENT  │ ──────▶ │   PII PROXY   │ ───▶ │ LLM PROVIDER │
    └──────────┘         └───────┬───────┘      └──────┬───────┘
          ▲                      │                     │
          │                      │ 1. Scan/Map         │
          │                      │ 2. Swap             │
          │                      │                     │
          │               ┌──────▼──────┐              │
          │               │ EPHEMERAL   │              │
          │               │ STATE (RAM) │              │
          │               └──────┬──────┘              │
          │                      │                     │
          │                      │ 3. Restore          │
    USER RESPONSE                │                  LLM RESPONSE
    (PII Restored)      ◀────────┴───────────────── (With Tokens)
```

## Features

- **Provider Support**: OpenAI, Anthropic, and Google Gemini.
- **Engine**: Pure  ([**NuNER_Zero**](https://huggingface.co/numind/NuNER_Zero)) inference for zero-shot detection.
- **Performance**: High-speed inference via transformer optimization.
- **Ephemeral State**: Request-scoped mapping storage with automated background cleanup tasks.
- **Streaming Support**: Advanced Server-Sent Events (SSE) handling with a lookahead buffer for fragmented de-anonymization.
- **Production Ready**: Under 3GB Docker footprint, non-root execution, and Bearer token security.

## Core Architecture

- `src/api/`: FastAPI routes and authentication middleware.
- `src/core/`: Configuration and security-hardened logging.
- `src/engine/`: Detection engine logic (`NuNERAnonymizer`) with entity merging for optimized recall.
- `src/proxy/`: `StreamHandler` for SSE de-anonymization and request state management.

## Quick Start (Recommended)

For the easiest setup, use the interactive configuration script. It will auto-detect your hardware (CPU/GPU), help you configure your API keys, and set up your environment automatically.

```bash
./scripts/interactive_setup.sh
```

## Installation

### Prerequisites

- Python 3.12+
- Docker (optional, for containerized deployment)
- Pytorch compatible GPU (optional, for 20x faster inference)

### Manual Setup

If you prefer not to use the interactive script:

1. **Environment**: 
   ```bash
   cp .env.example .env
   # Edit .env with your API keys and USE_GPU preference
   ```

2. **Python Setup**:
   The automated setup script handles `torch` versions (CPU vs GPU) and pre-downloads the NuNER model weights.
   ```bash
   uv venv .venv
   .venv/bin/python3 scripts/setup.py
   ```

## Docker Deployment

We recommend using **Docker Compose** for the easiest deployment. It automatically reads your `.env` (configured via `interactive_setup.sh`) to decide whether to build a CPU or GPU-accelerated version.

```bash
# Build and start the container
docker compose up -d --build

# View logs
docker compose logs -f

# Stop the proxy
docker compose down
```

> By default, GPU support is commented out in `docker-compose.yml` to prevent errors on machines without the **NVIDIA Container Toolkit**. 
> To enable GPU acceleration in Docker, ensure the toolkit is installed on your host, then uncomment the `deploy:` section in `docker-compose.yml`.

## Performance & Accuracy

The engine is optimized for high-throughput production environments:
- **Accuracy**: achieving **~54% recall** on diverse PII datasets using pure zero-shot inference (as configured).
- **Batching**: Payload de-identification is batched into a single model pass, yielding significant speedups for long conversation histories.

## Usage Example

Once the proxy is running (default `http://localhost:8000`), you can route LLM requests through it.

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Authorization: Bearer your-proxy-auth-token" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-3.5-turbo",
    "messages": [{"role": "user", "content": "My name is John Doe and my email is john@example.com."}],
    "stream": false
  }'
```
