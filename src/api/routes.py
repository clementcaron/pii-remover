from __future__ import annotations

import json
from fastapi import APIRouter, Request, HTTPException, Response
from fastapi.responses import StreamingResponse, JSONResponse
import httpx

from src.core.config import settings
from src.core.logger import logger
from src.engine.regex_nlp import RegexNLPAnonymizer
from src.proxy.client import ProxyClient
from src.proxy.stream_handler import StreamHandler
from src.proxy.state import state_manager

router = APIRouter()
anonymizer = RegexNLPAnonymizer()

@router.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE"])
async def proxy_unified(request: Request, path: str):
    """
    Unified proxy handler for OpenAI, Anthropic, and Gemini.
    """
    request_id = request.state.request_id
    method = request.method
    # Create clean headers for the outgoing request
    proxy_headers = {}
    for k, v in request.headers.items():
        if k.lower() not in ["host", "authorization", "x-request-id", "content-length", "connection", "accept-encoding"]:
            proxy_headers[k] = v

    # Determine provider based on path or headers (simplified logic)
    # In a real app, we might have specific routes for each.
    # Here we assume the client points to the correct endpoint on our proxy.
    
    target_url = "" # Kept one instance, removed the duplicate
    if "v1/chat/completions" in path: # OpenAI
        target_url = f"https://api.openai.com/{path}"
        proxy_headers["Authorization"] = f"Bearer {settings.OPENAI_API_KEY}"
    elif "v1/messages" in path: # Anthropic
        target_url = f"https://api.anthropic.com/{path}"
        proxy_headers["x-api-key"] = settings.ANTHROPIC_API_KEY
        proxy_headers["anthropic-version"] = "2023-06-01"  # Required for Anthropic
    elif "models" in path and ("generateContent" in path or "run_content" in path or "streamGenerateContent" in path): # Gemini
        # Handles /v1/models/... and /v1beta/models/...
        target_url = f"https://generativelanguage.googleapis.com/{path}"
        proxy_headers["x-goog-api-key"] = settings.GEMINI_API_KEY
    else:
        # Fallback to OpenAI style for other paths if key exists
        target_url = f"https://api.openai.com/{path}"
        if settings.OPENAI_API_KEY:
            proxy_headers["Authorization"] = f"Bearer {settings.OPENAI_API_KEY}"

    body = await request.body()
    payload = None
    if body:
        try:
            payload = json.loads(body)
            # Anonymize payload
            # This is complex as it depends on provider schema.
            # We'll deep-scan for 'content' or 'text' fields.
            payload = _anonymize_payload(payload, request_id)
            body = json.dumps(payload).encode("utf-8")
            
            # DEBUG: See what is actually sent to the provider
            logger.info(f"SENT TO PROVIDER: {body.decode('utf-8')}")
        except json.JSONDecodeError:
            pass

    client = ProxyClient.get_client()
    
    # Check for stream parameter (OpenAI/Anthropic) or path (Gemini)
    is_streaming = False
    if payload and isinstance(payload, dict):
        is_streaming = payload.get("stream", False)
    
    if "streamGenerateContent" in path:
        is_streaming = True

    try:
        # Use a stream pattern to handle both streaming and non-streaming responses efficiently
        async def response_streamer():
            async with client.stream(method, target_url, headers=proxy_headers, content=body) as resp:
                if resp.status_code != 200:
                    yield await resp.aread()
                    return

                if is_streaming:
                    handler = StreamHandler(request_id)
                    async for chunk in handler.handle_stream(resp.aiter_bytes()):
                        yield chunk
                else:
                    content = await resp.aread()
                    content_str = content.decode("utf-8")
                    # De-anonymize the whole response
                    content_str = anonymizer.deanonymize(content_str, request_id)
                    state_manager.delete_request(request_id)
                    yield content_str.encode("utf-8")

        if is_streaming:
            return StreamingResponse(response_streamer(), media_type="text/event-stream")
        else:
            # Collect full response for non-streaming
            full_response = b""
            async for chunk in response_streamer():
                if isinstance(chunk, str):
                    full_response += chunk.encode("utf-8")
                else:
                    full_response += chunk
            
            # Try to return as JSON if possible
            try:
                return JSONResponse(content=json.loads(full_response))
            except:
                return Response(content=full_response)

    except httpx.RequestError as exc:
        logger.error(f"An error occurred while requesting {exc.request.url!r}: {exc}")
        raise HTTPException(status_code=502, detail="Bad Gateway")


def _anonymize_payload(data: any, request_id: str) -> any:
    """
    Recursively scans and anonymizes 'content' or 'text' fields in LLM payloads.
    """
    if isinstance(data, dict):
        new_dict = {}
        for k, v in data.items():
            if k in ["content", "text"] and isinstance(v, str):
                new_dict[k] = anonymizer.anonymize(v, request_id)
            else:
                new_dict[k] = _anonymize_payload(v, request_id)
        return new_dict
    elif isinstance(data, list):
        return [_anonymize_payload(item, request_id) for item in data]
    return data
