from __future__ import annotations

import json
from fastapi import APIRouter, Request, HTTPException, Response
from fastapi.responses import StreamingResponse, JSONResponse
import httpx

from src.core.config import settings
from src.core.logger import logger
from src.engine.nuner import NuNERAnonymizer
from src.proxy.client import ProxyClient
from src.proxy.stream_handler import StreamHandler
from src.proxy.state import state_manager

router = APIRouter()
anonymizer = NuNERAnonymizer()

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
            logger.debug(f"[{request_id}] SENT TO PROVIDER: {payload}")
            body = json.dumps(payload).encode("utf-8")
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
            try:
                async with client.stream(method, target_url, headers=proxy_headers, content=body) as resp:
                    if resp.status_code != 200:
                        # For non-200, we still want to clean up state as we won't be de-anonymizing
                        state_manager.delete_request(request_id)
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
                        # delete_request is handled below in the finally block for non-streaming
                        yield content_str.encode("utf-8")
            except Exception as e:
                logger.error(f"Stream error: {e}")
                state_manager.delete_request(request_id)
                raise
            finally:
                # Ensure cleanup for non-streaming success path
                # Streaming path is handled by StreamHandler's finally block
                if not is_streaming:
                    state_manager.delete_request(request_id)

        if is_streaming:
            return StreamingResponse(response_streamer(), media_type="text/event-stream")
        else:
            # Collect full response for non-streaming
            full_response = b""
            try:
                async for chunk in response_streamer():
                    if isinstance(chunk, str):
                        full_response += chunk.encode("utf-8")
                    else:
                        full_response += chunk
            except Exception:
                # Errors are logged in streamer
                raise HTTPException(status_code=502, detail="Bad Gateway")
            
            # Try to return as JSON if possible
            try:
                return JSONResponse(content=json.loads(full_response))
            except:
                return Response(content=full_response)

    except httpx.RequestError as exc:
        logger.error(f"An error occurred while requesting {exc.request.url!r}: {exc}")
        state_manager.delete_request(request_id)
        raise HTTPException(status_code=502, detail="Bad Gateway")
    except Exception as exc:
        logger.error(f"Unexpected error in proxy: {exc}")
        state_manager.delete_request(request_id)
        raise HTTPException(status_code=500, detail="Internal Server Error")


def _anonymize_payload(payload: any, request_id: str) -> any:
    """
    Optimized version that collects all text fields and anonymizes them in a single batch.
    """
    # 1. Collect all paths and values that need anonymization
    text_items = [] # list of (parent_container, key_or_index, value)

    def _collect(data, target_list):
        if isinstance(data, dict):
            for k, v in data.items():
                if k in ["content", "text"] and isinstance(v, str):
                    target_list.append((data, k, v))
                else:
                    _collect(v, target_list)
        elif isinstance(data, list):
            for i, item in enumerate(data):
                if isinstance(item, str) and "role" in data[i-1 if i > 0 else 0]: # Simplified check for content-only lists
                    target_list.append((data, i, item))
                else:
                    _collect(item, target_list)

    # Initial pass to find all strings
    _collect(payload, text_items)

    if not text_items:
        return payload

    # 2. Batch anonymize
    texts_to_process = [item[2] for item in text_items]
    
    # Check if anonymizer supports batching
    if hasattr(anonymizer, "anonymize_batch"):
        anonymized_texts = anonymizer.anonymize_batch(texts_to_process, request_id)
    else:
        anonymized_texts = [anonymizer.anonymize(t, request_id) for t in texts_to_process]

    # 3. Re-inject
    for (container, key, _), anonymized_val in zip(text_items, anonymized_texts):
        container[key] = anonymized_val

    return payload
