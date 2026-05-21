from __future__ import annotations

import json
from fastapi import APIRouter, Request, HTTPException, Response
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.concurrency import run_in_threadpool
import httpx

from src.core.config import settings
from src.core.logger import logger
from src.engine.nuner import NuNERAnonymizer
from src.proxy.client import ProxyClient
from src.proxy.stream_handler import StreamHandler
from src.proxy.state import state_manager
from src.api.provider_router import resolve as resolve_provider

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

    provider = resolve_provider(path, settings)
    target_url = provider.target_url
    proxy_headers.update(provider.auth_headers)

    # Append query parameters if present
    query_params = str(request.url.query)
    if query_params:
        target_url = f"{target_url}?{query_params}"

    body = await request.body()
    payload = None
    if body:
        try:
            payload = json.loads(body)
            # Anonymize payload
            # This is complex as it depends on provider schema.
            # We'll deep-scan for 'content' or 'text' fields.
            # Bug 1: Offload heavy sync inference to threadpool to avoid blocking event loop
            payload = await run_in_threadpool(_anonymize_payload, payload, request_id)
            logger.debug(f"[{request_id}] SENT TO PROVIDER: {payload}")
            body = json.dumps(payload).encode("utf-8")
        except json.JSONDecodeError:
            pass

    client = ProxyClient.get_client()
    
    # Check for stream parameter (OpenAI/Anthropic) or provider-flagged streaming path (Gemini)
    is_streaming = provider.is_streaming_path
    if payload and isinstance(payload, dict):
        is_streaming = is_streaming or payload.get("stream", False)

    try:
        # Bug 3: Perform request first to capture upstream status code before initializing response
        req = client.build_request(method, target_url, headers=proxy_headers, content=body)
        resp = await client.send(req, stream=True)

        if is_streaming:
            async def response_streamer():
                try:
                    if resp.status_code != 200:
                        state_manager.delete_request(request_id)
                        # Stream the error body as is
                        async for chunk in resp.aiter_bytes():
                            yield chunk
                        return

                    handler = StreamHandler(request_id)
                    async for chunk in handler.handle_stream(resp.aiter_bytes()):
                        yield chunk
                except Exception as e:
                    logger.error(f"Stream error: {e}")
                    state_manager.delete_request(request_id)
                    raise
                finally:
                    await resp.aclose()

            return StreamingResponse(response_streamer(), status_code=resp.status_code, media_type="text/event-stream")
        else:
            try:
                # Capture full response for de-anonymization
                full_content = await resp.aread()
                if resp.status_code == 200:
                    content_str = full_content.decode("utf-8")
                    mapping = state_manager.get_mapping(request_id)
                    content_str = await run_in_threadpool(anonymizer.deanonymize, content_str, mapping)
                    full_content = content_str.encode("utf-8")
                
                # Cleanup state
                state_manager.delete_request(request_id)
                
                # Return with correct status code
                try:
                    return JSONResponse(content=json.loads(full_content), status_code=resp.status_code)
                except:
                    return Response(content=full_content, status_code=resp.status_code)
            finally:
                await resp.aclose()

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
    # Strip thinking config to avoid long 30-sec stalls and hidden tokens 
    # that third-party clients (like Chatbox) fail to render.
    if isinstance(payload, dict) and "generationConfig" in payload:
        if isinstance(payload["generationConfig"], dict):
            payload["generationConfig"].pop("thinkingConfig", None)

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
                # Check if item is a string and its predecessor is a dictionary containing 'role'
                predecessor = data[i-1] if i > 0 else {}
                if isinstance(item, str) and isinstance(predecessor, dict) and "role" in predecessor:
                    target_list.append((data, i, item))
                else:
                    _collect(item, target_list)

    # Initial pass to find all strings
    _collect(payload, text_items)

    if not text_items:
        return payload

    # 2. Batch anonymize — returns anonymized texts and the token→original mapping
    texts_to_process = [item[2] for item in text_items]
    anonymized_texts, mapping = anonymizer.anonymize_batch(texts_to_process)

    # Persist mapping to state so StreamHandler and deanonymize can resolve tokens
    for token, original in mapping.items():
        state_manager.set_mapping(request_id, token, original)

    # 3. Re-inject
    for (container, key, _), anonymized_val in zip(text_items, anonymized_texts):
        container[key] = anonymized_val

    return payload
