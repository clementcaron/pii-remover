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
from src.api.payload_scanner import normalize as normalize_payload, extract as extract_text_items

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
    Normalizes the payload, extracts all text fields, anonymizes them in a
    single batch, persists the mapping, and re-injects the anonymized text.
    """
    normalize_payload(payload)

    text_items = extract_text_items(payload)
    if not text_items:
        return payload

    texts = [item.value for item in text_items]
    anonymized_texts, mapping = anonymizer.anonymize_batch(texts)

    for token, original in mapping.items():
        state_manager.set_mapping(request_id, token, original)

    for item, anonymized_val in zip(text_items, anonymized_texts):
        item.container[item.key] = anonymized_val

    return payload
