from __future__ import annotations

import json
from typing import AsyncGenerator

from src.proxy.state import state_manager
from src.core.logger import logger


class StreamHandler:
    """
    Handles de-anonymization of SSE streams.
    Implements a lookahead buffer to handle fragmented tokens like [PER, SON_1].
    """
    def __init__(self, request_id: str):
        self.request_id = request_id
        self.buffer = ""
        self.mapping = state_manager.get_mapping(request_id)
        # We look for patterns like [ENTITY_N]
        self.token_pattern = r"\[[A-Z_]+_\d+\]"

    async def handle_stream(self, response_stream: AsyncGenerator[bytes, None]) -> AsyncGenerator[str, None]:
        """
        Parses bytes from the response stream, de-anonymizes on the fly, and yields SSE strings.
        """
        try:
            async for chunk in response_stream:
                chunk_str = chunk.decode("utf-8")
                logger.info(f"RAW CHUNK FROM PROVIDER: {chunk_str}")
                self.buffer += chunk_str

                # De-anonymize the buffer
                # Since tokens are in the form [PERSON_1], they might be split across chunks.
                # We only replace if we see a complete token.
                # If we see a partial token at the end of the buffer (e.g. "[PER"), we keep it in buffer.
                
                output_content, remaining_buffer = self._process_buffer(self.buffer)
                self.buffer = remaining_buffer
                
                if output_content:
                    yield output_content

            # Final flush of the buffer
            if self.buffer:
                yield self._replace_tokens(self.buffer)

        finally:
            # Clean up state when stream ends or is interrupted
            state_manager.delete_request(self.request_id)

    def _process_buffer(self, buffer: str) -> tuple[str, str]:
        """
        Scans for complete tokens and partial tokens at the end.
        Returns (reconstructed_content, remaining_buffer).
        """
        # Tokens are like [PERSON_1]
        # We find all occurrences of '['
        last_bracket_idx = buffer.rfind('[')
        
        if last_bracket_idx == -1:
            # No possible token start, we can safely output and clear buffer
            return self._replace_tokens(buffer), ""
        
        # Check if there's a closing ']' after the last bracket
        if ']' in buffer[last_bracket_idx:]:
            # The last token might be complete, or there's more text after it.
            # We can output everything up to the end of the last complete token.
            # But wait, there might be another partial token starting after the last ']'.
            # It's safer to output everything up to the last character that isn't part of a potential token.
            return self._replace_tokens(buffer), ""
        else:
            # We have a potential partial token at the end.
            # Output everything BEFORE the last '[' and keep the rest in buffer.
            return self._replace_tokens(buffer[:last_bracket_idx]), buffer[last_bracket_idx:]

    def _replace_tokens(self, text: str) -> str:
        """
        Performs replacement of all complete tokens found in text.
        """
        if not self.mapping:
            return text
            
        result = text
        for token, original in sorted(self.mapping.items(), key=lambda x: len(x[0]), reverse=True):
            result = result.replace(token, original)
        return result
