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
        # Tokens are like [PERSON_1]. 
        # Token starts with '['. We find the last one.
        last_bracket_idx = buffer.rfind('[')
        
        if last_bracket_idx == -1:
            # No possible token start, we can safely output and clear buffer
            return self._replace_tokens(buffer), ""
        
        # Check if there's a closing ']' after the last bracket.
        if ']' in buffer[last_bracket_idx:]:
            # If there's a closing bracket, we might have a complete token. 
            # We treat everything until the last character as potentially processable if another '[' isn't there.
            # However, for simplicity and to avoid complex state, if we see a closing bracket,
            # we assume the current "potential token" at the end of the buffer is closed.
            return self._replace_tokens(buffer), ""
        
        # If we have a '[' without a ']', check how long the lookahead is.
        # Most tokens are short (e.g. [EMAIL_ADDRESS_123] is < 30 chars).
        # If the text after '[' is > 50 chars, it's very likely a literal '[' and not a token.
        lookahead_len = len(buffer) - last_bracket_idx
        if lookahead_len > 50:
            # Treat as literal, output everything
            return self._replace_tokens(buffer), ""
        
        # Otherwise, keep the potential token in buffer and output preceding text
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
