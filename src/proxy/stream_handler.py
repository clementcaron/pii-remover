from __future__ import annotations

import json
import codecs
import re
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
        logger.debug(f"[{request_id}] StreamHandler initialized with mapping: {self.mapping}")
        # We look for patterns like [ENTITY_N]
        self.token_pattern = r"\[[A-Z_]+_\d+\]"
        self.decoder = codecs.getincrementaldecoder("utf-8")()

    async def handle_stream(self, response_stream: AsyncGenerator[bytes, None]) -> AsyncGenerator[str, None]:
        """
        Parses bytes from the response stream, de-anonymizes on the fly, and yields SSE strings.
        """
        try:
            async for chunk in response_stream:
                chunk_str = self.decoder.decode(chunk)
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
            
            # Flush any remaining bytes from the decoder (though there shouldn't be any if UTF-8 is valid)
            remaining = self.decoder.decode(b"", final=True)
            if remaining:
                yield self._replace_tokens(remaining)

        finally:
            # Clean up state when stream ends or is interrupted
            logger.debug(f"[{self.request_id}] StreamHandler cleaning up state")
            state_manager.delete_request(self.request_id)

    def _process_buffer(self, buffer: str) -> tuple[str, str]:
        """
        Scans for complete tokens and partial tokens at the end.
        Returns (reconstructed_content, remaining_buffer).
        """
        # Find the last potential token start '['
        last_bracket_idx = buffer.rfind('[')
        
        if last_bracket_idx == -1:
            return self._replace_tokens(buffer), ""
            
        # Check the content from the last '[' to the end
        substring = buffer[last_bracket_idx:]
        
        # If there is a closing bracket, we check if it's a complete token or just JSON
        if ']' in substring:
            # If the text between '[' and the FIRST ']' looks like a token, then it's complete.
            # Our tokens are [A-Z_0-9]+]. 
            first_closing_idx = substring.find(']')
            potential_token_content = substring[1:first_closing_idx]
            
            # If it matches the internal part of our token pattern, it's a complete token.
            if re.fullmatch(r"[A-Z0-9\_]+", potential_token_content):
                # Complete token found, we can process everything
                return self._replace_tokens(buffer), ""
            else:
                # It's a literal '[' (e.g. JSON array start) followed by something else.
                # We can safely output it.
                return self._replace_tokens(buffer), ""
        
        # If we have an unclosed '[', we must decide if it's a partial token.
        # Partial tokens only contain valid token characters.
        potential_partial = substring[1:]
        if not potential_partial or re.fullmatch(r"[A-Z0-9\_]+", potential_partial):
            # It's a potential partial token. We wait for more data.
            if len(substring) > 50:
                return self._replace_tokens(buffer), ""
            
            # Output everything BEFORE the last '['
            return self._replace_tokens(buffer[:last_bracket_idx]), buffer[last_bracket_idx:]
        else:
            # Literal '[', output it.
            return self._replace_tokens(buffer), ""

    def _replace_tokens(self, text: str) -> str:
        """
        Performs replacement of all complete tokens found in text.
        Safely escapes the original string so it doesn't break JSON formatting if it contains quotes/newlines.
        """
        if not self.mapping:
            return text
            
        import json
        import re
        result = text
        for token, original in sorted(self.mapping.items(), key=lambda x: len(x[0]), reverse=True):
            # The streaming response is raw JSON text inside SSE events.
            # We must escape the original string so that newlines/quotes don't break the JSON structure.
            safe_original = json.dumps(original)[1:-1]
            
            # The LLM (especially Gemini) often escapes markdown characters. 
            # e.g., `[PHONE_NUMBER_1]` might become `\[PHONE\_NUMBER\_1\]`
            # Inside a JSON stream, this looks like `\\[PHONE\\_NUMBER\\_1\\]`.
            # We build a regex pattern to optionally allow these JSON-escaped backslashes.
            escaped_token = ""
            for char in token:
                if char in ["_", "[", "]"]:
                    # Allow optional backslashes (up to 2) before special characters
                    # This handles both Markdown escaping \[...\] and JSON escaping \\[...\\].
                    escaped_token += r"\\{0,2}" + re.escape(char)
                else:
                    escaped_token += re.escape(char)
                    
            result = re.sub(escaped_token, lambda m: safe_original, result)
        return result
