from __future__ import annotations

import collections
import threading
import time
from typing import Dict


class StateManager:
    """
    Ephemeral, thread-safe state manager for PII mappings.
    Stores mappings scoped by request_id.
    Includes a TTL-based cleanup mechanism.
    """
    def __init__(self, ttl_seconds: int = 3600):
        self._storage: Dict[str, Dict[str, str]] = {}
        self._timestamps: Dict[str, float] = {}
        self._lock = threading.Lock()
        self._ttl = ttl_seconds

    def set_mapping(self, request_id: str, token: str, original: str):
        with self._lock:
            if request_id not in self._storage:
                self._storage[request_id] = {}
                self._timestamps[request_id] = time.time()
            self._storage[request_id][token] = original

    def get_mapping(self, request_id: str) -> Dict[str, str]:
        with self._lock:
            return self._storage.get(request_id, {}).copy()

    def get_original(self, request_id: str, token: str) -> str | None:
        with self._lock:
            return self._storage.get(request_id, {}).get(token)

    def delete_request(self, request_id: str):
        with self._lock:
            self._storage.pop(request_id, None)
            self._timestamps.pop(request_id, None)

    def cleanup(self):
        """
        Removes mappings that have exceeded the TTL.
        """
        now = time.time()
        with self._lock:
            expired = [rid for rid, ts in self._timestamps.items() if now - ts > self._ttl]
            for rid in expired:
                self._storage.pop(rid, None)
                self._timestamps.pop(rid, None)


state_manager = StateManager()
