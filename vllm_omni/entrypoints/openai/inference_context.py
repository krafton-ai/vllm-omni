# SPDX-License-Identifier: Apache-2.0
"""Thread-safe per-request inference context storage.

Provides a module-level singleton (``inference_ctx``) that accumulates
diagnostic context (output modalities, duplex flags, sampling params, …)
for each in-flight request so the ``/v1/inference/context`` endpoint can
retrieve it without touching the hot inference path.
"""

from __future__ import annotations

import threading
import time
from typing import Any

from vllm.logger import init_logger

logger = init_logger(__name__)

# Default TTL in seconds before an entry is considered stale and eligible
# for cleanup (5 minutes is generous for any real request lifecycle).
_DEFAULT_TTL_SECONDS: float = 300.0


class InferenceContextCollector:
    """Thread-safe store of per-request inference context.

    Entries are keyed by *request_id* and expire after ``ttl`` seconds so
    that long-running servers don't accumulate unbounded memory.
    """

    def __init__(self, ttl: float = _DEFAULT_TTL_SECONDS) -> None:
        self._ttl = ttl
        self._lock = threading.Lock()
        # Maps request_id -> {"ctx": dict, "ts": float}
        self._store: dict[str, dict[str, Any]] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def record_ctx(self, request_id: str, ctx_data: dict[str, Any]) -> None:
        """Create or replace the ctx entry for *request_id*."""
        with self._lock:
            self._store[request_id] = {
                "ctx": dict(ctx_data),
                "ts": time.monotonic(),
            }

    def update_ctx(self, request_id: str, updates: dict[str, Any]) -> None:
        """Merge *updates* into an existing entry (creates one if absent)."""
        with self._lock:
            entry = self._store.get(request_id)
            if entry is None:
                self._store[request_id] = {
                    "ctx": dict(updates),
                    "ts": time.monotonic(),
                }
            else:
                entry["ctx"].update(updates)
                entry["ts"] = time.monotonic()

    def get_ctx(self, request_id: str) -> dict[str, Any] | None:
        """Return the ctx dict for *request_id*, or ``None`` if not found."""
        with self._lock:
            entry = self._store.get(request_id)
            if entry is None:
                return None
            return dict(entry["ctx"])

    def delete_ctx(self, request_id: str) -> None:
        """Remove the entry for *request_id* if it exists."""
        with self._lock:
            self._store.pop(request_id, None)

    def cleanup_expired(self) -> int:
        """Delete all entries older than *ttl* seconds.

        Returns the number of entries removed.
        """
        cutoff = time.monotonic() - self._ttl
        with self._lock:
            expired = [rid for rid, entry in self._store.items() if entry["ts"] < cutoff]
            for rid in expired:
                del self._store[rid]
        if expired:
            logger.debug("InferenceContextCollector: evicted %d stale entries", len(expired))
        return len(expired)

    def __len__(self) -> int:
        with self._lock:
            return len(self._store)


# ---------------------------------------------------------------------------
# Module-level singleton — import this everywhere instead of constructing new
# instances.
# ---------------------------------------------------------------------------

inference_ctx: InferenceContextCollector = InferenceContextCollector()
