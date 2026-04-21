"""Persistent disk cache for LLM ``generate(prompt) -> response`` calls.

Backed by a single SQLite file (one row per prompt-hash). SQLite gives us:
    * atomic writes (no half-written JSON if a job dies mid-run),
    * concurrent reads from many worker processes,
    * one file to delete when you want a clean slate,
    * no millions-of-small-files inode pressure on Lustre/GPFS.

Usage:
    from opponent_model.cache import CachedLLM
    from prompt_engineer.llm.client import LlamaClient

    raw_client = LlamaClient(model_id="meta-llama/Llama-3.3-70B-Instruct",
                             temperature=0.0)
    client = CachedLLM(raw_client, cache_path="results/llm_cache.sqlite")
    # ... pass `client` to HybridAgent / anything else expecting .generate(...) ...
    print(client.stats())  # {'hits': 1240, 'misses': 312, 'hit_rate': 0.799, ...}

Caching is keyed on (namespace, prompt). The default ``namespace`` is auto-built
from the wrapped client's ``model_id`` / ``temperature`` / ``top_p`` /
``max_new_tokens`` attributes, so e.g. switching from Llama-70B greedy to
Llama-8B sampled will NOT collide. Pass a custom ``namespace`` if you wrap a
client whose attributes don't match the LlamaClient interface.

Caveat — sampling temperature > 0:
    The cache stores ONE sample per prompt. If you sample at T > 0, a cache
    hit gives you a single fixed sample, not a fresh draw from the
    distribution. For deterministic eval runs this is what you want; for
    studies of generation variance, disable the cache or namespace by run-id.
"""

from __future__ import annotations

import hashlib
import logging
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


# ── Low-level SQLite K/V store ─────────────────────────────────────────────


class DiskCache:
    """SHA256-keyed string → string cache backed by a single SQLite file."""

    _SCHEMA = """
        CREATE TABLE IF NOT EXISTS cache (
            key         TEXT PRIMARY KEY,
            namespace   TEXT NOT NULL,
            response    TEXT NOT NULL,
            created_at  REAL NOT NULL
        )
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(
            str(self.path),
            check_same_thread=False,
            isolation_level=None,
        )
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.execute(self._SCHEMA)

    @staticmethod
    def make_key(prompt: str, namespace: str = "") -> str:
        """Stable SHA256 over ``namespace || \\x00 || prompt`` (UTF-8)."""
        h = hashlib.sha256()
        h.update(namespace.encode("utf-8"))
        h.update(b"\x00")
        h.update(prompt.encode("utf-8"))
        return h.hexdigest()

    def get(self, prompt: str, namespace: str = "") -> Optional[str]:
        key = self.make_key(prompt, namespace)
        with self._lock:
            cur = self._conn.execute(
                "SELECT response FROM cache WHERE key = ?", (key,),
            )
            row = cur.fetchone()
        return row[0] if row else None

    def set(self, prompt: str, response: str, namespace: str = "") -> None:
        key = self.make_key(prompt, namespace)
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO cache (key, namespace, response, created_at) "
                "VALUES (?, ?, ?, ?)",
                (key, namespace, response, time.time()),
            )

    def __len__(self) -> int:
        with self._lock:
            cur = self._conn.execute("SELECT COUNT(*) FROM cache")
            return int(cur.fetchone()[0])

    def size_by_namespace(self) -> Dict[str, int]:
        with self._lock:
            cur = self._conn.execute(
                "SELECT namespace, COUNT(*) FROM cache GROUP BY namespace"
            )
            return {ns: int(n) for ns, n in cur.fetchall()}

    def clear(self, namespace: Optional[str] = None) -> int:
        """Drop entries (all by default, or just one namespace). Returns count."""
        with self._lock:
            if namespace is None:
                cur = self._conn.execute("DELETE FROM cache")
            else:
                cur = self._conn.execute(
                    "DELETE FROM cache WHERE namespace = ?", (namespace,),
                )
            return int(cur.rowcount or 0)

    def close(self) -> None:
        with self._lock:
            self._conn.close()


# ── LLM client wrapper ─────────────────────────────────────────────────────


class CachedLLM:
    """Wrap any object that exposes ``.generate(prompt: str) -> str``.

    Args:
        client: underlying LLM client. Must implement ``.generate(prompt)``.
        cache_path: SQLite file path. Created (and parent dirs) if missing.
        namespace: extra string mixed into every cache key. Use it to
            prevent cross-contamination between, e.g., different model IDs
            or decoding configs. ``None`` (default) → auto-detect from
            common attributes of the wrapped client (``model_id``,
            ``temperature``, ``top_p``, ``max_new_tokens``). Pass an empty
            string to share the cache across configs (rarely what you want).
        log_misses: if True, log a debug line on every cache miss.
    """

    _AUTO_ATTRS = ("model_id", "temperature", "top_p", "max_new_tokens")

    def __init__(
        self,
        client: Any,
        cache_path: str | Path,
        namespace: Optional[str] = None,
        log_misses: bool = False,
    ) -> None:
        if not hasattr(client, "generate"):
            raise TypeError("Wrapped client must implement .generate(prompt: str) -> str")
        self.client = client
        self.cache = DiskCache(cache_path)
        self.namespace = (
            namespace if namespace is not None else self._auto_namespace(client)
        )
        self.log_misses = log_misses
        self.hits = 0
        self.misses = 0

    @classmethod
    def _auto_namespace(cls, client: Any) -> str:
        parts = [
            f"{attr}={getattr(client, attr)}"
            for attr in cls._AUTO_ATTRS
            if hasattr(client, attr)
        ]
        return "|".join(parts)

    def generate(self, prompt: str) -> str:
        cached = self.cache.get(prompt, namespace=self.namespace)
        if cached is not None:
            self.hits += 1
            return cached
        if self.log_misses:
            logger.debug("LLM cache miss (namespace=%s, key=%s...)",
                         self.namespace, DiskCache.make_key(prompt, self.namespace)[:12])
        response = self.client.generate(prompt)
        self.cache.set(prompt, response, namespace=self.namespace)
        self.misses += 1
        return response

    def stats(self) -> Dict[str, Any]:
        """Return ``{hits, misses, total, hit_rate, size, namespace}``."""
        total = self.hits + self.misses
        return {
            "hits": self.hits,
            "misses": self.misses,
            "total": total,
            "hit_rate": (self.hits / total) if total else 0.0,
            "size": len(self.cache),
            "namespace": self.namespace,
        }

    def reset_stats(self) -> None:
        self.hits = 0
        self.misses = 0


__all__ = ["CachedLLM", "DiskCache"]
