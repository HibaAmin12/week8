"""Embedding providers. Every provider returns (vectors, token_count) so cost can always be shown.

local  : BAAI/bge-small-en-v1.5 through fastembed (CPU, free). Tokens are estimated.
openai : OpenAI embeddings API (billed). Tokens come from the API response.
fake   : deterministic hashing embedder for tests.
"""
import hashlib
import re
import threading

import numpy as np
import requests

from . import config
from .costs import estimate_tokens

_lock = threading.Lock()
_model = None


def embed_id() -> str:
    """Identifies the embedding setup; stored with each document so stale indexes are detected."""
    name = {"local": config.EMBED_MODEL, "openai": config.OPENAI_EMBED_MODEL}.get(config.EMBED_PROVIDER, "fake")
    return f"{config.EMBED_PROVIDER}:{name}"


def _normalize(m: np.ndarray) -> np.ndarray:
    return m / np.clip(np.linalg.norm(m, axis=1, keepdims=True), 1e-9, None)


def _local_model():
    global _model
    with _lock:
        if _model is None:
            from fastembed import TextEmbedding
            _model = TextEmbedding(model_name=config.EMBED_MODEL)
        return _model


def _fake(texts):
    out = np.zeros((len(texts), 256), dtype=np.float32)
    for i, t in enumerate(texts):
        for w in re.findall(r"\w+", t.lower()):
            out[i, int(hashlib.md5(w.encode()).hexdigest(), 16) % 256] += 1.0
    return _normalize(out + 1e-6)


def _openai(texts):
    if not config.OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY is not set (needed for OpenAI embeddings).")
    vecs, tokens = [], 0
    for i in range(0, len(texts), 96):
        r = requests.post(f"{config.OPENAI_BASE_URL}/embeddings",
                          headers={"Authorization": f"Bearer {config.OPENAI_API_KEY}"},
                          json={"model": config.OPENAI_EMBED_MODEL, "input": texts[i:i + 96]},
                          timeout=config.LLM_TIMEOUT)
        if r.status_code != 200:
            raise RuntimeError(f"OpenAI embeddings error {r.status_code}: {r.text[:300]}")
        j = r.json()
        vecs += [d["embedding"] for d in sorted(j["data"], key=lambda d: d["index"])]
        tokens += j.get("usage", {}).get("total_tokens", 0)
    return _normalize(np.array(vecs, dtype=np.float32)), tokens


def _embed(texts):
    if config.EMBED_PROVIDER == "fake":
        return _fake(texts), estimate_tokens(" ".join(texts))
    if config.EMBED_PROVIDER == "openai":
        return _openai(texts)
    model = _local_model()
    vecs = np.array(list(model.embed(texts, batch_size=config.EMBED_BATCH)), dtype=np.float32)
    return _normalize(vecs), sum(estimate_tokens(t) for t in texts)


def embed_documents(texts: list):
    """Embed chunk texts. Returns (matrix, tokens)."""
    return _embed(texts)


def embed_query(query: str):
    """Embed a search query. BGE models need an instruction prefix for queries. Returns (vector, tokens)."""
    text = query
    if config.EMBED_PROVIDER == "local" and "bge" in config.EMBED_MODEL.lower():
        text = config.BGE_QUERY_PREFIX + query
    vecs, tokens = _embed([text])
    return vecs[0], tokens
