"""Optional cross-encoder reranker (runs locally, free). Re-orders vector-search candidates by relevance."""
import threading

from . import config
from .costs import estimate_tokens

_lock = threading.Lock()
_model = None


def _load():
    global _model
    with _lock:
        if _model is None:
            from fastembed.rerank.cross_encoder import TextCrossEncoder
            _model = TextCrossEncoder(model_name=config.RERANK_MODEL)
        return _model


def rerank(query: str, texts: list):
    """Return (scores, tokens). Scores line up with `texts`. Falls back to None if reranking is unavailable."""
    if not config.RERANK_ENABLED or config.EMBED_PROVIDER == "fake" or len(texts) < 2:
        return None, 0
    try:
        scores = list(_load().rerank(query, texts))
    except Exception as exc:  # model missing or failed: keep vector order
        print(f"[rerank] disabled for this query: {exc}")
        return None, 0
    return scores, estimate_tokens(query) * len(texts) + sum(estimate_tokens(t) for t in texts)
