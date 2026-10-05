"""Hierarchical chunking.

Step 1 (loaders): split by the file's real structure (page, heading, slide, row group).
Step 2 (here): inside each unit, cut ~CHUNK_SIZE character windows with CHUNK_OVERLAP overlap,
preferring sentence ends, then word boundaries. A chunk never crosses a unit boundary, so the
citation label (page, slide, section) is always correct.
"""
import re

from . import config

_SENTENCE_END = re.compile(r"[.!?\u06d4][\"')\]]?\s")


def merge_small(units: list, min_chars: int = None) -> list:
    """Merge a tiny mergeable unit into the one after it, so headings are not left alone."""
    min_chars = config.MIN_SECTION_CHARS if min_chars is None else min_chars
    out, carry = [], None
    for u in units:
        if carry is not None:
            u = {**carry, "text": carry["text"] + "\n\n" + u["text"]}
            carry = None
        if u["mergeable"] and len(u["text"]) < min_chars:
            carry = u
        else:
            out.append(u)
    if carry is not None:
        out.append(carry)
    return out


def _cut_point(text: str, start: int, hard_end: int) -> int:
    """Best end position <= hard_end: sentence end, else whitespace, else hard cut."""
    window_from = start + (hard_end - start) // 2  # never cut inside the first half
    best = -1
    for m in _SENTENCE_END.finditer(text, window_from, hard_end):
        best = m.end()
    if best > 0:
        return best
    ws = text.rfind(" ", window_from, hard_end)
    return ws if ws > 0 else hard_end


def split_text(text: str, size: int = None, overlap: int = None) -> list:
    """Return (start, end) offsets of chunks inside `text`."""
    size = size or config.CHUNK_SIZE
    overlap = config.CHUNK_OVERLAP if overlap is None else overlap
    n = len(text)
    if n <= size:
        return [(0, n)]
    spans, pos = [], 0
    while pos < n:
        end = n if n - pos <= size else _cut_point(text, pos, pos + size)
        spans.append((pos, end))
        if end >= n:
            break
        nxt = max(end - overlap, pos + 1)
        space = text.find(" ", nxt, end)  # start the next chunk on a word boundary
        pos = space + 1 if space != -1 else end
    return spans


def make_chunks(units: list) -> list:
    """Turn units into chunks: {unit, label, text, start, ctx}.

    `text` is what the user sees and the LLM reads. `ctx` (heading + text) is what gets embedded.
    """
    chunks = []
    for ui, u in enumerate(units):
        for s, e in split_text(u["text"]):
            text = u["text"][s:e].strip()
            if not text:
                continue
            prefix = f"{u['heading']}\n" if u.get("heading") else ""
            chunks.append({"unit": ui, "label": u["label"], "text": text, "start": s, "ctx": prefix + text})
    return chunks
