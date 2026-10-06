"""Retrieval + answer generation. Returns the answer, citations and an itemised cost breakdown."""
import re
import time

from . import config, embeddings, llm, reranker
from .costs import usd

BROAD = re.compile(r"summar|overview|study guide|\bfaq\b|timeline|quiz|flashcard|key (points|takeaways)|"
                   r"outline|main (ideas|points)|briefing|table of contents", re.I)

SYSTEM = """You answer questions using ONLY the numbered passages below, taken from the user's documents.
Rules:
- After every claim, cite the passage number(s) in square brackets, like [2] or [1][4]. Cite each claim once, at the end of the sentence, and never repeat the same citations.
- A passage may include a little surrounding text; cite the passage whose words support the claim.
- If the passages do not contain the answer, say so plainly. Never invent facts.
- Answer in English unless the user writes in another language. Be clear and concise.

PASSAGES:
{passages}"""


def _search_query(message, history):
    """Short follow-ups ('explain more') are searched together with the previous question."""
    if len(message.split()) < 4:
        for h in reversed(history or []):
            if h.get("role") == "user":
                return f"{h['content']} {message}"
    return message


def retrieve(store, message, doc_ids, history=None):
    """Return (hits, usage) where usage covers the embedding and rerank steps."""
    usage = {"embed_tokens": 0, "embed_cost": 0.0, "rerank_tokens": 0, "rerank_cost": 0.0}
    if BROAD.search(message):
        return store.sample(doc_ids, config.BROAD_K), usage, "broad"
    query = _search_query(message, history)
    qvec, tokens = embeddings.embed_query(query)
    usage["embed_tokens"] = tokens
    usage["embed_cost"] = usd(tokens, config.EMBED_PRICE_PER_M)
    hits = store.search(qvec, doc_ids, config.RETRIEVE_K)
    scores, rtok = reranker.rerank(query, [h["text"] for h in hits])
    if scores is not None:
        for h, s in zip(hits, scores):
            h["score"] = float(s)
        hits.sort(key=lambda h: -h["score"])
        usage["rerank_tokens"] = rtok
        usage["rerank_cost"] = usd(rtok, config.RERANK_PRICE_PER_M)
    return hits[:config.TOP_K], usage, "rerank" if scores is not None else "vector"


def _label(segs):
    a, b = segs[0]["label"], segs[-1]["label"]
    return a if a == b else f"{a} \u2013 {b}"


def _passage(segs):
    if len(segs) == 1:
        return segs[0]["text"]
    return "\n".join(f"({s['label']}) {s['text']}" for s in segs)


def answer(store, message, doc_ids, history=None):
    t0 = time.time()
    hits, usage, mode = retrieve(store, message, doc_ids, history)
    if not hits:
        return {"answer": "No indexed documents are selected. Upload a file or tick a source on the left.",
                "citations": [], "usage": {**usage, "input_tokens": 0, "output_tokens": 0, "input_cost": 0.0,
                                           "output_cost": 0.0, "total_cost": usage["embed_cost"], "seconds": 0},
                "mode": mode, "retrieved": []}
    for h in hits:  # show neighbouring text too, so explanations split across pages stay complete
        h["segments"] = (store.expand(h, config.NEIGHBORS) if config.NEIGHBORS > 0 and mode != "broad"
                         else [{"unit": h["unit"], "label": h["label"], "text": h["text"]}])
        h["span"] = _label(h["segments"])
    passages = "\n\n".join(f"[{i}] ({h['doc']}, {h['span']})\n{_passage(h['segments'])}" for i, h in enumerate(hits, 1))
    messages = [{"role": "system", "content": SYSTEM.format(passages=passages)}]
    for h in (history or [])[-config.HISTORY_TURNS * 2:]:
        if h.get("role") in ("user", "assistant") and h.get("content"):
            messages.append({"role": h["role"], "content": h["content"]})
    messages.append({"role": "user", "content": message})

    res = llm.chat(messages)
    used = sorted({int(n) for n in re.findall(r"\[(\d+)\]", res["text"]) if 1 <= int(n) <= len(hits)})
    citations = [{"n": n, "label": hits[n - 1]["span"], "segments": hits[n - 1]["segments"],
                  **{k: hits[n - 1][k] for k in ("doc_id", "doc", "unit", "text")}} for n in used]

    in_cost = usd(res["input_tokens"], config.PRICE_IN)
    out_cost = usd(res["output_tokens"], config.PRICE_OUT)
    usage.update(input_tokens=res["input_tokens"], output_tokens=res["output_tokens"],
                 input_cost=in_cost, output_cost=out_cost, model=res["model"],
                 total_cost=usage["embed_cost"] + usage["rerank_cost"] + in_cost + out_cost,
                 seconds=round(time.time() - t0, 1))
    retrieved = [{"n": i, "doc": h["doc"], "label": h["span"]} for i, h in enumerate(hits, 1)]
    return {"answer": res["text"], "citations": citations, "usage": usage, "mode": mode, "retrieved": retrieved}
