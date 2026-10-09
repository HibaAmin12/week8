"""Retrieval + answer generation. Returns the answer, citations and an itemised cost breakdown."""
import json
import math
import re
import time

from . import config, embeddings, llm, reranker
from .costs import usd

BROAD = re.compile(r"summar|overview|study guide|\bfaq\b|timeline|quiz|flashcard|key (points|takeaways)|"
                   r"outline|main (ideas|points)|briefing|table of contents", re.I)

SYSTEM = """You are a helpful assistant inside a document Q&A app. The numbered passages below come from the user's documents.
Rules:
- For questions about the documents, use ONLY the passages. After every claim, cite the passage number(s) in square brackets, like [2] or [1][4]. Cite each claim once, at the end of the sentence, and never repeat the same citations.
- A passage may include a little surrounding text; cite the passage whose words support the claim.
- If the question is about the documents but the passages do not contain the answer, say so plainly. Never invent facts about the documents.
- If the message is casual conversation (greetings, thanks, "who are you?") or a simple general question unrelated to the documents (like arithmetic or basic common knowledge), answer it naturally and briefly. Do not use citations for these answers, and do not claim the answer comes from the documents.
- For anything about health, safety, law, money or other high-stakes topics, do not answer from general knowledge. Say it is outside the documents and suggest a qualified professional.
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


# ---------------- Evidence highlighting ----------------
_WORD = re.compile(r"\w+(?:['\u2019]\w+)?")
_STOP = set("""the and for are was were has have had this that these those with from into onto over under about
above below not but its his her their our your you who whom whose what which when where why how can could would should
will shall may might must than then them they she him also any all each both such very more most some other only own
same too just per via out off one two list name tell give show explain describe please document documents passage
passages text page according""".split())
_ANS_SPLIT = re.compile(r"(?<=[.!?\u06d4])\s+|\n+")
_SENT_SPLIT = re.compile(r"(?<=[.!?\u06d4])(?:\[[^\]\s]{1,4}\])*[\"')\]]?\s+|\n+")
_SUFFIXES = ("ations", "ation", "ings", "ing", "ions", "ion", "ies", "ed", "es", "ly", "s")


def _norm(w):
    w = w.lower().replace("\u2019", "'")
    if w.endswith("'s"):
        w = w[:-2]
    for suf in _SUFFIXES:
        if w.endswith(suf) and len(w) - len(suf) >= 4:
            return w[:-len(suf)]
    return w


def _kw(text):
    return {n for n in map(_norm, _WORD.findall(text)) if len(n) > 2 and n not in _STOP}


def _sentence_spans(text):
    """(start, end) of each sentence inside `text`, whitespace trimmed."""
    spans, pos = [], 0

    def add(a, b):
        while a < b and text[a].isspace():
            a += 1
        while b > a and text[b - 1].isspace():
            b -= 1
        if b > a:
            spans.append((a, b))

    for m in _SENT_SPLIT.finditer(text):
        add(pos, m.start())
        pos = m.end()
    add(pos, len(text))
    return spans

def _window(text, ckw, maxlen=300):
    """For a long sentence: the stretch (<= maxlen chars) holding the most distinct answer words."""
    ms = [m for m in _WORD.finditer(text) if _norm(m.group()) in ckw]
    if not ms:
        return 0, min(len(text), maxlen)
    best = (0, 0, 0)
    for i in range(len(ms)):
        j = i
        while j + 1 < len(ms) and ms[j + 1].end() - ms[i].start() <= maxlen:
            j += 1
        score = len({_norm(m.group()) for m in ms[i:j + 1]})
        if score > best[0]:
            best = (score, i, j)
    _, i, j = best
    return ms[i].start(), ms[j].end()


def _claim(answer_text, n):
    """Sentences of the answer that carry reference [n]. A citation split off by the sentence splitter
    (e.g. "...com. [1]") is re-attached to the sentence before it."""
    parts = []
    for x in _ANS_SPLIT.split(answer_text):
        if parts and re.fullmatch(r"\s*(\[\d+\]\s*)+", x):
            parts[-1] += " " + x
        else:
            parts.append(x)
    return " ".join(re.sub(r"\[\d+\]", "", x) for x in parts if f"[{n}]" in x)


_IDF = {}


def _doc_freq(store, doc_id):
    """How many chunks of this document contain each keyword (cached)."""
    d = store.docs.get(doc_id)
    if not d:
        return {}, 0
    key = (doc_id, len(d["chunks"]))
    if key not in _IDF:
        df = {}
        for c in d["chunks"]:
            for w in _kw(c["text"]):
                df[w] = df.get(w, 0) + 1
        _IDF[key] = (df, len(d["chunks"]))
    return _IDF[key]


def _weights(store, hit, ckw, qkw):
    """Rare words count more; words that only echo the question count less."""
    df, n_chunks = _doc_freq(store, hit["doc_id"])
    out = {}
    for w in ckw:
        idf = math.log((n_chunks + 1) / (df.get(w, 0) + 1)) + 0.1
        out[w] = idf * (0.4 if w in qkw else 1.0)
    return out


def _evidence(store, hit, n, answer_text, message):
    """The text to show for reference [n] plus the exact ranges to highlight inside it.
    Returns None when the passage does not really support the claim (the reference is then dropped)."""
    claim = _claim(answer_text, n)
    ckw = _kw(claim)
    if not ckw:
        return None
    W = _weights(store, hit, ckw, _kw(message))
    total = sum(W.values()) or 1.0

    def sc(words):
        return sum(W[w] for w in words if w in W) / total

    units = store.units(hit["doc_id"]) or []
    cands = []
    for seg in hit["segments"]:
        if seg["unit"] >= len(units):
            continue
        base = units[seg["unit"]]["text"].find(seg["text"])
        if base < 0:
            continue
        for a, b in _sentence_spans(seg["text"]):
            matched = ckw & _kw(seg["text"][a:b])
            cands.append((sc(matched), seg, base + a, base + b))
    if not cands:
        return None
    cands.sort(key=lambda c: (-c[0], c[3] - c[2]))
    best, best_seg = cands[0][0], cands[0][1]
    whole = {"unit": hit["unit"], "label": hit["label"], "text": hit["text"]}
    if best < config.CITE_MIN_SCORE:   # weak match: keep the reference, but do NOT highlight a wrong sentence
        return {"segment": whole, "highlight": [], "score": best}

    utext = units[best_seg["unit"]]["text"]
    pool = [c for c in cands if c[1] is best_seg]
    chosen, covered = [], set()
    limit = 1 if len(ckw) < 10 else 2
    while pool and len(chosen) < limit:
        def gain(c, covered=covered):
            return sc((ckw & _kw(utext[c[2]:c[3]])) - covered)
        c = max(pool, key=lambda c: (gain(c), -(c[3] - c[2])))
        if gain(c) < (config.CITE_MIN_SCORE if not chosen else 0.2):
            break
        chosen.append(c)
        covered |= ckw & _kw(utext[c[2]:c[3]])
        pool.remove(c)
    if not chosen:
        return {"segment": whole, "highlight": [], "score": best}

    chosen.sort(key=lambda c: c[2])
    hl = []
    for _, seg, s0, e0 in chosen:
        if e0 - s0 > 300:      # long sentence: highlight only the part that holds the answer words
            a, b = _window(utext[s0:e0], ckw, 300)
            s0, e0 = s0 + a, s0 + b
        hl.append({"unit": seg["unit"], "start": s0, "end": e0, "text": utext[s0:e0]})
    shown = best_seg
    if best_seg["unit"] == hit["unit"]:
        ob = utext.find(hit["text"])
        if ob >= 0 and hl[0]["start"] >= ob and hl[-1]["end"] <= ob + len(hit["text"]):
            shown = whole
    return {"segment": shown, "highlight": hl, "score": best}


_NEG = re.compile(r"(don't|do not|doesn't|does not|didn't|did not|no)\s+(say|mention|contain|include|provide|specify|state|"
                  r"have|cover)|not\s+(mentioned|found|stated|specified|provided|covered)|cannot find|can't find", re.I)


def _safe_evidence(*args):
    try:
        return _evidence(*args)
    except Exception as exc:  # highlighting must never break an answer
        print(f"[highlight] skipped: {exc}")
        return None


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
    if used and _NEG.search(res["text"]):
        used = []  # "the documents do not say..." -> nothing to cite
    citations, remap = [], {}
    for n in used:
        h = hits[n - 1]
        ev = _safe_evidence(store, h, n, res["text"], message)
        if not ev or ev["score"] < config.CITE_DROP_SCORE:
            continue  # no overlap with the claim at all -> unrelated reference, drop it
        seg = ev["segment"]
        remap[n] = len(citations) + 1
        citations.append({"n": remap[n], "label": seg["label"], "segments": [seg], "highlight": ev["highlight"],
                          "doc_id": h["doc_id"], "doc": h["doc"], "unit": seg["unit"], "text": seg["text"]})
    if used and not citations:  # e.g. the answer is in another language: keep what the model cited, whole chunk
        for n in used:
            h = hits[n - 1]
            remap[n] = len(citations) + 1
            citations.append({"n": remap[n], "label": h["label"], "segments": [{"unit": h["unit"], "label": h["label"], "text": h["text"]}],
                              "highlight": [], "doc_id": h["doc_id"], "doc": h["doc"], "unit": h["unit"], "text": h["text"]})
    answer_text = re.sub(r"\[(\d+)\]",
                         lambda m: f"[{remap[int(m.group(1))]}]" if int(m.group(1)) in remap else "", res["text"])
    answer_text = re.sub(r"\s+([.,;:])", r"\1", answer_text)

    in_cost = usd(res["input_tokens"], config.PRICE_IN)
    out_cost = usd(res["output_tokens"], config.PRICE_OUT)
    usage.update(input_tokens=res["input_tokens"], output_tokens=res["output_tokens"],
                 input_cost=in_cost, output_cost=out_cost, model=res["model"],
                 total_cost=usage["embed_cost"] + usage["rerank_cost"] + in_cost + out_cost,
                 seconds=round(time.time() - t0, 1))
    retrieved = [{"n": i, "doc": h["doc"], "label": h["span"]} for i, h in enumerate(hits, 1)]
    return {"answer": answer_text, "citations": citations, "usage": usage, "mode": mode, "retrieved": retrieved}


SUGGEST_SYSTEM = """You suggest follow-up questions for a document Q&A app.
Return ONLY a JSON array of exactly 3 short questions (at most 12 words each) the user could ask next.
Rules:
- Each question must be answerable from the document excerpts below.
- Do not repeat the question that was just asked.
- Make them different from each other: one about a detail, one about a related concept, one practical.
- Write in the same language as the user's question. No numbering, no extra text."""


def _parse_questions(text):
    text = re.sub(r"```(?:json)?", "", text)
    m = re.search(r"\[.*\]", text, re.S)
    items = []
    if m:
        try:
            items = json.loads(m.group(0))
        except ValueError:
            items = []
    if not items:  # fall back to one question per line
        items = [re.sub(r"^[\s\-*\d.)]+", "", ln).strip(' "') for ln in text.splitlines()]
    out = []
    for q in items:
        q = str(q).strip()
        if len(q.split()) > 2 and q not in out:
            out.append(q)
    return out[:3]


def suggest(store, message, answer_text, doc_ids, passages=None):
    """Three follow-up questions grounded in the documents. Returns {"questions", "usage"}."""
    if not config.SUGGESTIONS:
        return {"questions": [], "usage": None}
    excerpts = [t for t in (passages or []) if t] or [h["text"] for h in store.sample(doc_ids, 4)]
    excerpts = [t[:500] for t in excerpts[:4]]
    if not excerpts:
        return {"questions": [], "usage": None}
    user = (f"User question: {message}\n\nAnswer given: {answer_text[:800]}\n\nDocument excerpts:\n"
            + "\n---\n".join(excerpts))
    res = llm.chat([{"role": "system", "content": SUGGEST_SYSTEM}, {"role": "user", "content": user}])
    usage = {"input_tokens": res["input_tokens"], "output_tokens": res["output_tokens"],
             "input_cost": usd(res["input_tokens"], config.PRICE_IN),
             "output_cost": usd(res["output_tokens"], config.PRICE_OUT)}
    return {"questions": _parse_questions(res["text"]), "usage": usage}