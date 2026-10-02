import os, re, time, uuid, threading, math
from collections import Counter
import numpy as np, requests
from pypdf import PdfReader
from flask import Flask, request, jsonify, send_from_directory

KEY = os.getenv("GEMINI_API_KEY", "")
BASE = "https://generativelanguage.googleapis.com/v1beta/models/"
CHAT_MODEL = os.getenv("CHAT_MODEL", "gemini-3.8-flash")
EMBED_MODEL = os.getenv("EMBED_MODEL", "gemini-embedding-001")
TOP_K, CHUNK = 12, 3000
FILES = {}
ALLOWED = (".pdf", ".txt", ".md", ".csv", ".json", ".docx")
STATUS = {}  # id -> {done,total,error}


def call(url, body):
    for k in range(5):
        r = requests.post(url + "?key=" + KEY, json=body, timeout=300)
        if r.status_code == 429:
            print("429 limit, 15s ruk kar retry:", r.text[:300], flush=True)
            time.sleep(15); continue
        if r.status_code in (500, 503) and k < 2:
            print(r.status_code, "busy, retry", flush=True)
            time.sleep(4 * (k + 1)); continue
        if not r.ok:
            raise RuntimeError(r.json().get("error", {}).get("message", r.text[:200]))
        return r.json()
    raise RuntimeError("Free quota reached. Try again later.")


def other_models():
    try:
        r = requests.get(BASE.rstrip("/") + "?key=" + KEY, timeout=30).json()
        bad = ("image", "tts", "live", "audio", "embedding", "robotics", "computer")
        return [m["name"].split("/")[-1] for m in r.get("models", [])
                if "generateContent" in m.get("supportedGenerationMethods", [])
                and "flash" in m["name"] and not any(x in m["name"] for x in bad)]
    except Exception:
        return []


def generate(body):
    try:
        return call(BASE + CHAT_MODEL + ":generateContent", body)
    except RuntimeError as first:
        print(CHAT_MODEL, "failed:", first, flush=True)
        for m in [x for x in other_models() if x != CHAT_MODEL][:4]:
            try:
                print("trying", m, flush=True)
                return call(BASE + m + ":generateContent", body)
            except RuntimeError as e:
                print(m, "failed:", e, flush=True)
        raise first


app = Flask(__name__, static_folder="static")
UP = os.path.join(os.path.dirname(__file__), "uploads")
os.makedirs(UP, exist_ok=True)
DOCS = {}    # id -> filename
PAGES = {}   # id -> [page texts]
CHUNKS = []  # {doc_id, doc, page, text, vec}


def embed(texts, kind="RETRIEVAL_DOCUMENT"):
    out = []
    for i in range(0, len(texts), 50):
        j = call(BASE + EMBED_MODEL + ":batchEmbedContents", {"requests": [
            {"model": "models/" + EMBED_MODEL, "taskType": kind,
             "content": {"parts": [{"text": t}]}} for t in texts[i:i + 50]]})
        out += [e["values"] for e in j["embeddings"]]
    v = np.array(out, dtype="float32")
    return v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-9)


@app.get("/")
def index():
    return send_from_directory("static", "index.html")


@app.get("/api/docs")
def docs():
    return jsonify([{"id": i, "name": n, **STATUS.get(i, {"done": 0, "total": 0})}
                    for i, n in DOCS.items()])


def worker(items, did):
    try:
        for i in range(0, len(items), 20):
            if did not in DOCS:
                return
            part = items[i:i + 20]
            for c, v in zip(part, embed([c["text"] for c in part])):
                c["vec"] = v
            if did not in DOCS:
                return
            CHUNKS.extend(part)
            STATUS[did]["done"] = min(i + 20, len(items))
            print(f"[{DOCS.get(did)}] indexed {STATUS[did]['done']}/{len(items)}", flush=True)
    except Exception as e:
        STATUS[did]["error"] = str(e)
        print("INDEX ERROR:", e, flush=True)


STOP = set("the a an of and or to in is are was were be for on with as by at from that this it its "
           "what which who how why when do does did can could should would about into than then "
           "there their they you your we our me my not no".split())
BROAD = ("summar", "overview", "study guide", "faq", "timeline", "quiz", "flashcard",
         "all the sources", "key points", "main points")


def toks(t):
    return [w for w in re.findall(r"\w+", t.lower()) if w not in STOP and len(w) > 1]


def retrieve(q, pool):
    n = len(pool)
    qt = set(toks(q))
    avg = sum(c["len"] for c in pool) / n
    df = {t: sum(1 for x in pool if t in x["tf"]) for t in qt}
    scores = []
    for c in pool:
        sc = 0.0
        for t in qt:
            tf = c["tf"].get(t, 0)
            if tf:
                idf = math.log(1 + (n - df[t] + .5) / (df[t] + .5))
                sc += idf * tf * 2.2 / (tf + 1.2 * (.25 + .75 * c["len"] / avg))
        scores.append(sc)
    if any(k in q.lower() for k in BROAD) or max(scores) == 0:
        m = min(24, n)  # broad request: sample evenly across the whole source
        return [pool[int(i * n / m)] for i in range(m)]
    top = sorted(range(n), key=lambda i: -scores[i])[:TOP_K]
    return [pool[i] for i in sorted(top)]


def extract_pages(path, ext):
    if ext == ".pdf":
        return [re.sub(r"\s+", " ", p.extract_text() or "").strip() for p in PdfReader(path).pages]
    if ext == ".docx":
        try:
            from docx import Document
        except ImportError:
            raise RuntimeError("Run: pip install python-docx")
        text = "\n".join(p.text for p in Document(path).paragraphs)
    else:
        text = open(path, encoding="utf-8", errors="ignore").read()
    parts = re.split(r"\[\[PAGE \d+\]\]", text)  # keeps real pages of text exported from PDFs
    if len(parts) > 2:
        return [re.sub(r"\s+", " ", p).strip() for p in parts[1:]]
    text = re.sub(r"\s+", " ", text).strip()
    return [text[i:i + 3000] for i in range(0, len(text), 3000)] or [""]


@app.post("/api/upload")
def upload():
    for f in request.files.getlist("files"):
        ext = os.path.splitext(f.filename)[1].lower()
        if ext not in ALLOWED:
            raise RuntimeError(f"{f.filename}: unsupported type. Use {', '.join(ALLOWED)}")
        did = uuid.uuid4().hex[:8]
        path = os.path.join(UP, did + ext)
        f.save(path)
        try:
            pages = extract_pages(path, ext)
        except Exception as e:
            os.remove(path)
            raise RuntimeError(f"Could not read {f.filename}: {e}")
        DOCS[did], FILES[did], PAGES[did] = f.filename, path, pages
        items = [{"doc_id": did, "doc": f.filename, "page": pn, "text": t[s:s + CHUNK]}
                 for pn, t in enumerate(pages, 1) for s in range(0, len(t), CHUNK)]
        if not items:
            STATUS[did] = {"done": 0, "total": 0, "error": "No extractable text (scanned file?)"}
            continue
        for c in items:
            c["tf"] = Counter(toks(c["text"]))
            c["len"] = max(1, sum(c["tf"].values()))
        CHUNKS.extend(items)
        STATUS[did] = {"done": len(items), "total": len(items)}
        print(f"[{f.filename}] indexed {len(items)} chunks (local, no API)", flush=True)
    return docs()


@app.delete("/api/docs/<did>")
def delete(did):
    global CHUNKS
    DOCS.pop(did, None)
    STATUS.pop(did, None)
    PAGES.pop(did, None)
    CHUNKS = [c for c in CHUNKS if c["doc_id"] != did]
    p = FILES.pop(did, None)
    if p and os.path.exists(p):
        os.remove(p)
    return docs()


@app.get("/api/text/<did>")
def text(did):
    return jsonify(name=DOCS.get(did, ""), pages=PAGES.get(did, []))


@app.errorhandler(Exception)
def err(e):
    return jsonify(error=str(e)), 500


@app.get("/pdf/<did>")
def original(did):
    return send_from_directory(UP, os.path.basename(FILES[did]))


@app.post("/api/chat")
def chat():
    if not KEY:
        return jsonify(error="GEMINI_API_KEY is not set."), 400
    data = request.json
    q, history, sel = data["question"], data.get("history", []), data.get("selected")
    pool = [c for c in CHUNKS if sel is None or c["doc_id"] in sel]
    pend = [d for d, st in STATUS.items() if d in DOCS and st["done"] < st["total"] and not st.get("error")]
    if not pool and pend:
        st = STATUS[pend[0]]
        return jsonify(error=f"Still indexing ({st['done']}/{st['total']}). Please wait until it finishes."), 400
    if not pool:
        return jsonify(error="No source selected, or the PDF has no extractable text."), 400
    t0 = time.time()
    picked = retrieve(q, pool)
    ctx = "\n\n".join(f"[{n}] ({c['doc']}, page {c['page']})\n{c['text']}"
                      for n, c in enumerate(picked, 1))
    sys = ("Answer using ONLY the numbered passages below. After every claim, cite the "
           "passage number like [1] or [2]. If the answer is not in the passages, say it was "
           "not found in the sources. ALWAYS answer in English unless the user's question is "
           "written in another language. Use clear headings and bullet points.\n\nPassages:\n" + ctx)
    contents = [{"role": "user" if m["role"] == "user" else "model",
                 "parts": [{"text": m["content"]}]} for m in history]
    contents.append({"role": "user", "parts": [{"text": q}]})
    j = generate({"systemInstruction": {"parts": [{"text": sys}]}, "contents": contents})
    try:
        raw = "".join(p.get("text", "") for p in j["candidates"][0]["content"]["parts"])
    except Exception:
        raise RuntimeError("Gemini returned no answer (possibly blocked).")
    ans = re.sub(r"\s*\[(\d+)\]", r" [\1]", raw)
    used = {int(n) for n in re.findall(r"\[(\d+)\]", ans) if 1 <= int(n) <= len(picked)}
    ans = re.sub(r" \[(\d+)\]", lambda m: m.group(0) if int(m.group(1)) in used else "", ans)
    cites = [{"n": n, "doc_id": picked[n - 1]["doc_id"], "doc": picked[n - 1]["doc"],
              "page": picked[n - 1]["page"], "text": picked[n - 1]["text"]} for n in sorted(used)]
    u = j.get("usageMetadata", {})
    return jsonify(answer=ans, citations=cites, usage={
        "input": u.get("promptTokenCount", 0), "output": u.get("candidatesTokenCount", 0),
        "time": round(time.time() - t0, 1)})


if __name__ == "__main__":
    app.run(port=5000, debug=False)
