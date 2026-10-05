"""Flask API + static UI.

POST   /api/upload          multipart file -> indexed document + embedding cost
GET    /api/docs            list documents
GET    /api/docs/<id>/units pages/sections of a document (for the source viewer)
DELETE /api/docs/<id>       remove a document
POST   /api/chat            {message, history, doc_ids} -> answer + citations + cost
GET    /api/usage           session totals (also POST /api/usage/reset)
GET    /health
"""
import os
import uuid

from flask import Flask, jsonify, request
from werkzeug.utils import secure_filename

from . import chunking, config, embeddings, llm, loaders, rag
from .costs import Ledger, usd
from .store import Store

app = Flask(__name__, static_folder="static", static_url_path="")
app.config["MAX_CONTENT_LENGTH"] = config.MAX_UPLOAD_MB * 1024 * 1024

store = Store()
ledger = Ledger(os.path.join(config.DATA_DIR, "usage.json"))


@app.get("/")
def index():
    return app.send_static_file("index.html")


@app.get("/health")
def health():
    return {"status": "ok", "docs": len(store.list()), "embed": embeddings.embed_id(),
            "model": config.OPENAI_MODEL, "key_set": bool(config.OPENAI_API_KEY)}


@app.get("/api/docs")
def docs():
    return jsonify(store.list())


@app.get("/api/docs/<doc_id>/units")
def units(doc_id):
    u = store.units(doc_id)
    return (jsonify(u), 200) if u is not None else (jsonify(error="Document not found"), 404)


@app.delete("/api/docs/<doc_id>")
def delete(doc_id):
    return jsonify(deleted=store.delete(doc_id))


@app.post("/api/upload")
def upload():
    f = request.files.get("file")
    if f is None or not f.filename:
        return jsonify(error="No file received."), 400
    name = secure_filename(f.filename) or "file"
    ext = os.path.splitext(name)[1].lower()
    if ext not in config.ALLOWED_EXT:
        return jsonify(error=f"{ext or 'This'} files are not supported. Use: {', '.join(sorted(config.ALLOWED_EXT))}"), 400

    doc_id = uuid.uuid4().hex[:8]
    folder = os.path.join(config.DATA_DIR, doc_id)
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, "original" + ext)
    f.save(path)
    try:
        units_ = chunking.merge_small(loaders.load(path))
        chunks = chunking.make_chunks(units_)
        if not chunks:
            raise ValueError("No readable text found in this file (scanned file with OCR disabled or unavailable?).")
        vectors, tokens = embeddings.embed_documents([c["ctx"] for c in chunks])
        cost = usd(tokens, config.EMBED_PRICE_PER_M)
        meta = store.add(doc_id, name, units_, chunks, vectors, ext, tokens, cost)
        totals = ledger.add(embed_tokens=tokens, embed_cost=cost)
    except Exception as exc:
        store.delete(doc_id)
        return jsonify(error=f"Could not index {name}: {exc}"), 422
    return jsonify(doc=meta, usage={"embed_tokens": tokens, "embed_cost": cost,
                                    "provider": embeddings.embed_id()}, totals=totals)


@app.post("/api/chat")
def chat():
    body = request.get_json(silent=True) or {}
    message = (body.get("message") or "").strip()
    if not message:
        return jsonify(error="Type a question first."), 400
    try:
        out = rag.answer(store, message, set(body.get("doc_ids") or []), body.get("history") or [])
    except llm.LLMError as exc:
        return jsonify(error=str(exc)), 502
    except Exception as exc:
        return jsonify(error=f"Unexpected error: {exc}"), 500
    u = out["usage"]
    out["totals"] = ledger.add(embed_tokens=u["embed_tokens"], embed_cost=u["embed_cost"],
                               rerank_tokens=u["rerank_tokens"], rerank_cost=u["rerank_cost"],
                               input_tokens=u["input_tokens"], input_cost=u["input_cost"],
                               output_tokens=u["output_tokens"], output_cost=u["output_cost"], requests=1)
    return jsonify(out)


@app.get("/api/usage")
def usage():
    return jsonify(ledger.snapshot())


@app.post("/api/usage/reset")
def usage_reset():
    return jsonify(ledger.reset())


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "5000")))
