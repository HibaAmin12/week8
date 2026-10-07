"""Document store: chunks and vectors in memory, saved to DATA_DIR so restarts keep everything.

Layout on disk:  data/<doc_id>/doc.json  (units + chunks + metadata)
                 data/<doc_id>/vectors.npy
                 data/<doc_id>/original.<ext>
"""
import json
import os
import shutil
import threading

import numpy as np

from . import config, embeddings
from .costs import usd


class Store:
    def __init__(self, root: str = None):
        self.root = root or config.DATA_DIR
        os.makedirs(self.root, exist_ok=True)
        self.lock = threading.RLock()
        self.stale = {}  # doc_id -> name, indexed with a different embedding model
        self.docs = {}  # id -> {"meta": {...}, "units": [...], "chunks": [...], "vectors": ndarray}
        self._load_all()

    # ---------- persistence ----------
    def _dir(self, doc_id):
        return os.path.join(self.root, doc_id)

    def _load_all(self):
        for doc_id in sorted(os.listdir(self.root)):
            p = self._dir(doc_id)
            if not os.path.isfile(os.path.join(p, "doc.json")):
                continue
            with open(os.path.join(p, "doc.json")) as f:
                d = json.load(f)
            if d["meta"].get("embed_id") != embeddings.embed_id():
                self.stale[doc_id] = d["meta"]["name"]
                continue
            d["vectors"] = np.load(os.path.join(p, "vectors.npy"))
            self.docs[doc_id] = d

    def folder(self, doc_id):
        return self._dir(doc_id)

    def reindex(self):
        """Re-embed documents saved with another embedding model, using their stored chunks (no re-upload)."""
        done, tokens, cost = 0, 0, 0.0
        for doc_id in list(self.stale):
            p = self._dir(doc_id)
            with open(os.path.join(p, "doc.json")) as f:
                d = json.load(f)
            vecs, t = embeddings.embed_documents([c["ctx"] for c in d["chunks"]])
            c = usd(t, config.EMBED_PRICE_PER_M)
            d["meta"].update(embed_id=embeddings.embed_id(), embed_tokens=t, embed_cost=c)
            with open(os.path.join(p, "doc.json"), "w") as f:
                json.dump(d, f)
            np.save(os.path.join(p, "vectors.npy"), vecs)
            d["vectors"] = vecs
            with self.lock:
                self.docs[doc_id] = d
                del self.stale[doc_id]
            done, tokens, cost = done + 1, tokens + t, cost + c
        return done, tokens, cost

    def add(self, doc_id, name, units, chunks, vectors, ext, embed_tokens=0, embed_cost=0.0):
        meta = {"id": doc_id, "name": name, "ext": ext, "n_units": len(units), "n_chunks": len(chunks),
                "chars": sum(len(u["text"]) for u in units), "embed_id": embeddings.embed_id(),
                "embed_tokens": embed_tokens, "embed_cost": embed_cost}
        os.makedirs(self._dir(doc_id), exist_ok=True)
        with open(os.path.join(self._dir(doc_id), "doc.json"), "w") as f:
            json.dump({"meta": meta, "units": units, "chunks": chunks}, f)
        np.save(os.path.join(self._dir(doc_id), "vectors.npy"), vectors)
        with self.lock:
            self.docs[doc_id] = {"meta": meta, "units": units, "chunks": chunks, "vectors": vectors}
        return meta

    def delete(self, doc_id) -> bool:
        with self.lock:
            existed = self.docs.pop(doc_id, None) is not None
        shutil.rmtree(self._dir(doc_id), ignore_errors=True)
        return existed

    def original_path(self, doc_id):
        d = self._dir(doc_id)
        for f in os.listdir(d) if os.path.isdir(d) else []:
            if f.startswith("original."):
                return os.path.join(d, f)
        return None

    # ---------- queries ----------
    def list(self):
        with self.lock:
            return [d["meta"] for d in self.docs.values()]

    def units(self, doc_id):
        d = self.docs.get(doc_id)
        return None if d is None else d["units"]

    def _selected(self, doc_ids):
        with self.lock:
            return [d for i, d in self.docs.items() if not doc_ids or i in doc_ids]

    @staticmethod
    def _hit(d, idx, score=0.0):
        c = d["chunks"][idx]
        return {"doc_id": d["meta"]["id"], "doc": d["meta"]["name"], "idx": idx, "label": c["label"],
                "unit": c["unit"], "text": c["text"], "score": float(score)}

    @staticmethod
    def _span(unit_text, c):
        """Where a chunk sits inside its unit's text."""
        i = unit_text.find(c["text"], max(0, c["start"] - 10))
        if i < 0:
            i = max(unit_text.find(c["text"]), 0)
        return i, i + len(c["text"])

    def expand(self, hit, radius):
        """The hit plus `radius` chunks on each side, as segments [{unit, label, text}].

        Neighbours in the same page/section are merged into one segment; a neighbour on the next or
        previous page becomes its own segment, so an explanation split across pages stays complete.
        """
        d = self.docs[hit["doc_id"]]
        lo, hi = max(0, hit["idx"] - radius), min(len(d["chunks"]) - 1, hit["idx"] + radius)
        segs = []
        for j in range(lo, hi + 1):
            c = d["chunks"][j]
            a, b = self._span(d["units"][c["unit"]]["text"], c)
            if segs and segs[-1]["unit"] == c["unit"]:
                segs[-1]["b"] = max(segs[-1]["b"], b)
            else:
                segs.append({"unit": c["unit"], "label": c["label"], "a": a, "b": b})
        for sg in segs:
            sg["text"] = d["units"][sg["unit"]]["text"][sg.pop("a"):sg.pop("b")]
        return segs

    def search(self, qvec, doc_ids, k):
        """Top-k chunks by cosine similarity (vectors are normalised, so a dot product)."""
        docs = self._selected(doc_ids)
        scored = []
        for d in docs:
            sims = d["vectors"] @ qvec
            for idx in np.argsort(-sims)[:k]:
                scored.append((float(sims[idx]), d, int(idx)))
        scored.sort(key=lambda t: -t[0])
        return [self._hit(d, idx, s) for s, d, idx in scored[:k]]

    def sample(self, doc_ids, k):
        """Evenly spaced chunks across the selected documents, in reading order (for summaries)."""
        docs = self._selected(doc_ids)
        total = sum(len(d["chunks"]) for d in docs)
        if total == 0:
            return []
        out = []
        for d in docs:
            n = len(d["chunks"])
            share = max(1, round(k * n / total))
            step = n / min(share, n)
            out += [self._hit(d, int(i * step)) for i in range(min(share, n))]
        return out[:k + len(docs)]
