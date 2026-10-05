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


class Store:
    def __init__(self, root: str = None):
        self.root = root or config.DATA_DIR
        os.makedirs(self.root, exist_ok=True)
        self.lock = threading.RLock()
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
                print(f"[store] skipping {d['meta']['name']}: indexed with {d['meta'].get('embed_id')}, "
                      f"current is {embeddings.embed_id()}. Re-upload it.")
                continue
            d["vectors"] = np.load(os.path.join(p, "vectors.npy"))
            self.docs[doc_id] = d

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
        return {"doc_id": d["meta"]["id"], "doc": d["meta"]["name"], "label": c["label"],
                "unit": c["unit"], "text": c["text"], "score": float(score)}

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
