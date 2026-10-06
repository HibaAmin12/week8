"""Run with: EMBED_PROVIDER=fake pytest -q   (fake embedder and fake LLM, no network or models needed)"""
import io
import os
import tempfile

os.environ["EMBED_PROVIDER"] = "fake"
os.environ["DATA_DIR"] = tempfile.mkdtemp()
os.environ["EMBED_PRICE_PER_M"] = "0.02"   # pretend embeddings are billed, to test the maths
os.environ["OPENAI_API_KEY"] = "test"
os.environ["ALLOWED_EXT"] = ".pdf,.txt,.csv"   # the app defaults to PDF only

import pytest

from app import chunking, config, costs, llm, loaders
from app.main import app


def test_chunks_respect_size_and_overlap():
    text = " ".join(f"Sentence number {i} is here." for i in range(300))
    spans = chunking.split_text(text, 1200, 200)
    assert len(spans) > 3
    assert all(e - s <= 1200 for s, e in spans)
    assert spans[1][0] < spans[0][1]            # overlap
    assert spans[-1][1] == len(text)            # nothing lost


def test_chunk_text_is_substring_of_unit():
    units = [{"label": "Page 1", "text": "Alpha beta. " * 300, "heading": "H", "mergeable": False}]
    for c in chunking.make_chunks(units):
        assert c["text"] in units[0]["text"] and c["ctx"].startswith("H\n")


def test_small_sections_merge():
    u = [{"label": "Section: A", "text": "short", "heading": "A", "mergeable": True},
         {"label": "Section: B", "text": "x" * 500, "heading": "B", "mergeable": True}]
    assert len(chunking.merge_small(u)) == 1


def test_page_markers_and_csv(tmp_path):
    p = tmp_path / "a.txt"
    p.write_text("[[PAGE 1]] first page text [[PAGE 2]] second page text")
    assert [u["label"] for u in loaders.load(str(p))] == ["Page 1", "Page 2"]
    c = tmp_path / "b.csv"
    c.write_text("name,age\nAnn,30\nBob,41\n")
    text = loaders.load(str(c))[0]["text"]
    assert "name: Ann; age: 30" in text


def test_cost_math():
    assert costs.usd(1_000_000, 0.25) == 0.25
    assert round(costs.usd(2000, 2.0), 6) == 0.004


@pytest.fixture
def client(monkeypatch):
    def fake_chat(messages):
        return {"text": "ART is started early [1].", "input_tokens": 1000, "output_tokens": 200, "model": "fake"}
    monkeypatch.setattr(llm, "chat", fake_chat)
    return app.test_client()


def _upload(client, nid, body, name="doc.txt"):
    return client.post(f"/api/notebooks/{nid}/upload", data={"file": (io.BytesIO(body.encode()), name)},
                       content_type="multipart/form-data")


DOC = "[[PAGE 1]] Antiretroviral therapy should start early for all patients. [[PAGE 2]] Unrelated cooking notes."


def test_end_to_end_with_costs(client):
    nid = client.post("/api/notebooks", json={"name": "Test"}).json["id"]
    r = _upload(client, nid, DOC)
    assert r.status_code == 200, r.json
    assert r.json["usage"]["embed_tokens"] > 0 and r.json["usage"]["embed_cost"] > 0
    doc_id = r.json["doc"]["id"]

    r = client.post(f"/api/notebooks/{nid}/chat", json={"message": "When to start antiretroviral therapy?", "doc_ids": [doc_id]})
    assert r.status_code == 200, r.json
    u = r.json["usage"]
    assert u["input_tokens"] == 1000 and u["output_tokens"] == 200
    assert round(u["input_cost"], 8) == round(1000 / 1e6 * config.PRICE_IN, 8)
    assert round(u["output_cost"], 8) == round(200 / 1e6 * config.PRICE_OUT, 8)
    assert u["total_cost"] >= u["input_cost"] + u["output_cost"]
    assert r.json["citations"][0]["label"].startswith("Page 1")
    assert r.json["citations"][0]["segments"][0]["text"]
    assert r.json["totals"]["requests"] == 1

    assert client.get(f"/api/notebooks/{nid}/docs/{doc_id}/units").json[0]["label"] == "Page 1"
    assert client.delete(f"/api/notebooks/{nid}/docs/{doc_id}").json["deleted"] is True


def test_notebooks_are_separate_and_chat_is_saved(client):
    a = client.post("/api/notebooks", json={"name": "A"}).json["id"]
    b = client.post("/api/notebooks", json={"name": "B"}).json["id"]
    doc = _upload(client, a, DOC).json["doc"]["id"]
    client.post(f"/api/notebooks/{a}/chat", json={"message": "When to start therapy?", "doc_ids": [doc]})
    na, nb_ = client.get(f"/api/notebooks/{a}").json, client.get(f"/api/notebooks/{b}").json
    assert len(na["docs"]) == 1 and len(nb_["docs"]) == 0
    assert [m["role"] for m in na["messages"]] == ["user", "assistant"] and nb_["messages"] == []
    assert na["totals"]["total_cost"] > 0 and nb_["totals"]["total_cost"] == 0
    client.patch(f"/api/notebooks/{a}", json={"name": "Renamed"})
    assert client.get(f"/api/notebooks/{a}").json["name"] == "Renamed"
    assert client.delete(f"/api/notebooks/{b}").json["deleted"] is True
    assert client.get(f"/api/notebooks/{b}").status_code == 404


def test_reindex_after_embedding_change(tmp_path):
    import json
    from app.store import Store
    from app import chunking as ch, embeddings
    units = [{"label": "Page 1", "text": "Hello world. " * 20, "heading": "", "mergeable": False}]
    chunks = ch.make_chunks(units)
    vecs, _ = embeddings.embed_documents([c["ctx"] for c in chunks])
    s = Store(str(tmp_path))
    s.add("abc12345", "x.txt", units, chunks, vecs, ".txt")
    p = tmp_path / "abc12345" / "doc.json"
    d = json.loads(p.read_text())
    d["meta"]["embed_id"] = "local:other-model"
    p.write_text(json.dumps(d))
    s2 = Store(str(tmp_path))
    assert s2.stale == {"abc12345": "x.txt"} and s2.list() == []
    n, tokens, cost = s2.reindex()
    assert n == 1 and len(s2.list()) == 1 and not s2.stale


def test_bad_extension(client):
    nid = client.post("/api/notebooks", json={}).json["id"]
    assert _upload(client, nid, "x", "evil.exe").status_code == 400


def test_expand_includes_neighbour_on_next_page(tmp_path):
    from app.store import Store
    from app import chunking as ch, embeddings
    units = [{"label": "Page 7", "text": "Alpha beta gamma. " * 5, "heading": "", "mergeable": False},
             {"label": "Page 8", "text": "Delta epsilon zeta. " * 5, "heading": "", "mergeable": False}]
    chunks = ch.make_chunks(units)
    vecs, _ = embeddings.embed_documents([c["ctx"] for c in chunks])
    s = Store(str(tmp_path))
    s.add("d1", "x.pdf", units, chunks, vecs, ".pdf")
    hit = s.sample(set(), 10)[0]            # first chunk, on page 7
    segs = s.expand(hit, 1)
    assert [x["label"] for x in segs] == ["Page 7", "Page 8"]
    assert s.expand(hit, 0) == [{"unit": 0, "label": "Page 7", "text": units[0]["text"].strip()}]
