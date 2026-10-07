# NotebookRAG

Upload documents, ask questions, get answers with clickable citations (file + page/slide/section),
and see **every cost**: embedding, rerank, LLM input, LLM output, per answer and for the whole session.

## Run with Docker

```bash
cp .env.example .env        # then put your key after OPENAI_API_KEY=
docker compose up --build
```
Open http://localhost:5000. The first build downloads two small models (about 150 MB) into the image;
after that the container embeds and reranks offline. Only answers call OpenAI.

The OpenAI API is billed separately from ChatGPT: add credit at platform.openai.com.

## Run without Docker

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # OCR also needs the tesseract program installed
cp .env.example .env && export $(grep -v '^#' .env | xargs)
python -m app.main
```

## Notebooks

The dropdown next to the title lists all your notebooks and has **+ Create notebook**, rename and delete.
Each notebook has its own sources, saved chat, notes and cost totals, and the app reopens the last one you used.
If you change the embedding provider, sources indexed with the old one are hidden until you click
**Re-index now** (no re-upload; with OpenAI embeddings this costs a fraction of a cent per PDF).

## How it works

| Step | File | What happens |
|---|---|---|
| Load | `app/loaders.py` | PDF (OCR for scanned pages), DOCX, PPTX, XLSX, CSV, MD, TXT, JSON, PNG/JPG become *units*: page, heading section, slide, or 25-row group |
| Chunk | `app/chunking.py` | Each unit is cut into ~1200 char chunks with 200 overlap, on sentence ends, never across a unit boundary |
| Embed | `app/embeddings.py` | `BAAI/bge-small-en-v1.5` locally (or OpenAI embeddings). Heading is prepended for embedding only |
| Store | `app/store.py` | Vectors in memory, saved to the `/data` volume |
| Retrieve | `app/rag.py` | Top 20 by cosine similarity, then cross-encoder rerank to the best 8. Summary/quiz/FAQ questions sample 24 chunks evenly |
| Answer | `app/llm.py` | OpenAI chat with numbered passages; the model cites `[n]`, the server maps `n` back to file and page |
| Cost | `app/costs.py` | Exact token counts from OpenAI for input/output; saved totals in `data/usage.json` |

## Costs shown in the UI

* **Embedding**: local model = $0.00 (tokens are an estimate, about 4 chars/token). With `EMBED_PROVIDER=openai`
  the real token count comes from the API and `EMBED_PRICE_PER_M` is applied.
* **Rerank**: local, $0.00 (`RERANK_PRICE_PER_M` if you ever swap in a paid reranker).
* **LLM input / output**: exact tokens from OpenAI times `PRICE_IN` / `PRICE_OUT`.
* **Prices are settings, not looked up.** Check platform.openai.com/docs/pricing for your model and set
  `PRICE_IN`, `PRICE_OUT` in `.env`; otherwise the dollar figures will be wrong for a different model.
* Reasoning models bill their hidden thinking tokens as output, so output cost can exceed the visible answer length.

## Evaluate retrieval

```bash
python -m eval.evaluate --docs mydoc.pdf --dataset eval/sample_dataset.jsonl
```
Prints Hit@k and MRR for vector-only vs vector + rerank. Write your own questions in the dataset file.

## Tests

```bash
EMBED_PROVIDER=fake pytest -q
```

## Limits

* `bge-small-en-v1.5` is English only. For other languages set `EMBED_MODEL` to a multilingual model supported
  by fastembed and `OCR_LANG` (e.g. `urd`, plus the matching tesseract language package).
* After changing chunk settings or the embedding model, re-upload files (old indexes are skipped).
* One process, one shared notebook, no user accounts. Document text is sent to OpenAI when answering.
* Summary-style questions see 24 evenly spaced chunks, not the whole document.
