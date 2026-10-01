# Medical Guidelines — Fixed-Size Chunking Pipeline

## 📌 Project Overview

This project implements and evaluates **fixed-size chunking** for a Retrieval-Augmented Generation (RAG) system using **10 medical guideline PDFs**.

Large documents cannot be directly passed to an AI/RAG system as one huge piece of text. Therefore, the documents are divided into smaller **chunks**. These chunks can later be embedded, stored in a vector database, and retrieved when a user asks a question.

The main goal of this project is to determine:

> **Which fixed-size chunk size and overlap combination provides the best retrieval performance for medical guideline documents?**

The project evaluates **12 different fixed-size chunking configurations** using semantic retrieval and standard information-retrieval metrics.

---

# 🎯 Objectives

The main objectives are:

1. Extract text from multiple medical guideline PDFs.
2. Preserve page-level information for source traceability.
3. Save the extracted raw text.
4. Load a set of evaluation questions with known answer phrases.
5. Apply fixed-size chunking with different chunk sizes and overlaps.
6. Identify which chunks contain the correct answers.
7. Generate embeddings for questions and chunks.
8. Retrieve the top-5 most relevant chunks.
9. Evaluate retrieval performance using:
   - Recall@5
   - Precision@5
   - MRR
10. Inspect retrieved chunks manually for selected questions.
11. Determine a suitable fixed-size chunking configuration for the use case.

---

# 📚 Dataset

The project uses **10 medical guideline PDF documents**.

The PDFs are stored in:

```text
Data/
```

The notebook verifies:

- whether the `Data` directory exists
- the number of PDF files
- file names
- file sizes
- total dataset size

The dataset contains:

- **10 PDFs**
- approximately **22.8 MB** total PDF size
- **1308 pages**
- approximately **5.6 million extracted characters**

---

# 🔄 Overall Pipeline

The complete workflow is:

```text
Medical PDFs
     ↓
Locate and Validate PDFs
     ↓
Extract Text Page-by-Page
     ↓
Save Raw Text
     ↓
Load Evaluation Questions
     ↓
Generate Fixed-Size Chunks
     ↓
Save Chunks
     ↓
Identify Correct / Gold Chunks
     ↓
Generate Embeddings
     ↓
Semantic Top-5 Retrieval
     ↓
Calculate Recall@5
     ↓
Calculate Precision@5
     ↓
Calculate MRR
     ↓
Compare Chunking Configurations
     ↓
Inspect Top Settings
     ↓
Final Recommendation
```

---

# 🧩 Step 1 — Imports

The notebook imports the libraries required for the complete pipeline.

### Main libraries

| Library | Purpose |
|---|---|
| `json` | Read and save JSON data |
| `re` | Regular expressions and text processing |
| `Path` | File and directory handling |
| `PyMuPDF` | Extract text from PDFs |
| `pandas` | DataFrames and analysis |
| `tqdm` | Progress bars |
| `numpy` | Numerical calculations |
| `sentence-transformers` | Generate semantic embeddings |
| `bisect` | Locate page boundaries efficiently |
| `textwrap` | Format text previews |

---

# 📂 Step 2 — Locate the PDFs

The notebook checks the `Data` directory and discovers all PDF files.

It uses:

```python
Path("Data")
```

and recursively searches for PDF files.

The notebook also creates a DataFrame containing:

- file name
- file size in MB

This allows us to detect missing, empty, or unusually small files before processing.

---

# 📄 Step 3 — Extract Text Page-by-Page

PyMuPDF is used to extract text from every PDF.

Instead of combining everything immediately, each page is stored separately with:

```text
doc_name
page_num
text
```

For example:

```text
{
    "doc_name": "Doc1.pdf",
    "page_num": 4,
    "text": "..."
}
```

### Why preserve pages?

Page information is important for **source traceability**.

If a retrieved chunk contains the answer, we can identify the original PDF page from which that chunk came.

The notebook also checks:

- failed PDFs
- total pages
- total extracted characters
- pages containing almost no text

Pages with very little text may indicate scanned/image-based pages that could require OCR.

---

# 💾 Step 4 — Save Raw Text

The extracted text is saved as `.txt` files inside:

```text
raw_text/
```

Each PDF gets its own text file.

For example:

```text
Doc1.pdf → raw_text/Doc1.txt
Doc2.pdf → raw_text/Doc2.txt
```

Page markers are inserted:

```text
[[PAGE 1]]

page text...

[[PAGE 2]]

page text...
```

This is important because page boundaries can later be mapped back to chunks.

The text is saved using UTF-8 encoding so medical symbols such as:

```text
µ
≥
°
```

are preserved.

---

# ❓ Step 5 — Load Evaluation Questions

The notebook uses:

```text
questions.json
```

to evaluate retrieval quality.

There are **16 evaluation questions**.

Each question contains information such as:

```text
question
doc
answer_phrase
qid
```

Example conceptually:

```json
{
    "qid": "Q01",
    "doc": "Doc1.pdf",
    "question": "What is the WHO target blood pressure goal?",
    "answer_phrase": "target blood pressure treatment goal of <140/90 mmHg"
}
```

---

# 🏆 Gold / Correct Chunks

A chunk is considered a **correct chunk** when:

1. It belongs to the correct document.
2. It contains the complete `answer_phrase`.

This gives us a ground-truth reference for evaluating retrieval.

Importantly, the notebook does **not** permanently store a chunk ID as ground truth because chunk IDs change when chunk size or overlap changes.

Instead, it stores the answer phrase and finds the correct chunks separately for every configuration.

---

# ✂️ Step 6 — Fixed-Size Chunking

The notebook evaluates **12 different configurations**.

There are:

- 4 chunk sizes
- 3 overlap levels

### Chunk sizes

```text
256
500
1000
2000
```

### Overlap

```text
0%
10%
20%
```

Therefore:

```text
4 × 3 = 12 configurations
```

---

## Chunking Configurations

| Chunk Size | 0% Overlap | 10% Overlap | 20% Overlap |
|---:|---:|---:|---:|
| 256 | 0 | 26 | 51 |
| 500 | 0 | 50 | 100 |
| 1000 | 0 | 100 | 200 |
| 2000 | 0 | 200 | 400 |

The configuration naming convention is:

```text
S<size>_O<overlap>
```

For example:

```text
S1000_O200
```

means:

```text
Chunk size = 1000 characters
Overlap = 200 characters
```

---

# 🔁 How Fixed-Size Chunking Works

The chunking algorithm uses a sliding window.

For example:

```text
chunk size = 1000
overlap = 200
```

Then:

```text
Chunk 1 → characters 0–1000
Chunk 2 → characters 800–1800
Chunk 3 → characters 1600–2600
```

The movement between chunks is:

```text
step = chunk_size - overlap
```

Therefore:

```text
step = 1000 - 200
     = 800
```

Overlap allows some context from the previous chunk to appear in the next chunk.

---

# ⚠️ Fixed-Size Chunking Limitation

Fixed-size chunking does not understand the meaning or structure of the document.

Therefore, it can split:

- words
- sentences
- paragraphs
- tables

in the middle.

For example:

```text
Chunk 1:
"The recommended treatment should be continued for..."

Chunk 2:
"...at least six months unless..."
```

The sentence has been divided between two chunks.

This is one of the main limitations of fixed-size chunking.

---

# 📊 Chunk Statistics

The notebook generates chunks for all 12 configurations and records:

- chunk size
- overlap
- total number of chunks
- average chunk length

As overlap increases, more chunks are created because neighboring chunks contain repeated text.

For example, the notebook generated approximately:

```text
S256_O0       → 21,897 chunks
S256_O26      → 24,372 chunks
S256_O51      → 27,339 chunks

S500_O0       → 11,230 chunks
S500_O50      → 12,477 chunks
S500_O100     → 14,038 chunks

S1000_O0      → 5,622 chunks
S1000_O100    → 6,244 chunks
S1000_O200    → 7,024 chunks

S2000_O0      → 2,814 chunks
S2000_O200    → 3,123 chunks
S2000_O400    → 3,513 chunks
```

---

# 💾 Step 6b — Save Chunks

All generated chunks are saved inside:

```text
chunks/
```

Each configuration gets a separate text file.

For example:

```text
chunks/
├── S256_O0.txt
├── S256_O26.txt
├── S256_O51.txt
├── S500_O0.txt
├── S500_O50.txt
├── S500_O100.txt
├── S1000_O0.txt
├── S1000_O100.txt
├── S1000_O200.txt
├── S2000_O0.txt
├── S2000_O200.txt
└── S2000_O400.txt
```

Each chunk contains metadata such as:

```text
chunk ID
character range
page range
chunk text
```

This makes the chunks inspectable and traceable.

---

# 🎯 Identify Correct Chunks

For every question and every chunking configuration, the notebook searches for the **gold chunks**.

A chunk is correct only when:

```text
correct document
        AND
complete answer phrase exists
```

The results are stored in:

```text
correct_chunks.json
```

This step is important because some chunk configurations can split an answer phrase across chunk boundaries.

If the complete answer phrase is not present in any chunk, that configuration cannot retrieve that answer correctly using the current ground-truth definition.

---

# 🔎 Step 7 — Semantic Retrieval Evaluation

The notebook evaluates retrieval using:

```text
BAAI/bge-small-en-v1.5
```

This is a sentence-transformer embedding model.

The model converts:

```text
Question → vector
Chunk → vector
```

Then the notebook compares the vectors using cosine similarity.

The five highest-scoring chunks are retrieved for every question.

This is called:

```text
Top-5 Retrieval
```

---

# 📐 Evaluation Metrics

Three metrics are used.

## 1. Recall@5

Recall@5 measures how many of the correct chunks were retrieved within the top 5.

Formula:

```text
Recall@5 =
number of correct chunks retrieved
------------------------------------
total number of correct chunks
```

Example:

```text
Correct chunks = 2
Correct chunks found in top 5 = 1

Recall@5 = 1 / 2
         = 0.5
```

---

# 2. Precision@5

Precision@5 measures how many of the five retrieved chunks are actually correct.

Formula:

```text
Precision@5 =
number of correct chunks retrieved
------------------------------------
5
```

Example:

```text
2 correct chunks in top 5

Precision@5 = 2 / 5
             = 0.4
```

---

# 3. MRR — Mean Reciprocal Rank

MRR focuses on the position of the **first correct result**.

For one question:

```text
Rank 1 → RR = 1/1 = 1.0
Rank 2 → RR = 1/2 = 0.5
Rank 3 → RR = 1/3 = 0.333
Rank 4 → RR = 1/4 = 0.25
Rank 5 → RR = 1/5 = 0.2
Not found → RR = 0
```

Then:

```text
MRR = average of all reciprocal ranks
```

A higher MRR means the first relevant chunk tends to appear earlier in the retrieval results.

---

# ⚙️ Retrieval Configurations Tested

Five configurations were selected for semantic retrieval because they did not lose the answer phrase for any of the 16 questions:

```text
S500_O100
S1000_O100
S1000_O200
S2000_O200
S2000_O400
```

The embeddings are cached in:

```text
embeddings/
```

so they do not have to be regenerated every time the notebook is executed.

---

# 📊 Retrieval Results

The notebook produced the following results:

| Setting | Total Chunks | Recall@5 | Precision@5 | MRR |
|---|---:|---:|---:|---:|
| S2000_O400 | 3513 | 0.812 | 0.238 | 0.747 |
| S2000_O200 | 3123 | 0.844 | 0.213 | 0.609 |
| S500_O100 | 14038 | 0.656 | 0.163 | 0.575 |
| S1000_O200 | 7024 | 0.812 | 0.225 | 0.574 |
| S1000_O100 | 6244 | 0.719 | 0.163 | 0.573 |

The notebook's selection rule is:

```text
MRR
↓
Recall@5
↓
Precision@5
↓
fewer chunks
```

Under that rule, the recorded top configuration is:

```text
S2000_O400
```

with:

```text
MRR       = 0.747
Recall@5  = 0.812
Precision@5 = 0.238
```

These results are specific to this dataset, question set, embedding model, and evaluation setup.

---

# 🔬 Step 8 — Manual Inspection

The final section performs a more detailed inspection of retrieved results.

It examines:

- retrieved top-5 chunks
- similarity scores
- source document
- source pages
- whether the chunk is correct
- answer phrase location
- Recall@5
- Precision@5
- reciprocal rank

The purpose is to complement numerical metrics with **human inspection**.

This is useful because a metric alone may not explain *why* a particular chunking configuration performs differently.

---

# 📁 Project Output Structure

After running the notebook, the project contains files/folders similar to:

```text
Med_Fixed_chunking/
│
├── Data/
│   ├── Doc1.pdf
│   ├── Doc2.pdf
│   ├── ...
│   └── Doc10.pdf
│
├── raw_text/
│   ├── Doc1.txt
│   ├── Doc2.txt
│   └── ...
│
├── chunks/
│   ├── S256_O0.txt
│   ├── S256_O26.txt
│   ├── ...
│   └── S2000_O400.txt
│
├── embeddings/
│   ├── S500_O100.npy
│   ├── S1000_O100.npy
│   ├── ...
│   └── S2000_O400.npy
│
├── questions.json
├── correct_chunks.json
├── retrieval_results.csv
└── Med_fixedchunking.ipynb
```

---

# 🧠 Key Concepts Learned

This project demonstrates the complete retrieval-side workflow of a basic RAG system:

```text
Documents
   ↓
Text Extraction
   ↓
Chunking
   ↓
Ground Truth
   ↓
Embeddings
   ↓
Semantic Similarity
   ↓
Top-K Retrieval
   ↓
Evaluation
```

Important concepts include:

- Fixed-size chunking
- Chunk overlap
- Sliding windows
- Page-level metadata
- Ground-truth chunks
- Embeddings
- Semantic similarity
- Cosine similarity
- Top-K retrieval
- Recall@K
- Precision@K
- Reciprocal Rank
- Mean Reciprocal Rank (MRR)
- Retrieval evaluation

---

# ⚠️ Important Limitations

The results should not be treated as universally optimal.

The evaluation has several limitations:

### 1. Small evaluation set

Only:

```text
16 questions
```

are used.

Therefore, small differences between configurations may not generalize to a larger question set.

### 2. Fixed-size chunking

The algorithm does not understand semantic boundaries and may split sentences or paragraphs.

### 3. Embedding model limitation

The evaluation depends on:

```text
BAAI/bge-small-en-v1.5
```

A different embedding model may produce different retrieval results.

### 4. Character-based chunk sizes

Chunk sizes are measured in **characters**, not tokens.

Therefore:

```text
2000 characters
```

does not mean:

```text
2000 tokens
```

### 5. Embedding model context limit

The notebook sets:

```python
model.max_seq_length = 512
```

Therefore, very long chunks may be truncated by the embedding model.

### 6. Ground-truth definition

A chunk is considered correct only if it contains the **complete answer phrase**.

This is useful for reproducible evaluation, but it may not capture every chunk that is genuinely useful for answering a question.

---

# 🏁 Conclusion

This project evaluates fixed-size chunking systematically rather than choosing a chunk size arbitrarily.

The notebook compares multiple combinations of:

```text
chunk size
+
overlap
```

and evaluates their retrieval performance using:

```text
Recall@5
Precision@5
MRR
```

For this particular dataset and evaluation setup, the recorded retrieval comparison selected:

```text
S2000_O400
```

based primarily on the highest MRR, while **S2000_O200 achieved the highest Recall@5** among the tested configurations.

Therefore, the final choice depends on the retrieval objective being prioritized. The notebook uses MRR as its primary selection criterion, while the results and manual inspection should be considered together before deploying the chunking strategy in a real RAG system.