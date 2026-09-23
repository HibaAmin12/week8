# RAG Basics — Day 1

## Overview

This project covers the fundamental building blocks of **Retrieval-Augmented Generation (RAG)** through practical experiments.

The main focus was on:

* Document extraction and preprocessing
* Token-based chunking
* Chunk overlap and context preservation
* Text embeddings
* Cosine similarity
* Vector databases with Qdrant
* Semantic retrieval
* Exact-token retrieval limitations
* Precision@3 and Recall@3
* The "Lost in the Middle" problem

The experiments use the research paper **"Lost in the Middle: How Language Models Use Long Contexts"** as the main document.

---

## Project Structure

```text
Day1/
├── data/
│   └── Lost in the Middle.pdf
│
├── outputs/
│   ├── extracted_text.txt
│   ├── chunks_400_no_overlap.txt
│   └── chunks_400_overlap.txt
│
├── src/
│   ├── chunk_comparison.py
│   ├── inspect_boundaries.py
│   ├── cosine_manual.py
│   ├── qdrant_demo.py
│   └── retrieve.py
│
└── requirements.txt
```

---

# 1. Document Extraction

The PDF was converted into cleaned text before performing RAG-related experiments.

### Extraction Results

| Metric     |  Value |
| ---------- | -----: |
| Characters | 64,831 |
| Words      |  9,484 |
| Tokens     | 16,048 |

The extracted text was saved as:

```text
outputs/extracted_text.txt
```

---

# 2. Chunking Experiment

Chunking divides a long document into smaller pieces so that relevant sections can later be retrieved.

Two fixed-size chunking strategies were compared.

### Strategy 1 — No Overlap

```text
Chunk size = 400 tokens
Overlap = 0 tokens
```

Result:

```text
41 chunks
```

### Strategy 2 — With Overlap

```text
Chunk size = 400 tokens
Overlap = 60 tokens
```

Result:

```text
48 chunks
```

The 60-token overlap represents approximately **15% of the chunk size**.

The overlap causes some content to be repeated between neighboring chunks, increasing the total number of chunks but helping preserve context around boundaries.

---

# 3. Awkward Chunk Boundary

An actual awkward cut was identified in the no-overlap chunking strategy.

The sentence was split between two chunks:

### End of Chunk 4

```text
...which requires models to find relevant information within an input
```

### Start of Next Chunk

```text
context and use it to answer the question.
```

The complete sentence is:

```text
...requires models to find relevant information within an input context and use it to answer the question.
```

This demonstrates how fixed-size chunking can split a sentence or idea across chunk boundaries.

### Effect of Overlap

Using a 60-token overlap allows part of the previous context to appear again in the next chunk. This reduces the risk that important sentence-level context will be lost during retrieval.

### Finding

> Chunk overlap helps preserve contextual continuity when a sentence or idea crosses a chunk boundary, although it increases the total number of chunks.

---

# 4. Text Embeddings

For semantic representation, the following Sentence Transformer model was used:

```text
all-MiniLM-L6-v2
```

Three sentences were converted into numerical vectors.

### Embedding Results

```text
Number of texts: 3
Embedding shape: (3, 384)
Embedding dimensions: 384
```

Therefore, each sentence was represented by a **384-dimensional vector**.

---

# 5. Manual Cosine Similarity

Cosine similarity was implemented manually using NumPy rather than relying on a ready-made similarity function.

The formula is:

```text
cosine similarity =
dot(A, B) / (||A|| × ||B||)
```

The three test sentences were:

```text
1. Language models use context to answer questions.

2. Retrieval helps language models find relevant information.

3. The weather is sunny today.
```

### Results

| Sentence Pair            | Cosine Similarity |
| ------------------------ | ----------------: |
| Sentence 1 vs Sentence 2 |            0.5619 |
| Sentence 1 vs Sentence 3 |            0.0196 |
| Sentence 2 vs Sentence 3 |           -0.0355 |

### Interpretation

Sentence 1 and Sentence 2 have a similarity of **0.5619**, indicating that their meanings are relatively related.

Sentence 1 and Sentence 3 have a similarity of **0.0196**, indicating very low semantic similarity.

Sentence 2 and Sentence 3 have a similarity of **-0.0355**, which is also very low similarity.

This experiment demonstrates how embeddings allow semantically related text to be compared using vector similarity.

---

# 6. Qdrant Vector Database

A small in-memory Qdrant collection was created to demonstrate vector storage and retrieval.

### Configuration

```text
Embedding model: all-MiniLM-L6-v2
Vector dimension: 384
Distance metric: Cosine
Database: Qdrant
Storage: In-memory
```

A corpus of **10 real sentences** related to the Lost in the Middle study was embedded and stored.

Each Qdrant point contained:

```text
vector
text payload
document_id payload
```

### Storage Result

```text
Number of documents: 10
Embedding shape: (10, 384)
Stored vectors: 10
```

---

# 7. Semantic Retrieval

A query was used to search the Qdrant collection:

```text
How does the position of relevant information affect language model performance?
```

### Top 3 Results

| Rank | Document ID | Similarity |
| ---- | ----------: | ---------: |
| 1    |           0 |     0.7482 |
| 2    |           6 |     0.6661 |
| 3    |           1 |     0.6503 |

The top result was:

```text
Language models perform better when relevant information appears near the beginning or end of the context.
```

The retrieved result directly addresses the query, demonstrating that semantic vector search can identify relevant information based on meaning rather than requiring an exact keyword match.

---

# 8. Exact-ID Retrieval Experiment

A synthetic sentence containing an exact identifier was added:

```text
Ticket REF-4471 was resolved by rotating the API key.
```

The query was:

```text
REF-4471
```

Pure vector search returned the correct sentence at:

```text
Rank: 1
Similarity score: 0.3412
```

### Finding

The exact identifier was successfully retrieved in this small experiment. However, its similarity score was relatively low compared with the semantic retrieval example.

This demonstrates why **keyword or hybrid search can be useful for exact identifiers**, such as:

```text
REF-4471
ticket IDs
order numbers
product codes
UUIDs
error codes
```

A hybrid retrieval system can combine:

```text
Semantic Search
      +
Keyword / BM25 Search
      ↓
Better retrieval for both meaning and exact tokens
```

---

# 9. Precision@3 and Recall@3

Three queries were evaluated using manually selected relevant chunks.

> The relevance labels were manually selected for this small experiment and are not intended to represent an exhaustive benchmark annotation.

### Query 1

```text
How does the position of relevant information affect language model performance?
```

Retrieved chunks:

```text
[2, 28, 5]
```

Result:

```text
Precision@3 = 1.0000
Recall@3 = 1.0000
```

All three retrieved chunks were considered relevant according to the manual relevance labels.

---

### Query 2

```text
What tasks are used to evaluate language models with long contexts?
```

Retrieved chunks:

```text
[0, 43, 42]
```

Result:

```text
Precision@3 = 0.3333
Recall@3 = 0.3333
```

Only the first retrieved chunk directly matched the manually selected relevant set.

This shows that semantic retrieval can vary depending on the wording and specificity of the query.

---

### Query 3

```text
What happens when relevant information is placed in the middle of a long context?
```

Retrieved chunks:

```text
[2, 5, 0]
```

Result:

```text
Precision@3 = 1.0000
Recall@3 = 1.0000
```

All three retrieved chunks were considered relevant according to the manual labels.

---

# 10. Overall Retrieval Finding

Across the three queries:

```text
Average Precision@3 = 0.7778
Average Recall@3 = 0.7778
```

The experiment shows that semantic retrieval returned highly relevant chunks for some queries, while retrieval quality varied for other queries.

Because the relevance labels were manually selected for a very small corpus, these values should be interpreted as a **small practical demonstration rather than a rigorous retrieval benchmark**.

---

# 11. Lost in the Middle

The main paper used in this exercise investigates how the position of relevant information affects language model performance in long contexts.

A key observation is a **U-shaped performance pattern**:

```text
Performance
    ↑
    │ ●                       ●
    │  ●                     ●
    │   ●                   ●
    │    ●                 ●
    │       ●           ●
    │          ● ● ●
    └──────────────────────────→
      Beginning  Middle  End
```

Language models tend to perform better when relevant information appears near the **beginning or end** of the context, while performance can degrade when relevant information is located in the middle.

This is one reason why RAG systems should retrieve and provide focused relevant context instead of unnecessarily passing a large amount of unrelated information to the model.

---

# 12. Key Learnings

### RAG Pipeline

The basic RAG workflow can be summarized as:

```text
Documents
    ↓
Load
    ↓
Chunk
    ↓
Embed
    ↓
Store in Vector Database
    ↓
User Query
    ↓
Embed Query
    ↓
Vector Search
    ↓
Retrieve Relevant Chunks
    ↓
Optional Reranking
    ↓
Prompt LLM with Retrieved Context
    ↓
Grounded Answer
```

### Main concepts learned

* Documents need to be chunked before efficient retrieval.
* Chunk size affects retrieval granularity.
* Chunk overlap helps preserve boundary context.
* Embeddings convert text into numerical vector representations.
* Cosine similarity measures semantic closeness between vectors.
* Vector databases allow efficient storage and retrieval of embeddings.
* Semantic search is useful for meaning-based queries.
* Keyword search is useful for exact identifiers and tokens.
* Hybrid retrieval can combine semantic and keyword search.
* Retrieve-then-rerank can improve retrieval quality.
* Precision@k measures how many retrieved results are relevant.
* Recall@k measures how much of the relevant information was retrieved.
* Long contexts can suffer from the "Lost in the Middle" effect.

---

# 13. Conclusion

This Day 1 practical demonstrated the core retrieval components required for a basic RAG system.

The experiments progressed from **raw document → chunks → embeddings → cosine similarity → Qdrant vector storage → semantic retrieval → retrieval evaluation**.

The results also demonstrated two important retrieval considerations:

1. **Chunk overlap** helps preserve context across chunk boundaries.
2. **Semantic search alone may not be sufficient for exact-token queries**, making keyword or hybrid retrieval useful in practical RAG systems.

These experiments provide the foundation for building a complete RAG pipeline with retrieval and generation in later tasks.
