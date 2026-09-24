# Week 8 — Day 2: Chunking Strategies for RAG

## Overview

The objective of this task was to explore and implement different **text chunking strategies** used in Retrieval-Augmented Generation (RAG) systems.

Five chunking approaches were implemented and compared using the **same cleaned dataset** to ensure a fair comparison:

1. Fixed-size Chunking
2. Sentence Chunking
3. Paragraph Chunking
4. Semantic Chunking
5. Recursive Chunking

The generated chunks were inspected based on their size, boundaries, structure, and overall suitability for a RAG pipeline.

---

## Project Structure

```text
Day2/
├── data/
│   └── train-v1.1.json
│
├── src/
│   ├── load_dataset.py
│   ├── fixed_chunking.py
│   ├── sentence_chunking.py
│   ├── paragraph_chunking.py
│   ├── semantic_chunking.py
│   ├── recursive_chunking.py
│   └── compare_chunks.py
│
└── outputs/
    ├── clean_text.txt
    ├── fixed_chunks.txt
    ├── sentence_chunks.txt
    ├── paragraph_chunks.txt
    ├── semantic_chunks.txt
    ├── recursive_chunks.txt
    └── chunking_verdict.txt
```

---

## Dataset

The SQuAD v1.1 dataset was used for the experiment.

A cleaned text file was created from the dataset and used as the **common input for all five chunking strategies**.

This ensured that the comparison focused on the chunking methods rather than differences in the input data.

---

## Implemented Strategies

### 1. Fixed-size Chunking

Fixed-size chunking divides the text into chunks with a predefined word limit and overlap.

**Configuration:**

* Chunk size: 500 words
* Overlap: 50 words
* Total chunks: 7
* Average chunk size: 477 words

**Observation:**

It provides predictable chunk sizes but can split sentences and ideas across chunk boundaries.

---

### 2. Sentence Chunking

Sentence chunking groups a fixed number of sentences into each chunk.

**Configuration:**

* 5 sentences per chunk
* Total chunks: 31
* Average chunk size: 98.81 words

**Observation:**

It preserves sentence boundaries and produces relatively focused chunks, but chunk sizes can vary because sentence lengths are different.

---

### 3. Paragraph Chunking

Paragraph chunking uses each paragraph as an individual chunk.

**Configuration:**

* One paragraph per chunk
* Total chunks: 20
* Average chunk size: 152.60 words

**Observation:**

It preserves the natural structure of the document and generally keeps related information together. However, chunk size depends completely on the original paragraph length.

---

### 4. Semantic Chunking

Semantic chunking uses embeddings to identify changes in the meaning of the text and create chunks based on semantic similarity.

**Configuration:**

* Embedding model: `all-MiniLM-L6-v2`
* Similarity threshold: 0.50
* Total chunks: 126
* Average chunk size: 25.06 words

**Observation:**

It considers meaning rather than only text position. However, the current configuration produced many very small chunks, so further threshold tuning and minimum chunk-size constraints would be useful.

---

### 5. Recursive Chunking

Recursive chunking attempts to split text hierarchically using natural boundaries.

The current hierarchy is:

```text
Paragraph → Sentence → Word
```

**Configuration:**

* Target chunk size: 500 words
* Overlap: 50 words
* Total chunks: 7
* Average chunk size: 434.14 words

**Observation:**

It provides a balance between natural text boundaries and controlled chunk size. The current implementation can still combine multiple paragraphs in a chunk, so the separators and chunk-size rules can be further refined.

---

## Comparison

| Strategy   | Chunks |    Avg. Size | Main Characteristic           |
| ---------- | -----: | -----------: | ----------------------------- |
| Fixed-size |      7 |    477 words | Simple and predictable        |
| Sentence   |     31 |  98.81 words | Preserves sentence boundaries |
| Paragraph  |     20 | 152.60 words | Preserves document structure  |
| Semantic   |    126 |  25.06 words | Meaning-based splitting       |
| Recursive  |      7 | 434.14 words | Balance of structure and size |

---

## Final Verdict

The detailed verdict for every strategy is available in:

```text
outputs/chunking_verdict.txt
```

The verdict file contains:

* Configuration of each strategy
* Chunk statistics
* Strengths
* Limitations
* Overall assessment
* Final recommendation

---

## Recommendation

Based on the current experiment, **Recursive Chunking** is recommended as the starting strategy for the RAG use case.

It provides a practical balance between:

* Natural text boundaries
* Controlled chunk size
* Context preservation
* Implementation simplicity

Semantic Chunking is also promising, but the current implementation produces very small chunks and would require further tuning before being used as the primary approach.

---

## Conclusion

This experiment demonstrated that different chunking strategies produce significantly different chunk structures even when the same cleaned text is used.

The choice of chunking strategy should depend on the document structure, desired context size, and requirements of the downstream RAG system.
