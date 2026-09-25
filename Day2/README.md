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

An additional retrieval evaluation was also performed using **Recall@3, Precision@3, and MRR** to quantitatively validate how the different chunking strategies affected retrieval.

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
│   ├── compare_chunks.py
│   ├── create_eval_questions.py
│   └── retrieval_evaluation.py
│
└── outputs/
    ├── chunking_verdict.txt
    ├── clean_text.txt
    ├── fixed_chunks.txt
    ├── sentence_chunks.txt
    ├── paragraph_chunks.txt
    ├── semantic_chunks.txt
    ├── recursive_chunks.txt
    ├── chunking_verdict.txt
    ├── eval_questions.json
    └── retrieval_evaluation.txt
```

---

## Dataset

The **SQuAD v1.1** dataset was used for the experiment.

A cleaned text file was created from the dataset and used as the **common input for all five chunking strategies**.

This ensured that the comparison focused on the chunking methods rather than differences in the input data.

For the additional retrieval evaluation, 20 questions and their corresponding ground-truth answers were selected from the same dataset.

---

# Implemented Strategies

## 1. Fixed-size Chunking

Fixed-size chunking divides the text into chunks with a predefined word limit and overlap.

### Configuration

- Chunk size: 500 words
- Overlap: 50 words
- Total chunks: 7
- Average chunk size: 477 words

### Observation

It provides predictable chunk sizes but can split sentences and ideas across chunk boundaries. Overlap helps preserve some context between neighboring chunks but can also introduce duplicated information.

---

## 2. Sentence Chunking

Sentence chunking groups a fixed number of sentences into each chunk.

### Configuration

- 5 sentences per chunk
- Total chunks: 31
- Average chunk size: 98.81 words

### Observation

It preserves sentence boundaries and produces relatively focused chunks, but chunk sizes can vary because sentence lengths are different.

---

## 3. Paragraph Chunking

Paragraph chunking uses each paragraph as an individual chunk.

### Configuration

- One paragraph per chunk
- Total chunks: 20
- Average chunk size: 152.60 words

### Observation

It preserves the natural structure of the document and generally keeps related information together. However, chunk size depends completely on the original paragraph length.

---

## 4. Semantic Chunking

Semantic chunking uses embeddings to identify changes in the meaning of the text and create chunks based on semantic similarity.

### Configuration

- Embedding model: `all-MiniLM-L6-v2`
- Similarity threshold: 0.50
- Total chunks: 126
- Average chunk size: 25.06 words

### Observation

It considers meaning rather than only text position. However, the current configuration produced many very small chunks, so further threshold tuning and minimum chunk-size constraints would be useful.

---

## 5. Recursive Chunking

Recursive chunking attempts to split text hierarchically using natural boundaries.

The current hierarchy is:

```text
Paragraph → Sentence → Word
```

### Configuration

- Target chunk size: 500 words
- Overlap: 50 words
- Total chunks: 7
- Average chunk size: 434.14 words

### Observation

It provides a balance between natural text boundaries and controlled chunk size. The current implementation can still combine multiple paragraphs in a chunk, so the separators and chunk-size rules can be further refined.

---

# Chunking Comparison

| Strategy | Chunks | Avg. Size | Main Characteristic |
|---|---:|---:|---|
| Fixed-size | 7 | 477 words | Simple and predictable |
| Sentence | 31 | 98.81 words | Preserves sentence boundaries |
| Paragraph | 20 | 152.60 words | Preserves document structure |
| Semantic | 126 | 25.06 words | Meaning-based splitting |
| Recursive | 7 | 434.14 words | Balance of structure and size |

---

# Additional Retrieval Evaluation

After implementing and inspecting the chunking strategies, an additional retrieval evaluation was performed.

The purpose was to measure whether the generated chunks could successfully retrieve information relevant to a user query.

## Evaluation Setup

- Evaluation dataset: SQuAD v1.1
- Evaluation questions: 20
- Embedding model: `all-MiniLM-L6-v2`
- Retrieval method: Cosine similarity
- Top-K: 3
- Relevance criterion: Retrieved chunk contains the ground-truth answer

The **same questions, embedding model, and retrieval procedure** were used for all five strategies.

## Metrics

### Recall@3

Measures whether the relevant answer-containing chunk was retrieved within the top 3 results.

For this evaluation, there is one ground-truth answer per question, so Recall@3 effectively behaves as a **Hit@3** measure.

### Precision@3

Measures how many of the top 3 retrieved chunks were relevant.

### MRR

Mean Reciprocal Rank measures how high the first relevant chunk appears in the retrieved ranking.

---

## Retrieval Results

| Strategy | Chunks | Recall@3 | Precision@3 | MRR |
|---|---:|---:|---:|---:|
| Fixed-size | 7 | 0.9000 | 0.3333 | 0.8250 |
| Sentence | 31 | 0.9500 | **0.3833** | 0.8250 |
| Paragraph | 20 | **1.0000** | 0.3667 | **0.8500** |
| Semantic | 126 | 0.9000 | 0.3333 | 0.7917 |
| Recursive | 7 | 0.9500 | 0.3667 | 0.8250 |

---

## Retrieval Observations

### Fixed-size

Fixed-size chunking achieved a Recall@3 of 0.9000. It successfully retrieved the relevant answer for most questions, but fixed boundaries can split related information.

### Sentence

Sentence chunking achieved a Recall@3 of 0.9500 and the highest Precision@3 of 0.3833. Its sentence-level boundaries generally produce focused retrieval units.

### Paragraph

Paragraph chunking achieved the highest Recall@3 of 1.0000 and the highest MRR of 0.8500 in this evaluation. This indicates that the relevant information was retrieved within the top 3 results for all 20 evaluation questions.

### Semantic

Semantic chunking achieved a Recall@3 of 0.9000. The current configuration generated 126 chunks with an average size of only 25.06 words, which may have contributed to weaker retrieval performance. Further tuning could change these results.

### Recursive

Recursive chunking achieved a Recall@3 of 0.9500 and MRR of 0.8250. It maintained high retrieval coverage while using larger, controlled chunks.

---

# Retrieval Evaluation Limitations

The retrieval evaluation should be treated as an **additional validation**, not as a universal benchmark.

Limitations include:

1. Only 20 evaluation questions were used.
2. Relevance was defined using answer-string containment.
3. Recall@3 therefore represents whether an answer-containing chunk was retrieved in the top 3.
4. Only one embedding model was used.
5. Only Top-3 retrieval was evaluated.
6. Semantic chunking was evaluated with the current threshold and minimum-size configuration.
7. The evaluation dataset may not represent all types of documents used in RAG systems.

The complete quantitative evaluation is available in:

```text
outputs/retrieval_evaluation.txt
```

---

# Final Verdict

The detailed manual verdict for each chunking strategy is available in:

```text
outputs/chunking_verdict.txt
```

The verdict file contains:

- Configuration of each strategy
- Chunk statistics
- Strengths
- Limitations
- Overall assessment
- Final recommendation

The retrieval evaluation provides an additional quantitative perspective through Recall@3, Precision@3, and MRR.

---

# Recommendation

Based on the **current implementation, chunk inspection, chunk-size behavior, and practical RAG considerations**, **Recursive Chunking** is recommended as the starting strategy for the RAG use case.

It provides a practical balance between:

- Natural text boundaries
- Controlled chunk size
- Context preservation
- Implementation simplicity

The retrieval evaluation also showed that Recursive Chunking achieved a high Recall@3 of 0.9500.

Paragraph Chunking achieved the highest Recall@3 and MRR in the current retrieval experiment, while Sentence Chunking achieved the highest Precision@3. These quantitative results are useful for validation, but they are considered alongside the manual chunk-quality analysis rather than being used as the sole basis for the primary recommendation.

Semantic Chunking is also promising because it considers meaning rather than only text boundaries. However, the current implementation produces many very small chunks and would require further tuning before being considered as the primary approach.

---

# Conclusion

This experiment demonstrated that different chunking strategies produce significantly different chunk structures even when the same cleaned text is used.

The manual analysis showed differences in chunk size, boundaries, context preservation, and implementation complexity.

The additional retrieval evaluation showed that chunking strategy also affects retrieval behavior:

- Paragraph Chunking achieved the highest Recall@3 and MRR.
- Sentence Chunking achieved the highest Precision@3.
- Recursive Chunking achieved high retrieval coverage while maintaining controlled chunk sizes.
- Semantic Chunking requires further tuning in its current configuration.

The choice of chunking strategy should ultimately depend on the document structure, desired context size, retrieval requirements, and downstream RAG application.