import json
import re
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer


QUESTIONS_PATH = Path("Day2/outputs/eval_questions.json")
OUTPUT_PATH = Path("Day2/outputs/retrieval_evaluation.txt")

CHUNK_FILES = {
    "Fixed-size": "Day2/outputs/fixed_chunks.txt",
    "Sentence": "Day2/outputs/sentence_chunks.txt",
    "Paragraph": "Day2/outputs/paragraph_chunks.txt",
    "Semantic": "Day2/outputs/semantic_chunks.txt",
    "Recursive": "Day2/outputs/recursive_chunks.txt",
}

TOP_K = 3


def normalize(text):
    """Normalize text for answer matching."""
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def load_chunks(file_path):
    """Load chunks from the generated chunk file."""
    text = Path(file_path).read_text(encoding="utf-8")

    parts = re.split(r"CHUNK\s+\d+", text)

    chunks = []

    for part in parts[1:]:
        chunk = part.strip()

        if chunk:
            chunk = re.sub(r"^=+\s*", "", chunk)
            chunk = re.sub(r"^=+\s*$", "", chunk).strip()

            if chunk:
                chunks.append(chunk)

    return chunks


def calculate_metrics(chunks, questions, model):
    """Calculate Recall@K, Precision@K and MRR."""

    chunk_embeddings = model.encode(
        chunks,
        normalize_embeddings=True,
        show_progress_bar=False
    )

    question_embeddings = model.encode(
        [item["question"] for item in questions],
        normalize_embeddings=True,
        show_progress_bar=False
    )

    recall_hits = 0
    precision_scores = []
    reciprocal_ranks = []

    for i, item in enumerate(questions):

        question_embedding = question_embeddings[i]

        # Cosine similarity because embeddings are normalized
        similarities = np.dot(chunk_embeddings, question_embedding)

        ranked_indices = np.argsort(similarities)[::-1]

        top_indices = ranked_indices[:TOP_K]

        answer = normalize(item["answer"])

        relevant_positions = []

        for rank, chunk_index in enumerate(top_indices, start=1):
            chunk_text = normalize(chunks[chunk_index])

            if answer in chunk_text:
                relevant_positions.append(rank)

        # Recall@K / Hit@K
        if relevant_positions:
            recall_hits += 1

        # Precision@K
        relevant_count = len(relevant_positions)
        precision_scores.append(relevant_count / TOP_K)

        # MRR
        if relevant_positions:
            first_rank = relevant_positions[0]
            reciprocal_ranks.append(1 / first_rank)
        else:
            reciprocal_ranks.append(0)

    recall_at_k = recall_hits / len(questions)
    precision_at_k = np.mean(precision_scores)
    mrr = np.mean(reciprocal_ranks)

    return recall_at_k, precision_at_k, mrr


def main():

    with open(QUESTIONS_PATH, "r", encoding="utf-8") as f:
        questions = json.load(f)

    print(f"Loaded {len(questions)} evaluation questions.")

    print("Loading embedding model...")
    model = SentenceTransformer(
        "all-MiniLM-L6-v2",
        device="cpu"
    )

    results = []

    for strategy, file_path in CHUNK_FILES.items():

        print(f"\nEvaluating {strategy}...")

        chunks = load_chunks(file_path)

        print(f"Chunks loaded: {len(chunks)}")

        recall, precision, mrr = calculate_metrics(
            chunks,
            questions,
            model
        )

        results.append({
            "strategy": strategy,
            "chunks": len(chunks),
            "recall": recall,
            "precision": precision,
            "mrr": mrr,
        })

        print(f"Recall@{TOP_K}:    {recall:.4f}")
        print(f"Precision@{TOP_K}: {precision:.4f}")
        print(f"MRR:              {mrr:.4f}")

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:

        f.write("RETRIEVAL EVALUATION\n")
        f.write("=" * 70 + "\n\n")

        f.write(
            f"Evaluation questions: {len(questions)}\n"
            f"Top-K: {TOP_K}\n"
            f"Embedding model: all-MiniLM-L6-v2\n"
            f"Relevance criterion: retrieved chunk contains the ground-truth answer\n\n"
        )

        f.write(
            f"{'Strategy':<15}"
            f"{'Chunks':<10}"
            f"{'Recall@3':<12}"
            f"{'Precision@3':<15}"
            f"{'MRR':<10}\n"
        )

        f.write("-" * 70 + "\n")

        for result in results:
            f.write(
                f"{result['strategy']:<15}"
                f"{result['chunks']:<10}"
                f"{result['recall']:<12.4f}"
                f"{result['precision']:<15.4f}"
                f"{result['mrr']:<10.4f}\n"
            )

    print("\nEvaluation complete.")
    print(f"Results saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
