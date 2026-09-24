from pathlib import Path
import re

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_PATH = BASE_DIR / "outputs" / "clean_text.txt"
OUTPUT_PATH = BASE_DIR / "outputs" / "semantic_chunks.txt"


def split_sentences(text):
    """
    Split text into sentences.
    """

    sentences = re.split(r"(?<=[.!?])\s+", text.strip())

    return [
        sentence.strip()
        for sentence in sentences
        if sentence.strip()
    ]


def semantic_chunking(
    text,
    model,
    similarity_threshold=0.50
):
    """
    Create chunks based on semantic similarity
    between consecutive sentences.
    """

    sentences = split_sentences(text)

    embeddings = model.encode(
        sentences,
        convert_to_numpy=True
    )

    chunks = []
    current_chunk = [sentences[0]]

    for i in range(1, len(sentences)):

        similarity = cosine_similarity(
            embeddings[i - 1].reshape(1, -1),
            embeddings[i].reshape(1, -1)
        )[0][0]

        if similarity < similarity_threshold:
            chunks.append(" ".join(current_chunk))
            current_chunk = [sentences[i]]
        else:
            current_chunk.append(sentences[i])

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


if __name__ == "__main__":

    text = INPUT_PATH.read_text(encoding="utf-8")

    print("Loading embedding model...")

    model = SentenceTransformer(
        "all-MiniLM-L6-v2"
    )

    chunks = semantic_chunking(
        text,
        model,
        similarity_threshold=0.50
    )

    with OUTPUT_PATH.open("w", encoding="utf-8") as f:

        for i, chunk in enumerate(chunks, start=1):

            f.write(f"\n{'=' * 70}\n")
            f.write(f"CHUNK {i}\n")
            f.write(f"{'=' * 70}\n")
            f.write(chunk)
            f.write("\n")

    sentences = split_sentences(text)

    print(f"Total words: {len(text.split())}")
    print(f"Total sentences: {len(sentences)}")
    print(f"Total semantic chunks: {len(chunks)}")
    print("Similarity threshold: 0.50")
    print(f"Saved to: {OUTPUT_PATH}")