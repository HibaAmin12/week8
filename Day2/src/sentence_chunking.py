from pathlib import Path
import re


BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_PATH = BASE_DIR / "outputs" / "clean_text.txt"
OUTPUT_PATH = BASE_DIR / "outputs" / "sentence_chunks.txt"


def split_sentences(text):
    """
    Split text into sentences using sentence-ending punctuation.
    """

    sentences = re.split(r"(?<=[.!?])\s+", text.strip())

    return [
        sentence.strip()
        for sentence in sentences
        if sentence.strip()
    ]


def sentence_chunking(text, sentences_per_chunk=5):
    """
    Group a fixed number of sentences into each chunk.
    """

    sentences = split_sentences(text)

    chunks = []

    for i in range(0, len(sentences), sentences_per_chunk):
        chunk = " ".join(
            sentences[i:i + sentences_per_chunk]
        )

        chunks.append(chunk)

    return chunks


if __name__ == "__main__":
    text = INPUT_PATH.read_text(encoding="utf-8")

    sentences = split_sentences(text)

    chunks = sentence_chunking(
        text,
        sentences_per_chunk=5
    )

    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        for i, chunk in enumerate(chunks, start=1):
            f.write(f"\n{'=' * 70}\n")
            f.write(f"CHUNK {i}\n")
            f.write(f"{'=' * 70}\n")
            f.write(chunk)
            f.write("\n")

    print(f"Total words: {len(text.split())}")
    print(f"Total sentences: {len(sentences)}")
    print(f"Total chunks: {len(chunks)}")
    print("Sentences per chunk: 5")
    print(f"Saved to: {OUTPUT_PATH}")
