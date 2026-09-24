from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_PATH = BASE_DIR / "outputs" / "clean_text.txt"
OUTPUT_PATH = BASE_DIR / "outputs" / "fixed_chunks.txt"


def fixed_size_chunking(text, chunk_size=500, overlap=50):
    """
    Split text into fixed-size word chunks with overlap.
    """

    words = text.split()

    chunks = []

    start = 0
    step = chunk_size - overlap

    while start < len(words):
        end = start + chunk_size

        chunk = " ".join(words[start:end])

        if chunk:
            chunks.append(chunk)

        start += step

    return chunks


if __name__ == "__main__":
    text = INPUT_PATH.read_text(encoding="utf-8")

    chunks = fixed_size_chunking(
        text,
        chunk_size=500,
        overlap=50
    )

    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        for i, chunk in enumerate(chunks, start=1):
            f.write(f"\n{'=' * 70}\n")
            f.write(f"CHUNK {i}\n")
            f.write(f"{'=' * 70}\n")
            f.write(chunk)
            f.write("\n")

    print(f"Total words: {len(text.split())}")
    print(f"Total chunks: {len(chunks)}")
    print("Chunk size: 500 words")
    print("Overlap: 50 words")
    print(f"Saved to: {OUTPUT_PATH}")
