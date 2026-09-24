from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_PATH = BASE_DIR / "outputs" / "clean_text.txt"
OUTPUT_PATH = BASE_DIR / "outputs" / "paragraph_chunks.txt"


def paragraph_chunking(text):
    """
    Split text into chunks based on paragraph boundaries.
    """

    paragraphs = [
        paragraph.strip()
        for paragraph in text.split("\n\n")
        if paragraph.strip()
    ]

    return paragraphs


if __name__ == "__main__":
    text = INPUT_PATH.read_text(encoding="utf-8")

    chunks = paragraph_chunking(text)

    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        for i, chunk in enumerate(chunks, start=1):
            f.write(f"\n{'=' * 70}\n")
            f.write(f"CHUNK {i}\n")
            f.write(f"{'=' * 70}\n")
            f.write(chunk)
            f.write("\n")

    print(f"Total words: {len(text.split())}")
    print(f"Total paragraphs: {len(chunks)}")
    print(f"Saved to: {OUTPUT_PATH}")
