from pathlib import Path
import re


BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_PATH = BASE_DIR / "outputs" / "clean_text.txt"
OUTPUT_PATH = BASE_DIR / "outputs" / "recursive_chunks.txt"


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


def recursive_split(text, chunk_size=500, overlap=50):
    """
    Recursively split text using paragraph, sentence,
    and word boundaries.
    """

    text = text.strip()

    if not text:
        return []

    words = text.split()

    # If text already fits within the target size,
    # keep it as one chunk.
    if len(words) <= chunk_size:
        return [text]

    # First try paragraph boundaries.
    paragraphs = [
        paragraph.strip()
        for paragraph in text.split("\n\n")
        if paragraph.strip()
    ]

    if len(paragraphs) > 1:

        chunks = []

        current = []

        for paragraph in paragraphs:

            paragraph_words = len(paragraph.split())
            current_words = len(" ".join(current).split())

            if current and current_words + paragraph_words > chunk_size:

                chunks.extend(
                    recursive_split(
                        "\n\n".join(current),
                        chunk_size,
                        overlap
                    )
                )

                current = []

            current.append(paragraph)

        if current:
            chunks.extend(
                recursive_split(
                    "\n\n".join(current),
                    chunk_size,
                    overlap
                )
            )

        return chunks

    # If there is only one paragraph and it is too large,
    # try sentence boundaries.
    sentences = split_sentences(text)

    if len(sentences) > 1:

        chunks = []
        current = []

        for sentence in sentences:

            current_words = len(" ".join(current).split())
            sentence_words = len(sentence.split())

            if current and current_words + sentence_words > chunk_size:

                chunks.append(" ".join(current))

                # Keep overlap from previous sentences.
                overlap_words = []
                count = 0

                for previous_sentence in reversed(current):

                    previous_words = previous_sentence.split()

                    if count + len(previous_words) > overlap:
                        break

                    overlap_words.insert(0, previous_sentence)
                    count += len(previous_words)

                current = overlap_words

            current.append(sentence)

        if current:
            chunks.append(" ".join(current))

        return chunks

    # Final fallback: word-level splitting.
    chunks = []

    step = chunk_size - overlap

    for start in range(0, len(words), step):

        chunk = " ".join(
            words[start:start + chunk_size]
        )

        if chunk:
            chunks.append(chunk)

    return chunks


if __name__ == "__main__":

    text = INPUT_PATH.read_text(encoding="utf-8")

    chunks = recursive_split(
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
    print(f"Total recursive chunks: {len(chunks)}")
    print("Chunk size: 500 words")
    print("Overlap: 50 words")
    print(f"Saved to: {OUTPUT_PATH}")
