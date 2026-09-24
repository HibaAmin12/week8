from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = BASE_DIR / "outputs"


FILES = {
    "Fixed-size": "fixed_chunks.txt",
    "Sentence": "sentence_chunks.txt",
    "Paragraph": "paragraph_chunks.txt",
    "Semantic": "semantic_chunks.txt",
    "Recursive": "recursive_chunks.txt",
}


def load_chunks(file_path):
    """Read chunks from an output file."""

    text = file_path.read_text(encoding="utf-8")

    # Split whenever a new CHUNK starts
    sections = text.split("CHUNK ")

    chunks = []

    for section in sections[1:]:

        # Remove the chunk number and separator
        lines = section.splitlines()

        if len(lines) < 3:
            continue

        # Actual text starts after the second separator line
        chunk_text = " ".join(lines[2:]).strip()

        if chunk_text:
            chunks.append(chunk_text)

    return chunks


def get_statistics(chunks):
    """Calculate chunk-size statistics."""

    word_counts = [
        len(chunk.split())
        for chunk in chunks
    ]

    if not word_counts:
        return {
            "chunks": 0,
            "average": 0,
            "minimum": 0,
            "maximum": 0,
        }

    return {
        "chunks": len(word_counts),
        "average": sum(word_counts) / len(word_counts),
        "minimum": min(word_counts),
        "maximum": max(word_counts),
    }


if __name__ == "__main__":

    print("\nChunking Strategy Comparison")
    print("=" * 80)

    for strategy, filename in FILES.items():

        file_path = OUTPUTS_DIR / filename

        chunks = load_chunks(file_path)
        stats = get_statistics(chunks)

        print(f"\n{strategy}")
        print("-" * 40)
        print(f"Total chunks  : {stats['chunks']}")
        print(f"Average words : {stats['average']:.2f}")
        print(f"Minimum words : {stats['minimum']}")
        print(f"Maximum words : {stats['maximum']}")