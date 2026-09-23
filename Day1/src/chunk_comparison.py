import re
import tiktoken
from pathlib import Path


# ============================================================
# 1. LOAD DOCUMENT
# ============================================================

INPUT_FILE = Path("../outputs/extracted_text.txt")

text = INPUT_FILE.read_text(encoding="utf-8")


# ============================================================
# 2. TOKENIZER
# ============================================================

tokenizer = tiktoken.get_encoding("cl100k_base")

tokens = tokenizer.encode(text)


# ============================================================
# 3. CHUNKING SETTINGS
# ============================================================

CHUNK_SIZE = 400
OVERLAP = 60


# ============================================================
# 4. CREATE CHUNKS
# ============================================================

def create_chunks(tokens, chunk_size, overlap):

    chunks = []

    step = chunk_size - overlap

    for start in range(0, len(tokens), step):

        chunk_tokens = tokens[start:start + chunk_size]

        if not chunk_tokens:
            break

        chunk_text = tokenizer.decode(chunk_tokens)

        chunks.append(chunk_text)

        if start + chunk_size >= len(tokens):
            break

    return chunks


# ============================================================
# 5. CREATE BOTH VERSIONS
# ============================================================

chunks_no_overlap = create_chunks(
    tokens,
    CHUNK_SIZE,
    0
)

chunks_overlap = create_chunks(
    tokens,
    CHUNK_SIZE,
    OVERLAP
)


# ============================================================
# 6. SAVE OUTPUTS
# ============================================================

output_dir = Path("../outputs")

no_overlap_file = output_dir / "chunks_400_no_overlap.txt"
overlap_file = output_dir / "chunks_400_overlap.txt"


with no_overlap_file.open("w", encoding="utf-8") as f:

    for i, chunk in enumerate(chunks_no_overlap):

        f.write(f"\n{'=' * 80}\n")
        f.write(f"CHUNK {i}\n")
        f.write(f"{'=' * 80}\n")
        f.write(chunk)
        f.write("\n")


with overlap_file.open("w", encoding="utf-8") as f:

    for i, chunk in enumerate(chunks_overlap):

        f.write(f"\n{'=' * 80}\n")
        f.write(f"CHUNK {i}\n")
        f.write(f"{'=' * 80}\n")
        f.write(chunk)
        f.write("\n")


# ============================================================
# 7. SUMMARY
# ============================================================

print(f"Total document tokens: {len(tokens)}")
print(f"Chunk size: {CHUNK_SIZE}")
print(f"Overlap: {OVERLAP} tokens")

print(
    f"No-overlap chunks: "
    f"{len(chunks_no_overlap)}"
)

print(
    f"Overlap chunks: "
    f"{len(chunks_overlap)}"
)

print(f"\nSaved: {no_overlap_file}")
print(f"Saved: {overlap_file}")
