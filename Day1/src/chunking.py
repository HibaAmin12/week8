import tiktoken


# ============================================================
# CHUNKING CONFIGURATION
# ============================================================

# Strategy A: 512 tokens with 10% overlap
# CHUNK_SIZE = 512
# OVERLAP = int(CHUNK_SIZE * 0.10)


# Strategy B: 256 tokens with 20% overlap
CHUNK_SIZE = 256
OVERLAP = int(CHUNK_SIZE * 0.20)


# ============================================================
# CHUNKING FUNCTION
# ============================================================

def chunk_text(text, chunk_size, overlap):
    encoding = tiktoken.get_encoding("cl100k_base")

    tokens = encoding.encode(text)

    chunks = []

    step = chunk_size - overlap

    for start in range(0, len(tokens), step):
        end = start + chunk_size

        chunk_tokens = tokens[start:end]

        if not chunk_tokens:
            break

        chunks.append(encoding.decode(chunk_tokens))

        if end >= len(tokens):
            break

    return chunks


# ============================================================
# LOAD CLEANED TEXT
# ============================================================

with open(
    "outputs/extracted_text.txt",
    "r",
    encoding="utf-8"
) as file:
    text = file.read()


# ============================================================
# CREATE CHUNKS
# ============================================================

chunks = chunk_text(
    text,
    CHUNK_SIZE,
    OVERLAP
)


# ============================================================
# DISPLAY RESULTS
# ============================================================

print("=== Active Chunking Strategy ===")
print(f"Chunk size: {CHUNK_SIZE}")
print(f"Overlap: {OVERLAP}")
print(f"Number of chunks: {len(chunks)}")
print("=" * 60)


for i, chunk in enumerate(chunks):
    print(f"Chunk {i}:")
    print(chunk[:300])
    print("-" * 60)