from pathlib import Path

import tiktoken


CHUNK_SIZE = 512
OVERLAP = int(CHUNK_SIZE * 0.10)


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


with open(
    "outputs/extracted_text.txt",
    "r",
    encoding="utf-8"
) as file:
    text = file.read()


chunks = chunk_text(
    text,
    CHUNK_SIZE,
    OVERLAP
)


for i, chunk in enumerate(chunks):
    print(f"Chunk {i}:")
    print(chunk[:300])
    print("-" * 60)