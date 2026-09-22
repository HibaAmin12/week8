from pathlib import Path

import tiktoken
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams


MODEL_NAME = "all-MiniLM-L6-v2"
COLLECTION_NAME = "lost_in_the_middle"
VECTOR_SIZE = 384

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

        chunk = encoding.decode(chunk_tokens)

        chunks.append(chunk)

        if end >= len(tokens):
            break

    return chunks


def main():

    # Load extracted paper text
    text_path = Path("outputs/extracted_text.txt")

    with open(text_path, "r", encoding="utf-8") as file:
        text = file.read()

    # Create chunks
    chunks = chunk_text(
        text,
        CHUNK_SIZE,
        OVERLAP
    )

    print(f"Number of chunks: {len(chunks)}")

    # Load embedding model
    model = SentenceTransformer(MODEL_NAME)

    # Generate embeddings
    embeddings = model.encode(chunks)

    print(f"Embedding shape: {embeddings.shape}")

    # Create in-memory Qdrant
    client = QdrantClient(":memory:")

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(
            size=VECTOR_SIZE,
            distance=Distance.COSINE,
        ),
    )

    # Create Qdrant points
    points = []

    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):

        point = PointStruct(
            id=i,
            vector=embedding.tolist(),
            payload={
                "text": chunk,
                "chunk_id": i,
            },
        )

        points.append(point)

    # Insert vectors
    client.upsert(
        collection_name=COLLECTION_NAME,
        points=points,
    )

    print(f"Inserted points: {len(points)}")

    # Verify collection
    collection_info = client.get_collection(COLLECTION_NAME)

    print(
        f"Stored vectors: "
        f"{collection_info.points_count}"
    )


if __name__ == "__main__":
    main()