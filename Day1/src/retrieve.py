import tiktoken
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams


MODEL_NAME = "all-MiniLM-L6-v2"
COLLECTION_NAME = "lost_in_the_middle"

# Strategy A: 512 tokens with 10% overlap
# CHUNK_SIZE = 512
# OVERLAP = int(CHUNK_SIZE * 0.10)

# Strategy B: 256 tokens with 20% overlap
CHUNK_SIZE = 256
OVERLAP = int(CHUNK_SIZE * 0.20)

TOP_K = 3


QUERIES = [
    {
        "query": "How does the position of relevant information affect language model performance?",
        "relevant_chunks": [1, 2, 5, 8, 16, 20, 28],
    },
    {
        "query": "What tasks are used to evaluate language models with long contexts?",
        "relevant_chunks": [0, 2, 3, 6, 7, 11],
    },
    {
        "query": "What happens when relevant information is placed in the middle of a long context?",
        "relevant_chunks": [1, 2, 5, 8, 16, 20, 28, 29, 30],
    },
]


def chunk_text(text, chunk_size, overlap):
    encoding = tiktoken.get_encoding("cl100k_base")

    tokens = encoding.encode(text)

    chunks = []

    step = chunk_size - overlap

    for start in range(0, len(tokens), step):
        end = start + chunk_size

        token_slice = tokens[start:end]

        if not token_slice:
            break

        chunks.append(
            encoding.decode(token_slice)
        )

        if end >= len(tokens):
            break

    return chunks


def precision_at_k(retrieved, relevant, k):
    retrieved = retrieved[:k]

    relevant_retrieved = len(
        set(retrieved) & set(relevant)
    )

    return relevant_retrieved / k


def recall_at_k(retrieved, relevant, k):
    retrieved = retrieved[:k]

    relevant_retrieved = len(
        set(retrieved) & set(relevant)
    )

    return relevant_retrieved / len(relevant)


def main():

    # --------------------------------------------------
    # 1. Load cleaned paper text
    # --------------------------------------------------

    with open(
        "outputs/extracted_text.txt",
        "r",
        encoding="utf-8"
    ) as file:
        text = file.read()


    # --------------------------------------------------
    # 2. Create chunks
    # --------------------------------------------------

    chunks = chunk_text(
        text,
        CHUNK_SIZE,
        OVERLAP
    )

    print("=== Chunking Configuration ===")
    print(f"Chunk size: {CHUNK_SIZE}")
    print(f"Overlap: {OVERLAP}")
    print(f"Number of chunks: {len(chunks)}")
    print("=" * 60)


    # --------------------------------------------------
    # 3. Load embedding model
    # --------------------------------------------------

    model = SentenceTransformer(MODEL_NAME)

    embeddings = model.encode(chunks)

    print(f"Embedding shape: {embeddings.shape}")


    # --------------------------------------------------
    # 4. Create in-memory Qdrant collection
    # --------------------------------------------------

    client = QdrantClient(":memory:")

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(
            size=384,
            distance=Distance.COSINE,
        ),
    )


    # --------------------------------------------------
    # 5. Store chunk embeddings in Qdrant
    # --------------------------------------------------

    points = []

    for chunk_id, (chunk, embedding) in enumerate(
        zip(chunks, embeddings)
    ):

        points.append(
            PointStruct(
                id=chunk_id,
                vector=embedding.tolist(),
                payload={
                    "chunk_id": chunk_id,
                    "text": chunk,
                },
            )
        )


    client.upsert(
        collection_name=COLLECTION_NAME,
        points=points,
    )

    print(f"Stored vectors: {len(points)}")


    # --------------------------------------------------
    # 6. Retrieve and evaluate each query
    # --------------------------------------------------

    for item in QUERIES:

        query = item["query"]
        relevant_chunks = item["relevant_chunks"]


        # Create embedding for query
        query_embedding = model.encode(query)


        # Search Qdrant
        results = client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_embedding.tolist(),
            limit=TOP_K,
            with_payload=True,
        ).points


        # Get retrieved chunk IDs
        retrieved_chunks = [
            result.payload["chunk_id"]
            for result in results
        ]


        # Calculate Precision@K
        precision = precision_at_k(
            retrieved_chunks,
            relevant_chunks,
            TOP_K
        )


        # Calculate Recall@K
        recall = recall_at_k(
            retrieved_chunks,
            relevant_chunks,
            TOP_K
        )


        # --------------------------------------------------
        # 7. Display evaluation results
        # --------------------------------------------------

        print("\n" + "=" * 60)

        print(f"Query: {query}")

        print(
            f"Retrieved chunks: {retrieved_chunks}"
        )

        print(
            f"Relevant chunks: {relevant_chunks}"
        )

        print(
            f"Precision@{TOP_K}: {precision:.4f}"
        )

        print(
            f"Recall@{TOP_K}: {recall:.4f}"
        )


        # --------------------------------------------------
        # 8. Display actual retrieved text
        # --------------------------------------------------

        print("\nRetrieved Chunk Text:")

        for result in results:

            chunk_id = result.payload["chunk_id"]
            retrieved_text = result.payload["text"]
            score = result.score

            print(
                f"\nChunk {chunk_id} | "
                f"Similarity Score: {score:.4f}"
            )

            print("-" * 60)

            print(
                retrieved_text[:500]
            )

            print("-" * 60)


if __name__ == "__main__":
    main()