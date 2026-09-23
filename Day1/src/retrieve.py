import tiktoken
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams


# ============================================================
# CONFIGURATION
# ============================================================

TEXT_PATH = "../outputs/extracted_text.txt"

CHUNK_SIZE = 256
OVERLAP = int(CHUNK_SIZE * 0.20)

MODEL_NAME = "all-MiniLM-L6-v2"

COLLECTION_NAME = "lost_in_the_middle"

TOP_K = 3


# ============================================================
# 1. LOAD CLEANED TEXT
# ============================================================

with open(TEXT_PATH, "r", encoding="utf-8") as file:
    text = file.read()

print("=" * 60)
print("TEXT LOADING")
print("=" * 60)

print(f"Characters: {len(text)}")


# ============================================================
# 2. CHUNK TEXT
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

        chunk = encoding.decode(chunk_tokens)

        chunks.append(chunk)

        if end >= len(tokens):
            break

    return chunks


chunks = chunk_text(
    text,
    CHUNK_SIZE,
    OVERLAP
)

print(f"Chunk size: {CHUNK_SIZE} tokens")
print(f"Overlap: {OVERLAP} tokens")
print(f"Number of chunks: {len(chunks)}")


# ============================================================
# 3. LOAD EMBEDDING MODEL
# ============================================================

print("\n" + "=" * 60)
print("EMBEDDING MODEL")
print("=" * 60)

model = SentenceTransformer(MODEL_NAME)

embeddings = model.encode(chunks)

print(f"Embedding model: {MODEL_NAME}")
print(f"Embedding shape: {embeddings.shape}")


# ============================================================
# 4. CREATE QDRANT IN-MEMORY DATABASE
# ============================================================

print("\n" + "=" * 60)
print("QDRANT SETUP")
print("=" * 60)

client = QdrantClient(":memory:")

client.create_collection(
    collection_name=COLLECTION_NAME,
    vectors_config=VectorParams(
        size=embeddings.shape[1],
        distance=Distance.COSINE,
    ),
)


# ============================================================
# 5. STORE CHUNKS + EMBEDDINGS
# ============================================================

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


# ============================================================
# 6. QUERIES
# ============================================================

QUERIES = [
    {
        "query": (
            "How does the position of relevant information "
            "affect language model performance?"
        ),
    },
    {
        "query": (
            "What tasks are used to evaluate language models "
            "with long contexts?"
        ),
    },
    {
        "query": (
            "What happens when relevant information is placed "
            "in the middle of a long context?"
        ),
    },
]


# ============================================================
# 7. MANUAL GROUND-TRUTH RELEVANT CHUNKS
# ============================================================

# These are manually selected based on whether the chunk
# directly answers the corresponding query.

RELEVANT_CHUNKS = {
    1: [2, 5, 28],
    2: [0, 4, 6],
    3: [0, 2, 5],
}


# ============================================================
# 8. RETRIEVAL FUNCTION
# ============================================================

def retrieve(query, top_k=3):

    query_embedding = model.encode(query)

    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_embedding.tolist(),
        limit=top_k,
        with_payload=True,
    ).points

    return results


# ============================================================
# 9. PRECISION@K
# ============================================================

def precision_at_k(retrieved, relevant, k):

    retrieved = retrieved[:k]

    relevant_retrieved = len(
        set(retrieved) & set(relevant)
    )

    return relevant_retrieved / k


# ============================================================
# 10. RECALL@K
# ============================================================

def recall_at_k(retrieved, relevant, k):

    retrieved = retrieved[:k]

    relevant_retrieved = len(
        set(retrieved) & set(relevant)
    )

    return relevant_retrieved / len(relevant)


# ============================================================
# 11. QUERY 1
# ============================================================

query_1 = QUERIES[0]["query"]

results_1 = retrieve(
    query_1,
    TOP_K
)

print("\n" + "=" * 60)
print("QUERY 1")
print("=" * 60)

print(f"Query: {query_1}")

retrieved_1 = [
    result.payload["chunk_id"]
    for result in results_1
]

print(f"Retrieved chunks: {retrieved_1}")

for rank, result in enumerate(
    results_1,
    start=1
):

    print(
        f"\nRank {rank} | "
        f"Chunk {result.payload['chunk_id']} | "
        f"Similarity Score: {result.score:.4f}"
    )

    print("-" * 60)

    print(result.payload["text"])


# ============================================================
# 12. QUERY 2
# ============================================================

query_2 = QUERIES[1]["query"]

results_2 = retrieve(
    query_2,
    TOP_K
)

print("\n" + "=" * 60)
print("QUERY 2")
print("=" * 60)

print(f"Query: {query_2}")

retrieved_2 = [
    result.payload["chunk_id"]
    for result in results_2
]

print(f"Retrieved chunks: {retrieved_2}")


# ============================================================
# QUERY 2 - MANUAL RELEVANCE INSPECTION
# ============================================================

print("\n" + "=" * 60)
print("QUERY 2 - MANUAL RELEVANCE CHECK")
print("=" * 60)

print(
    "The following chunks are printed so that relevant "
    "evaluation-task chunks can be manually inspected."
)

for chunk_id in range(
    min(12, len(chunks))
):

    print(
        f"\n\nChunk {chunk_id}"
    )

    print("-" * 60)

    print(chunks[chunk_id])


# ============================================================
# 13. QUERY 3
# ============================================================

query_3 = QUERIES[2]["query"]

results_3 = retrieve(
    query_3,
    TOP_K
)

print("\n" + "=" * 60)
print("QUERY 3")
print("=" * 60)

print(f"Query: {query_3}")

retrieved_3 = [
    result.payload["chunk_id"]
    for result in results_3
]

print(f"Retrieved chunks: {retrieved_3}")

for rank, result in enumerate(
    results_3,
    start=1
):

    print(
        f"\nRank {rank} | "
        f"Chunk {result.payload['chunk_id']} | "
        f"Similarity Score: {result.score:.4f}"
    )

    print("-" * 60)

    print(result.payload["text"])


# ============================================================
# 14. EXACT-ID TEST
# ============================================================

exact_id_text = (
    "Ticket REF-4471 was resolved by rotating the API key."
)

exact_id_embedding = model.encode(
    exact_id_text
)

client.upsert(
    collection_name=COLLECTION_NAME,
    points=[
        PointStruct(
            id=1000,
            vector=exact_id_embedding.tolist(),
            payload={
                "chunk_id": 1000,
                "text": exact_id_text,
            },
        )
    ],
)


exact_query = "REF-4471"

exact_query_embedding = model.encode(
    exact_query
)

exact_results = client.query_points(
    collection_name=COLLECTION_NAME,
    query=exact_query_embedding.tolist(),
    limit=TOP_K,
    with_payload=True,
).points


print("\n" + "=" * 60)
print("PART C - EXACT-ID SEMANTIC SEARCH TEST")
print("=" * 60)

print(f"Query: {exact_query}")

exact_retrieved_ids = []

for rank, result in enumerate(
    exact_results,
    start=1
):

    chunk_id = result.payload["chunk_id"]

    exact_retrieved_ids.append(chunk_id)

    print(
        f"\nRank {rank} | "
        f"Chunk {chunk_id} | "
        f"Similarity Score: {result.score:.4f}"
    )

    print(result.payload["text"])


if 1000 in exact_retrieved_ids:

    exact_rank = (
        exact_retrieved_ids.index(1000) + 1
    )

    print(
        f"\nREF-4471 sentence was retrieved "
        f"at rank {exact_rank}."
    )

else:

    print(
        "\nREF-4471 sentence was not retrieved "
        "in the top results."
    )


# ============================================================
# 15. P@3 / R@3 EVALUATION
# ============================================================

print("\n" + "=" * 60)
print("PART C - RETRIEVAL EVALUATION")
print("=" * 60)


all_precisions = []
all_recalls = []


for query_number, item in enumerate(
    QUERIES,
    start=1
):

    query = item["query"]

    results = retrieve(
        query,
        TOP_K
    )

    retrieved_ids = [
        result.payload["chunk_id"]
        for result in results
    ]

    relevant_ids = RELEVANT_CHUNKS[
        query_number
    ]

    precision = precision_at_k(
        retrieved_ids,
        relevant_ids,
        TOP_K
    )

    recall = recall_at_k(
        retrieved_ids,
        relevant_ids,
        TOP_K
    )

    all_precisions.append(
        precision
    )

    all_recalls.append(
        recall
    )

    print(
        f"\nQuery {query_number}: {query}"
    )

    print(
        f"Retrieved: {retrieved_ids}"
    )

    print(
        f"Relevant:  {relevant_ids}"
    )

    print(
        f"Precision@3: {precision:.4f}"
    )

    print(
        f"Recall@3:    {recall:.4f}"
    )


# ============================================================
# 16. AVERAGE PERFORMANCE
# ============================================================

average_precision = (
    sum(all_precisions)
    / len(all_precisions)
)

average_recall = (
    sum(all_recalls)
    / len(all_recalls)
)


print("\n" + "=" * 60)
print("AVERAGE RETRIEVAL PERFORMANCE")
print("=" * 60)

print(
    f"Average Precision@3: "
    f"{average_precision:.4f}"
)

print(
    f"Average Recall@3: "
    f"{average_recall:.4f}"
)


# ============================================================
# 17. HONEST FINDING
# ============================================================

print("\n" + "=" * 60)
print("HONEST FINDING")
print("=" * 60)

print(
    f"Across the three queries, the average "
    f"Precision@3 was {average_precision:.4f} "
    f"and the average Recall@3 was "
    f"{average_recall:.4f}. "
    "Semantic retrieval returned highly relevant "
    "chunks for some queries, while performance "
    "varied across queries. This small experiment "
    "also shows why exact-token matching can benefit "
    "from keyword or hybrid search."
)