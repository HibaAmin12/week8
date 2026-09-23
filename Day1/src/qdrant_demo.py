from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct


# ============================================================
# 1. LOAD EMBEDDING MODEL
# ============================================================

MODEL_NAME = "all-MiniLM-L6-v2"

model = SentenceTransformer(MODEL_NAME)


# ============================================================
# 2. SMALL REAL CORPUS
# ============================================================

documents = [
    "Language models perform better when relevant information appears near the beginning or end of the context.",
    "The Lost in the Middle study investigates how language models use information placed at different positions in long contexts.",
    "Multi-document question answering requires a model to find relevant information among several documents.",
    "Key-value retrieval tests whether a language model can retrieve a specific value associated with a key.",
    "Increasing the length of the input context does not always improve language model performance.",
    "Relevant information placed in the middle of a long context can lead to substantial performance degradation.",
    "The study evaluates language models using controlled changes to context length and information position.",
    "Longer contexts can increase the amount of content that a language model must reason over.",
    "The researchers observe a distinctive U-shaped performance curve across different information positions.",
    "Open-domain question answering is used to study the trade-off between additional context and model accuracy.",
]


# ============================================================
# 3. CREATE EMBEDDINGS
# ============================================================

embeddings = model.encode(documents)

print("=" * 70)
print("DOCUMENT EMBEDDINGS")
print("=" * 70)

print(f"Number of documents: {len(documents)}")
print(f"Embedding shape: {embeddings.shape}")


# ============================================================
# 4. CREATE QDRANT IN-MEMORY DATABASE
# ============================================================

client = QdrantClient(":memory:")

COLLECTION_NAME = "rag_demo"


client.create_collection(
    collection_name=COLLECTION_NAME,
    vectors_config=VectorParams(
        size=embeddings.shape[1],
        distance=Distance.COSINE,
    ),
)


# ============================================================
# 5. STORE VECTORS + PAYLOAD
# ============================================================

points = []

for i, (text, vector) in enumerate(
    zip(documents, embeddings)
):

    points.append(
        PointStruct(
            id=i,
            vector=vector.tolist(),
            payload={
                "text": text,
                "document_id": i,
            },
        )
    )


client.upsert(
    collection_name=COLLECTION_NAME,
    points=points,
)


print(f"Stored vectors: {len(points)}")


# ============================================================
# 6. QUERY
# ============================================================

query = (
    "How does the position of relevant information "
    "affect language model performance?"
)

query_embedding = model.encode(query)


# ============================================================
# 7. VECTOR SEARCH
# ============================================================

results = client.query_points(
    collection_name=COLLECTION_NAME,
    query=query_embedding.tolist(),
    limit=3,
).points


# ============================================================
# 8. DISPLAY RESULTS
# ============================================================

print("\n" + "=" * 70)
print("QUERY")
print("=" * 70)

print(query)


print("\n" + "=" * 70)
print("TOP 3 RESULTS")
print("=" * 70)


for rank, result in enumerate(results, start=1):

    print(f"\nRank {rank}")
    print(f"Document ID: {result.payload['document_id']}")
    print(f"Similarity Score: {result.score:.4f}")
    print(f"Text: {result.payload['text']}")


# ============================================================
# 9. TOP RESULT CHECK
# ============================================================

top_result = results[0]

print("\n" + "=" * 70)
print("TOP RESULT CHECK")
print("=" * 70)

print(
    f"Top document ID: "
    f"{top_result.payload['document_id']}"
)

print(
    f"Top similarity score: "
    f"{top_result.score:.4f}"
)

print(
    "The top result is about the position of relevant "
    "information and its effect on language model performance."
)
