import numpy as np
from sentence_transformers import SentenceTransformer


# ============================================================
# 1. LOAD MODEL
# ============================================================

MODEL_NAME = "all-MiniLM-L6-v2"

model = SentenceTransformer(MODEL_NAME)


# ============================================================
# 2. THREE SENTENCES
# ============================================================

texts = [
    "Language models use context to answer questions.",
    "Retrieval helps language models find relevant information.",
    "The weather is sunny today.",
]


# ============================================================
# 3. CREATE EMBEDDINGS
# ============================================================

embeddings = model.encode(texts)

print("=" * 60)
print("EMBEDDINGS")
print("=" * 60)

print(f"Number of texts: {len(texts)}")
print(f"Embedding shape: {embeddings.shape}")
print(f"Embedding dimensions: {len(embeddings[0])}")


# ============================================================
# 4. MANUAL COSINE SIMILARITY
# ============================================================

def cosine_similarity_manual(vector_a, vector_b):

    dot_product = np.dot(
        vector_a,
        vector_b
    )

    norm_a = np.linalg.norm(vector_a)

    norm_b = np.linalg.norm(vector_b)

    similarity = (
        dot_product
        / (norm_a * norm_b)
    )

    return similarity


# ============================================================
# 5. CALCULATE SIMILARITY
# ============================================================

similarity_1_2 = cosine_similarity_manual(
    embeddings[0],
    embeddings[1]
)

similarity_1_3 = cosine_similarity_manual(
    embeddings[0],
    embeddings[2]
)

similarity_2_3 = cosine_similarity_manual(
    embeddings[1],
    embeddings[2]
)


# ============================================================
# 6. DISPLAY RESULTS
# ============================================================

print("\n" + "=" * 60)
print("MANUAL COSINE SIMILARITY")
print("=" * 60)

print(
    f"Sentence 1 vs Sentence 2: "
    f"{similarity_1_2:.4f}"
)

print(
    f"Sentence 1 vs Sentence 3: "
    f"{similarity_1_3:.4f}"
)

print(
    f"Sentence 2 vs Sentence 3: "
    f"{similarity_2_3:.4f}"
)


# ============================================================
# 7. INTERPRETATION
# ============================================================

print("\n" + "=" * 60)
print("INTERPRETATION")
print("=" * 60)

print(
    f"Sentence 1 and Sentence 2 have a cosine similarity "
    f"of {similarity_1_2:.4f}, indicating that they are "
    "semantically related."
)

print(
    f"Sentence 1 and Sentence 3 have a cosine similarity "
    f"of {similarity_1_3:.4f}, indicating very low "
    "semantic similarity."
)

print(
    f"Sentence 2 and Sentence 3 have a cosine similarity "
    f"of {similarity_2_3:.4f}, also indicating very low "
    "semantic similarity."
)