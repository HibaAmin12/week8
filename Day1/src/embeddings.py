from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


MODEL_NAME = "all-MiniLM-L6-v2"

model = SentenceTransformer(MODEL_NAME)

texts = [
    "Language models use context to answer questions.",
    "Retrieval helps language models find relevant information.",
    "The weather is sunny today."
]

embeddings = model.encode(texts)

similarity_matrix = cosine_similarity(embeddings)

print(f"Number of texts: {len(texts)}")
print(f"Embedding shape: {embeddings.shape}")
print(f"First embedding dimensions: {len(embeddings[0])}")

print("\nCosine Similarity Matrix:")
print(similarity_matrix)