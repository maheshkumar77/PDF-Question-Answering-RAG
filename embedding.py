import chromadb
from sentence_transformers import SentenceTransformer

# =========================
# LOAD MODEL
# =========================

model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

# =========================
# CONNECT TO CHROMA
# =========================

client = chromadb.PersistentClient(
    path="./chroma_db"
)

collection = client.get_collection(
    name="documents"
)

# =========================
# ASK QUESTION
# =========================

question = input("\nAsk Question: ")

# =========================
# CREATE QUESTION EMBEDDING
# =========================

query_embedding = model.encode(
    question
)

# =========================
# SEARCH
# =========================

results = collection.query(
    query_embeddings=[
        query_embedding.tolist()
    ],
    n_results=3
)

# =========================
# SHOW RESULTS
# =========================

print("\nTop Matching Chunks\n")

for i, doc in enumerate(results["documents"][0]):

    print(f"\nResult {i+1}")
    print("-" * 50)

    print(doc)

    print("\nDistance:")
    print(results["distances"][0][i])

    print("-" * 50)