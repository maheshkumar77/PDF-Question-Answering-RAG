from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import chromadb

# =========================
# READ PDF
# =========================

reader = PdfReader("xxx.pdf")

text = ""

for page in reader.pages:
    page_text = page.extract_text()

    if page_text:
        text += page_text + "\n"

print(f"Total Characters: {len(text)}")

# =========================
# CREATE CHUNKS
# =========================

chunk_size = 1000
overlap = 200

chunks = []

for i in range(0, len(text), chunk_size - overlap):
    chunk = text[i:i + chunk_size]

    if chunk.strip():
        chunks.append(chunk)

print(f"Total Chunks: {len(chunks)}")

# =========================
# LOAD EMBEDDING MODEL
# =========================

print("Loading Model...")

model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

print("Model Loaded")

# =========================
# CREATE EMBEDDINGS
# =========================

embeddings = model.encode(chunks)

print("Embeddings Generated")

# =========================
# CREATE CHROMA DB
# =========================

client = chromadb.PersistentClient(
    path="./chroma_db"
)

# delete old collection if exists
try:
    client.delete_collection("documents")
except:
    pass

collection = client.get_or_create_collection(
    name="documents"
)

# =========================
# STORE DATA
# =========================

for i, chunk in enumerate(chunks):

    collection.add(
        ids=[str(i)],
        documents=[chunk],
        embeddings=[embeddings[i].tolist()],
        metadatas=[{
            "chunk_number": i
        }]
    )

print("================================")
print("PDF Stored Successfully")
print(f"Chunks Stored: {collection.count()}")
print("================================")