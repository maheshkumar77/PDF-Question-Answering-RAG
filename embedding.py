import chromadb
from sentence_transformers import SentenceTransformer
from google import genai
import streamlit as st

# =========================
# GEMINI SETUP
# =========================
# Put your Gemini API key here
GEMINI_API_KEY = "Put your Gemini API key herePut your Gemini API key here"

st.set_page_config(page_title="PDF Question Answering", page_icon=":books:")
st.title("PDF Question Answering with Gemini and ChromaDB")


client_ai = genai.Client(api_key=GEMINI_API_KEY)

# =========================
# LOAD EMBEDDING MODEL
# =========================
model = SentenceTransformer("all-MiniLM-L6-v2")

# =========================
# CONNECT TO CHROMA
# =========================
client = chromadb.PersistentClient(path="./chroma_db")

collection = client.get_collection(name="documents")

# =========================
# ASK QUESTION
# =========================
question = st.text_input("\nAsk Question: ")

# =========================
# CREATE QUESTION EMBEDDING
# =========================
query_embedding = model.encode(question)

# =========================
# SEARCH TOP MATCHING CHUNKS
# =========================
results = collection.query(
    query_embeddings=[query_embedding.tolist()],
    n_results=3
)

# =========================
# EXTRACT RETRIEVED DOCUMENTS
# =========================
retrieved_docs = results["documents"][0]

# print("\nTop Matching Chunks\n")
# for i, doc in enumerate(retrieved_docs):
#     print(f"\nResult {i+1}")
#     print("-" * 50)
#     print(doc)
#     print("\nDistance:")
#     print(results["distances"][0][i])
#     print("-" * 50)
st.header("Top Matching Results")
for i, result in enumerate(retrieved_docs, start=1):
    st.subheader(f"Result {i}")
    st.write(result)
    st.divider()

# =========================
# COMBINE CHUNKS INTO CONTEXT
# =========================
context = "\n\n".join(retrieved_docs)

# =========================
# BUILD PROMPT FOR AI
# =========================
prompt = f"""
You are a helpful AI assistant.

Answer the user's question using ONLY the information provided in the context below.
If the answer is not present in the context, say:
"I could not find the exact answer in the provided documents."

Give the answer in a clear, natural, human-like way.

Context:
{context}

User Question:
{question}
"""

# =========================
# GENERATE FINAL ANSWER USING GEMINI
# =========================
response = client_ai.models.generate_content(
    model="gemini-2.5-flash",
    contents=prompt
)

# =========================
# SHOW FINAL AI RESPONSE
# =========================
st.header("AI Answer")
st.write(response.text)
    
    
