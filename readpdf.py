# from pypdf import PdfReader
# from sentence_transformers import SentenceTransformer
# import chromadb

# # =========================
# # READ PDF
# # =========================

# reader = PdfReader("xxx.pdf")

# text = ""

# for page in reader.pages:
#     page_text = page.extract_text()

#     if page_text:
#         text += page_text + "\n"

# print(f"Total Characters: {len(text)}")

# # =========================
# # CREATE CHUNKS
# # =========================

# chunk_size = 1000
# overlap = 200

# chunks = []

# for i in range(0, len(text), chunk_size - overlap):
#     chunk = text[i:i + chunk_size]

#     if chunk.strip():
#         chunks.append(chunk)

# print(f"Total Chunks: {len(chunks)}")

# # =========================
# # LOAD EMBEDDING MODEL
# # =========================

# print("Loading Model...")

# model = SentenceTransformer(
#     "all-MiniLM-L6-v2"
# )

# print("Model Loaded")

# # =========================
# # CREATE EMBEDDINGS
# # =========================

# embeddings = model.encode(chunks)

# print("Embeddings Generated")

# # =========================
# # CREATE CHROMA DB
# # =========================

# client = chromadb.PersistentClient(
#     path="./chroma_db"
# )

# # delete old collection if exists
# try:
#     client.delete_collection("documents")
# except:
#     pass

# collection = client.get_or_create_collection(
#     name="documents"
# )

# # =========================
# # STORE DATA
# # =========================

# for i, chunk in enumerate(chunks):

#     collection.add(
#         ids=[str(i)],
#         documents=[chunk],
#         embeddings=[embeddings[i].tolist()],
#         metadatas=[{
#             "chunk_number": i
#         }]
#     )

# print("================================")
# print("PDF Stored Successfully")
# print(f"Chunks Stored: {collection.count()}")
# print("================================")



import streamlit as st
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import chromadb
import tempfile
import os

st.set_page_config(page_title="PDF Embedding App")

st.title("📄 PDF Embedding Generator")

uploaded_file = st.file_uploader(
    "Upload PDF",
    type=["pdf"]
)

if uploaded_file:

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".pdf"
    ) as tmp_file:
        tmp_file.write(uploaded_file.read())
        pdf_path = tmp_file.name

    st.info("Reading PDF...")

    reader = PdfReader(pdf_path)

    text = ""

    for page in reader.pages:
        page_text = page.extract_text()

        if page_text:
            text += page_text + "\n"

    st.success(f"PDF Loaded ({len(text)} characters)")

    chunk_size = 1000
    overlap = 200

    chunks = []

    for i in range(
        0,
        len(text),
        chunk_size - overlap
    ):
        chunk = text[i:i + chunk_size]

        if chunk.strip():
            chunks.append(chunk)

    st.write(f"Total Chunks: {len(chunks)}")

    if st.button("Generate Embeddings"):

        with st.spinner("Loading model..."):

            model = SentenceTransformer(
                "all-MiniLM-L6-v2"
            )

            embeddings = model.encode(chunks)

        st.success("Embeddings Generated")

        client = chromadb.PersistentClient(
            path="./chroma_db"
        )

        try:
            client.delete_collection(
                "documents"
            )
        except:
            pass

        collection = client.get_or_create_collection(
            name="documents"
        )

        for i, chunk in enumerate(chunks):

            collection.add(
                ids=[str(i)],
                documents=[chunk],
                embeddings=[
                    embeddings[i].tolist()
                ],
                metadatas=[
                    {
                        "chunk_number": i
                    }
                ]
            )

        st.success(
            f"Stored {collection.count()} chunks in ChromaDB"
        )

        os.remove(pdf_path)