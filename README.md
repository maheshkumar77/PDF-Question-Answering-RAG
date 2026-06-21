# AI-PDF-RAG-Chatbot

An AI-powered Retrieval-Augmented Generation (RAG) system that allows users to upload PDF documents, generate embeddings, store them in ChromaDB, and ask questions based on the document content.

The project uses semantic search to retrieve relevant information from PDFs and can be integrated with LLMs such as Gemini, OpenAI GPT, or Llama for intelligent question answering.

---

## Features

- PDF Text Extraction
- Intelligent Text Chunking
- Sentence Transformer Embeddings
- ChromaDB Vector Database
- Semantic Search
- Question Answering over Documents
- Local Vector Storage
- Easy Integration with LLMs

---

## Project Workflow

```text
PDF
 ↓
Text Extraction
 ↓
Chunking
 ↓
Embeddings
 ↓
ChromaDB
 ↓
Semantic Retrieval
 ↓
LLM (Optional)
 ↓
Answer Generation
```

---

## Tech Stack

- Python
- PyMuPDF
- Sentence Transformers
- ChromaDB
- Transformers
- NLP
- Vector Database
- RAG (Retrieval-Augmented Generation)

---

## Installation

### Clone Repository

```bash
git clone https://github.com/yourusername/AI-PDF-RAG-Chatbot.git

cd AI-PDF-RAG-Chatbot
```

### Create Virtual Environment

```bash
python -m venv venv
```

### Activate Virtual Environment

Windows:

```bash
venv\Scripts\activate
```

Linux/Mac:

```bash
source venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Required Packages

```bash
pip install pymupdf
pip install sentence-transformers
pip install chromadb
```

---

## Project Structure

```text
AI-PDF-RAG-Chatbot/
│
├── resume.pdf
├── store_vectors.py
├── query.py
├── chroma_db/
├── requirements.txt
├── README.md
```

---

## Store PDF Embeddings

Run:

```bash
python store_vectors.py
```

This will:

1. Read the PDF
2. Extract text
3. Create chunks
4. Generate embeddings
5. Store vectors in ChromaDB

Example Output:

```text
Total Chunks: 12
Embeddings Generated
PDF Stored Successfully
Chunks Stored: 12
```

---

## Query the Document

Run:

```bash
python query.py
```

Example:

```text
Ask Question:
What skills does the candidate have?
```

Output:

```text
Top Matching Chunks:

Python
Machine Learning
Deep Learning
React
Node.js
```

---

## How It Works

### Step 1: PDF Extraction

The system extracts text from PDF documents using PyMuPDF.

### Step 2: Chunking

Large documents are split into smaller chunks with overlap to preserve context.

### Step 3: Embeddings

Each chunk is converted into a dense vector using the Sentence Transformer model:

```text
all-MiniLM-L6-v2
```

### Step 4: Vector Storage

Embeddings are stored inside ChromaDB for efficient retrieval.

### Step 5: Semantic Search

User questions are converted into embeddings and matched against stored document embeddings.

### Step 6: Response Generation

Retrieved chunks can be passed to an LLM such as Gemini, GPT, or Llama to generate final answers.

---

## Future Improvements

- Multi-PDF Support
- FastAPI Backend
- React Frontend
- Gemini Integration
- OpenAI Integration
- Chat History
- Source Citations
- Hybrid Search
- Docker Deployment
- Authentication System

---

## Sample Use Cases

- Resume Question Answering
- Research Paper Assistant
- Legal Document Search
- Educational PDF Chatbot
- Knowledge Base Assistant
- Company Policy Search

---

## Author

Mahesh Kumar Sahu

AI/ML Developer

GitHub: https://github.com/yourusername

LinkedIn: https://linkedin.com/in/yourprofile
