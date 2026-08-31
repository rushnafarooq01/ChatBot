# NotesGenie - AI-Powered Document Q&A Chatbot

NotesGenie is a RAG (Retrieval-Augmented Generation) chatbot that lets you upload any PDF - lecture notes, resumes, documentation - and ask natural language questions about it. It retrieves the most relevant sections from your document and generates accurate, source-backed answers using an LLM.

Live Demo: https://chatbot-krp7ycfedjxejevh9utch3.streamlit.app/

---

## Features

- Upload any PDF and start asking questions instantly
- Semantic search - finds the most relevant parts of your document, not just keyword matches
- AI-generated answers grounded strictly in your document's content
- Source citations - see exactly which section of the PDF an answer came from
- Chat-style interface with conversation history
- Fast inference powered by Groq's LLM API

---

## Tech Stack

| Layer | Technology |
|---|---|
| Language | Python |
| LLM | Groq API (openai/gpt-oss-20b) |
| Orchestration | LangChain |
| Embeddings | HuggingFace Sentence Transformers |
| Vector Database | ChromaDB |
| PDF Parsing | PyPDF |
| Frontend / UI | Streamlit |
| Deployment | Streamlit Community Cloud |

---

## How It Works

1. User uploads a PDF through the sidebar
2. The document is split into overlapping text chunks
3. Each chunk is converted into a vector embedding
4. Embeddings are stored in a Chroma vector database
5. When a question is asked, the most relevant chunks are retrieved via similarity search
6. The retrieved context and question are passed to the LLM, which generates an answer grounded in the document

```
PDF Upload -> Text Chunking -> Embeddings -> Vector DB (Chroma)

User Question -> Similarity Search -> Relevant Chunks -> LLM -> Answer
```

---

## Running Locally

**1. Clone the repository**
```bash
git clone https://github.com/rushnafarooq01/ChatBot.git
cd ChatBot
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Add your API key**

Create a `.env` file in the project root:
```
GROQ_API_KEY=your_groq_api_key_here
```

Get a free key at console.groq.com

**4. Run the app**
```bash
streamlit run app.py
```

---

## Project Structure

```
ChatBot/
|-- app.py              # Main Streamlit app (UI + RAG pipeline)
|-- ingest.py           # Standalone script for offline PDF ingestion
|-- query.py            # Standalone CLI version for testing Q&A
|-- requirements.txt    # Python dependencies
|-- logo.png            # App logo
|-- .gitignore
```

---

## Author

**Rushna Farooq**

Built as a hands-on project to learn Retrieval-Augmented Generation (RAG), vector databases, and LLM-powered application development.
