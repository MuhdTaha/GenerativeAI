# RAG PDF CHATBOT

A simple Retrieval-Augmented Generation (RAG) system that allows you to ask questions about any PDF using local embeddings, Qdrant vector database, and a local LLM via LM Studio.

## 🚀 Features

- Load any PDF and chunk its content
- Generate embeddings using `BAAI/bge-large-en-v1.5`
- Store and retrieve chunks from Qdrant vector store
- Rephrase context-dependent queries into standalone ones
- Stream intelligent answers from a local LLM (e.g., DeepSeek, Gemma, etc.)

## 🧱 Tech Stack

- [LangChain](https://github.com/langchain-ai/langchain)
- [Qdrant](https://qdrant.tech/)
- [LM Studio](https://lmstudio.ai/)
- [PyTorch](https://pytorch.org/)
- HuggingFace Embeddings

## 📦 Setup

### 1. Clone the Repo

```bash
git clone https://github.com/yourusername/rag-pdf-chatbot.git
cd rag-pdf-chatbot
```

### 2. Set Up Python Environment

```bash
python -m venv env
source env/bin/activate  # or .\env\Scripts\activate on Windows
pip install -r requirements.txt
```

### 3. Create `.env` File

```bash
cp .env.example .env
# Fill in the required environment variables
```

### 4. Run Qdrant (if not already running)

```bash
docker pull qdrant/qdrant
docker run -p 6333:6333 -v "$(pwd)/qdrant_storage:/qdrant/storage:z" qdrant/qdrant
```

### 5. Run the Script

```bash
python rag_pdf_chatbot.py
```

You’ll be prompted to enter a PDF file and start asking questions about its content.

## 📚 Example

```
Enter the path to your PDF: docs/my_report.pdf
Ask a question from the PDF: What is a knowledge graph?
💬 Answer:
A knowledge graph is a structured representation of information...
```

## 🤖 LLM Configuration

This project is designed to work with a **local LLM** served through [LM Studio](https://lmstudio.ai/), e.g., `deepseek-r1-distill-qwen-7b`, `gemma-1.1-2b-it`, etc.

Make sure LM Studio is running at `http://127.0.0.1:1234/v1`.

## ✅ TODO

- [ ] Add web-based interface (Gradio / Streamlit)
- [ ] Add multi-file/document support
- [ ] Add citation highlighting from retrieved chunks

## 🛡️ License

MIT License.

---

## 👤 Author

Made by [Muhammad Taha](https://github.com/MuhdTaha)
