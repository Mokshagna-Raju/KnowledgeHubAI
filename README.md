📚 KnowledgeHub AI

A local AI-powered document assistant for chatting with PDFs,
retrieving relevant information, and comparing multiple documents.

KnowledgeHub AI uses Retrieval-Augmented Generation (RAG), hybrid
retrieval, reranking, and a locally running language model through a
Streamlit interface.

✨ Features

Upload and index multiple PDFs.

Ask questions across all indexed PDFs or selected documents.

Inspect retrieved source passages and document/page information.

Hybrid retrieval using ChromaDB semantic search and BM25 keyword
search.

Reciprocal Rank Fusion (RRF) to combine retrieval rankings.

Cross-encoder reranking to prioritize relevant passages.

Compare two or more PDFs for topics, key information, similarities,
differences, and unique information.

Persistent conversation history using SQLite.

Document management for indexed and unindexed PDFs.

Performance timing details for supported operations.

Local LLM inference with Ollama (qwen2.5:3b by default).

Answers depend on PDF text extraction and the passages retrieved for a
query. Review supporting sources for important decisions.
Scanned/image-only PDFs may require OCR.

🧰 Tech Stack

Component              Technology

UI                     Streamlit
Language               Python
RAG orchestration      LangChain
Local language model   Ollama --- qwen2.5:3b
Vector database        ChromaDB
Embeddings             sentence-transformers/all-MiniLM-L6-v2
Keyword retrieval      BM25 (rank-bm25)
Ranking fusion         Reciprocal Rank Fusion (RRF)
Reranking              cross-encoder/ms-marco-MiniLM-L-6-v2
Conversation storage   SQLite

🏗️ Architecture

PDF Upload
    |
    v
Text Extraction and Chunking
    |
    v
Embedding Model -> ChromaDB
    |
    +--------------------------+
    |                          |
    v                          v
Semantic Search            BM25 Keyword Search
    |                          |
    +------------+-------------+
                 |
                 v
      Reciprocal Rank Fusion
                 |
                 v
       Cross-Encoder Reranking
                 |
                 v
       Context + User Question
                 |
                 v
          Ollama LLM
                 |
                 v
     Answer with Supporting Sources

Conversation history is stored separately in SQLite.

📁 Project Structure

KnowledgeHubAI/
├── app.py              # Streamlit interface, PDF chat, and comparison UI
├── ingest.py           # PDF loading, chunking, indexing, document management
├── rag_pipeline.py     # Retrieval, ranking, question answering, comparison
├── chat_history.py     # SQLite-backed conversation history
├── requirements.txt    # Python dependencies
├── .gitignore          # Excludes local data and environment files
├── data/documents/     # Uploaded PDFs (local; not committed)
├── chroma_db/          # Local ChromaDB data (not committed)
└── chat_history.db     # Local chat history (not committed)

⚙️ Prerequisites

Python

Git

Ollama installed and running locally

Enough memory and disk space for the language model and
embedding/reranking models

The embedding and reranking models may be downloaded the first time they
are initialized.

🚀 Installation

1. Clone the repository

git clone https://github.com/Mokshagna-Raju/KnowledgeHubAI.git
cd KnowledgeHubAI

2. Create and activate a virtual environment

Windows Command Prompt:

python -m venv venv
venv\Scripts\activate

Windows PowerShell:

python -m venv venv
.\venv\Scripts\Activate.ps1

macOS / Linux:

python3 -m venv venv
source venv/bin/activate

3. Install dependencies

python -m pip install --upgrade pip
pip install -r requirements.txt

4. Download the Ollama model

ollama pull qwen2.5:3b
ollama list

Ensure the Ollama service is running before launching the app.

5. Start KnowledgeHub AI

streamlit run app.py

Open the local address printed in your terminal.

📖 How to Use

Start the application.

Upload PDF files in the document management area.

Select Index Documents and wait for indexing to finish.

Open Chat with PDFs to ask questions across all indexed PDFs or
selected PDFs.

Expand the source panel to inspect retrieved passages and page
information.

Open Document Comparison, select at least two indexed PDFs,
enter a comparison question, and click Compare Documents.

View performance timing details when needed.

🔍 Retrieval Pipeline

Semantic search: ChromaDB retrieves passages using embedding
similarity.

Keyword search: BM25 retrieves passages based on relevant terms.

Reciprocal Rank Fusion: combines the rankings from the retrieval
methods.

Cross-encoder reranking: re-scores candidate passages for query
relevance.

Context construction: selected passages are assembled for the
language model.

Generation: Ollama generates an answer or comparison from the
retrieved context.

This combines semantic relevance and keyword matching, but results still
depend on text extraction, chunking, the query, and the retrieved
passages.

🔐 Privacy and Local Data

Uploaded PDFs, ChromaDB files, and SQLite chat history are local
runtime data and should remain excluded from Git.

Ollama runs the configured language model locally by default.

Embedding and reranking models may be downloaded from their model
repositories.

Never commit passwords, API keys, private documents, databases, or
other sensitive information.

Review git status before pushing changes.

🛠️ Troubleshooting

Ollama model is unavailable - Confirm Ollama is installed and
running. - Run ollama list. - If needed, run ollama pull qwen2.5:3b.

A PDF returns no useful text - Confirm the PDF contains selectable
text. - Scanned PDFs may require OCR before indexing.

A PDF is missing from comparison - Confirm it has been uploaded and
indexed. - Select at least two indexed PDFs in the comparison tab.

The first response is slow - The initial run may load the language
model, embedding model, and reranker. Subsequent runs may be faster
while components remain loaded.

🗺️ Potential Future Improvements

OCR support for scanned PDFs.

Automated retrieval-quality evaluation.

Support for additional document formats.

Streaming responses.

More comprehensive automated tests.

Optional deployment configuration.

👨‍💻 Author

Mokshagna Raju Borra

GitHub: Mokshagna-Raju