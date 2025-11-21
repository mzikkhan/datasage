# DataSage
DataSage is a fully local, extensible Retrieval-Augmented Generation (RAG) engine designed for flexible data ingestion, semantic search, and context-aware question answering. It supports pdf, txt, csv, and xslx; and uses modern open-source LLM tooling such as Ollama - with zero cloud dependencies.

🌟 Features

Document Ingestion: Support for multiple file formats (CSV, PDF, TXT)
Intelligent Chunking: Configurable text splitting with overlap for context preservation
Vector Storage: ChromaDB-backed vector database for efficient similarity search
Semantic Search: HuggingFace embeddings for accurate document retrieval
LLM Integration: Local LLM support via Ollama for answer generation
Modular Architecture: Easy to extend and customize components

🏗️ Architecture
DataSage
├── Ingestion Layer     → Load and chunk documents
├── Indexing Layer      → Embed and store in vector database
├── Query Layer         → Retrieve relevant context and generate answers
└── RAG Pipeline        → End-to-end question answering system
📋 Prerequisites

Python 3.8+
Ollama (for local LLM inference)

🚀 Installation
1. Clone the repository
bashgit clone https://github.com/yourusername/datasage.git
cd datasage
2. Install dependencies
bashpip install -r requirements.txt
Required packages:
txtlangchain
langchain-huggingface
langchain-chroma
langchain-ollama
langchain-core
chromadb
sentence-transformers
pypdf
3. Install Ollama
Download and install Ollama from ollama.com
Pull a model:
bashollama pull llama3.1
Verify installation:
bashollama run llama3.1

📁 Project Structure
datasage/
├── indexing/
│   ├── embedder.py          # Text embedding using HuggingFace
│   ├── vector_store.py      # ChromaDB vector storage
│   └── index_engine.py      # High-level indexing pipeline
├── ingestion/
│   ├── loaders.py           # Document loaders (PDF, CSV, TXT)
│   └── chunker.py           # Text chunking utilities
├── query/
│   ├── retriever.py         # Semantic search retriever
│   └── generator.py         # LLM answer generation
└── __init__.py
Supported File Formats

CSV: Loaded with metadata for each row
PDF: Extracted page by page
TXT: Loaded as single document

🎯 Use Cases

Document Q&A: Query large documents using natural language
Knowledge Base Search: Build searchable knowledge bases
Customer Support: Answer questions from documentation
Research Assistant: Extract information from academic papers
Code Documentation: Query codebases and technical docs

📝 License
This project is licensed under the MIT License - see the LICENSE file for details.
🙏 Acknowledgments

Built with LangChain
Embeddings powered by HuggingFace
Vector storage by ChromaDB
Local LLM inference via Ollama

📧 Contact
For questions or support, please open an issue on GitHub.

Made with ❤️ by the DataSage Team
