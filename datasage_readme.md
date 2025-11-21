# DataSage 🧙‍♂️

A lightweight, modular Python package for building Retrieval-Augmented Generation (RAG) systems. DataSage enables you to query your documents using natural language by combining semantic search with large language models.

## 🌟 Features

- **Document Ingestion**: Support for multiple file formats (CSV, PDF, TXT)
- **Intelligent Chunking**: Configurable text splitting with overlap for context preservation
- **Vector Storage**: ChromaDB-backed vector database for efficient similarity search
- **Semantic Search**: HuggingFace embeddings for accurate document retrieval
- **LLM Integration**: Local LLM support via Ollama for answer generation
- **Modular Architecture**: Easy to extend and customize components

## 🏗️ Architecture

```
DataSage
├── Ingestion Layer     → Load and chunk documents
├── Indexing Layer      → Embed and store in vector database
├── Query Layer         → Retrieve relevant context and generate answers
└── RAG Pipeline        → End-to-end question answering system
```

## 📋 Prerequisites

- Python 3.8+
- [Ollama](https://ollama.com/download) (for local LLM inference)

## 🚀 Installation

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/datasage.git
cd datasage
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

**Required packages:**
```txt
langchain
langchain-huggingface
langchain-chroma
langchain-ollama
langchain-core
chromadb
sentence-transformers
pypdf
```

### 3. Install Ollama

Download and install Ollama from [ollama.com](https://ollama.com/download)

Pull a model:
```bash
ollama pull llama3.1
```

Verify installation:
```bash
ollama run llama3.1
```

## 📖 Usage

### Basic Indexing

```python
from datasage.indexing.index_engine import IndexingEngine

# Initialize the indexing engine
indexer = IndexingEngine(
    persist_dir="./my_index",
    embedding_model="all-MiniLM-L6-v2",
    chunk_size=1000,
    overlap=200
)

# Index a document
indexer.index("path/to/your/document.csv")

# Search the index
results = indexer.search("Your query here", k=5)
print(results)
```

### Full RAG Pipeline

```python
from datasage.indexing.index_engine import IndexingEngine
from datasage.query.retriever import Retriever
from datasage.indexing.embedder import Embedder
from langchain_ollama import OllamaLLM
from langchain_core.documents import Document
from typing import List

# Simple LLM Generator
class LLMGenerator:
    def __init__(self, model: str = "llama3.1"):
        self.llm = OllamaLLM(model=model)
    
    def generate_answer(self, question: str, context_docs: List[Document]) -> str:
        context_str = "\n\n".join(
            f"Content: {doc.page_content}" for doc in context_docs
        )
        prompt = f"""Use the following context to answer the question.

Context:
{context_str}

Question: {question}

Answer:"""
        return self.llm.invoke(prompt)

# Initialize components
indexer = IndexingEngine(persist_dir="./my_index")
embedder = Embedder()
retriever = Retriever(
    vector_store=indexer.vector_store,
    embedder=embedder
)
generator = LLMGenerator()

# Ask questions
question = "What is in the dataset?"
docs = retriever.retrieve(question, k=3)
answer = generator.generate_answer(question, docs)
print(answer)
```

## 📁 Project Structure

```
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
```

## 🧪 Running Tests

```bash
# Test basic indexing
python test.py

# Test full RAG pipeline
python test_full_rag.py
```

## 🔧 Configuration

### Indexing Engine Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `persist_dir` | `"./datasage_store"` | Directory for vector database |
| `embedding_model` | `"all-MiniLM-L6-v2"` | HuggingFace embedding model |
| `chunk_size` | `1000` | Maximum characters per chunk |
| `overlap` | `200` | Overlapping characters between chunks |

### Supported File Formats

- **CSV**: Loaded with metadata for each row
- **PDF**: Extracted page by page
- **TXT**: Loaded as single document

## 🎯 Use Cases

- **Document Q&A**: Query large documents using natural language
- **Knowledge Base Search**: Build searchable knowledge bases
- **Customer Support**: Answer questions from documentation
- **Research Assistant**: Extract information from academic papers
- **Code Documentation**: Query codebases and technical docs

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built with [LangChain](https://langchain.com/)
- Embeddings powered by [HuggingFace](https://huggingface.co/)
- Vector storage by [ChromaDB](https://www.trychroma.com/)
- Local LLM inference via [Ollama](https://ollama.com/)

## 📧 Contact

For questions or support, please open an issue on GitHub.

---

**Made with ❤️ by the DataSage Team**