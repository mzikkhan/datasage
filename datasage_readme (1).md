# DataSage 🧙‍♂️

DataSage is a fully local, extensible Retrieval-Augmented Generation (RAG) engine designed for flexible data ingestion, semantic search, and context-aware question answering. It supports PDF, TXT, CSV, and XLSX files and uses modern open-source LLM tooling such as Ollama - with zero cloud dependencies.

## 🌟 Features

- **Document Ingestion**: Support for multiple file formats (CSV, PDF, TXT)
- **Intelligent Chunking**: Configurable text splitting with overlap for context preservation
- **Vector Storage**: ChromaDB-backed vector database for efficient similarity search
- **Semantic Search**: HuggingFace embeddings for accurate document retrieval
- **LLM Integration**: Local LLM support via Ollama for answer generation
- **Modular Architecture**: Easy to extend and customize components
- **Zero Cloud Dependencies**: Runs completely offline on your local machine
- **High-Level API**: Simple `RagEngine` class for quick setup
- **Component-Level Access**: Use individual components for custom workflows

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

### 2. Install the package

```bash
# Install in development mode (recommended for development)
pip install -e .

# Or install dependencies manually
pip install -r requirements.txt
```

### 3. Install and setup Ollama

**Download Ollama:**
- Visit [ollama.com](https://ollama.com/download)
- Download and install for your OS

**Pull a model:**
```bash
ollama pull llama3.1
```

**Verify installation:**
```bash
ollama run llama3.1
# Type a message to test, then type /bye to exit
```

## 📖 Usage

### Quick Start with RagEngine (Recommended)

The easiest way to use DataSage is with the high-level `RagEngine` API:

```python
from datasage import RagEngine

# Initialize RAG engine (automatically indexes your documents)
rag = RagEngine(
    data_files="your_document.csv",  # Can be a string or list of files
    model_name="llama3.1"             # Ollama model to use
)

# Ask questions
answer = rag.query("What is in the dataset?")
print(answer)

# Ask multiple questions
questions = [
    "What are the main topics?",
    "Summarize the key findings",
    "What data is available?"
]

for question in questions:
    answer = rag.query(question, top_k=5)  # top_k = number of chunks to retrieve
    print(f"Q: {question}")
    print(f"A: {answer}\n")
```

### Basic Indexing Only

If you just want to index documents and search without LLM generation:

```python
from datasage import IndexingEngine

# Initialize the indexing engine
indexer = IndexingEngine(
    persist_dir="./my_index",           # Where to store the vector database
    embedding_model="all-MiniLM-L6-v2", # HuggingFace embedding model
    chunk_size=1000,                     # Characters per chunk
    overlap=200                          # Overlap between chunks
)

# Index a document
chunks = indexer.index("path/to/document.pdf")
print(f"Created {len(chunks)} chunks")

# Search (returns Document objects with content and metadata)
results = indexer.search("Your search query", k=5)

for doc in results:
    print(f"Content: {doc.page_content}")
    print(f"Metadata: {doc.metadata}\n")
```

### Manual RAG Pipeline (Advanced)

For full control over each component:

```python
from datasage import IndexingEngine, Retriever, LLMGenerator, Embedder

# 1. Index your documents
indexer = IndexingEngine(persist_dir="./my_index")
indexer.index("document.csv")

# 2. Set up retriever
embedder = Embedder()
retriever = Retriever(
    vector_store=indexer.vector_store,
    embedder=embedder
)

# 3. Set up LLM generator
generator = LLMGenerator(model="llama3.1")

# 4. Query pipeline
question = "What is the main finding?"

# Retrieve relevant context
docs = retriever.retrieve(question, k=3)

# Generate answer
answer = generator.generate_answer(question, docs)
print(answer)
```

### Multiple File Types

DataSage automatically detects file types:

```python
from datasage import RagEngine

# Index multiple files at once
rag = RagEngine(
    data_files=[
        "report.pdf",
        "data.csv",
        "notes.txt"
    ],
    model_name="llama3.1"
)

answer = rag.query("Summarize all the information")
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
├── retrieval/
│   ├── retriever.py         # Semantic search retriever
│   └── generator.py         # LLM answer generation
└── __init__.py              # Package exports and RagEngine class
```

## 🔧 Configuration

### IndexingEngine Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `persist_dir` | `"./datasage_store"` | Directory for vector database |
| `embedding_model` | `"all-MiniLM-L6-v2"` | HuggingFace embedding model |
| `chunk_size` | `1000` | Maximum characters per chunk |
| `overlap` | `200` | Overlapping characters between chunks |

### RagEngine Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `data_files` | Required | Path(s) to files (string or list) |
| `model_name` | `"llama3.1"` | Ollama model for generation |
| `persist_dir` | `"./vector_db"` | Directory for vector database |

### Supported File Formats

- **CSV**: Loaded row-by-row with all fields in metadata
- **PDF**: Extracted page-by-page with page numbers
- **TXT**: Loaded as single document

## 🎯 Use Cases

- **Document Q&A**: Query large documents using natural language
- **Knowledge Base Search**: Build searchable knowledge bases from multiple sources
- **Customer Support**: Answer questions from documentation
- **Research Assistant**: Extract information from academic papers
- **Code Documentation**: Query codebases and technical docs
- **Data Analysis**: Ask questions about datasets in natural language

## 🧪 Running Tests

DataSage includes several test files to verify functionality:

```bash
# Test basic indexing and search
python test_basic.py

# Test high-level RAG API
python test_rag.py

# Test manual RAG pipeline
python test_manual_rag.py

# Test Ollama connection
python test_ollama.py
```

## 📚 Example Outputs

### Indexing Example
```
Indexing fruits_processed.csv...
✓ Created 10 chunks

Searching...
--- Result 1 ---
Content: Fruit Name: Apple
Mass(g): 200
Colour: Red
Rating: 8
```

### RAG Query Example
```
❓ What fruits are in the dataset?
🤖 Based on the data provided, the fruits in the dataset are Apple, 
Banana, Orange, and Grape. Each fruit has information about mass, 
colour, rating, and various scores.
```

## 🛠️ Development

### Installing in Development Mode

```bash
# Clone the repo
git clone https://github.com/yourusername/datasage.git
cd datasage

# Install in editable mode
pip install -e .

# Make changes to the code
# Changes are immediately reflected without reinstalling
```

### Project Dependencies

See `requirements.txt` for the full list. Key dependencies:
- `langchain` - LLM application framework
- `langchain-huggingface` - HuggingFace embeddings
- `langchain-chroma` - ChromaDB integration
- `langchain-ollama` - Ollama LLM integration
- `chromadb` - Vector database
- `sentence-transformers` - Embedding models
- `pypdf` - PDF parsing

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 🐛 Troubleshooting

### "No module named 'datasage'"
```bash
# Make sure you're in the project directory
cd /path/to/datasage

# Install the package
pip install -e .
```

### "Connection refused" when using Ollama
```bash
# Make sure Ollama is running
ollama serve

# Or on Windows, start Ollama from the Start Menu
```

### "Model not found"
```bash
# Pull the model you want to use
ollama pull llama3.1

# List available models
ollama list
```

### Import errors with yellow highlights
- Ensure all subfolders have `__init__.py` files
- Run `pip install -e .` to reinstall the package
- Check that folder is named `datasage/` not `rag_engine/`

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