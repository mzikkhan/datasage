# Retrieval Subpackage Documentation

## Overview

The `retrieval` subpackage provides the core functionality for retrieving relevant documents and generating answers using language models. It implements custom solutions using Python's standard library to interact with Ollama, replacing external dependencies on LangChain and LlamaIndex.

## Modules

### `generator.py`
Contains classes for generating answers using the Ollama LLM API.

### `retriever.py`
Contains classes for embedding text and retrieving relevant documents from a vector store.

### `data_models.py`
Defines shared data models used across the retrieval subpackage.

---

## Classes and Functions

### `generator.py`

#### `Ollama`
A simple Ollama client to connect to Ollama.

**Methods:**
- `__init__(model: str, base_url: str = "http://localhost:11434")` - Initialize the Ollama client with model name and base URL
- `complete(prompt: str) -> str` - Send a prompt to Ollama and return the generated response

#### `LLMGenerator`
Generates answers to questions using retrieved context documents.

**Methods:**
- `__init__(model: str = "llama3.1")` - Initialize the generator with an Ollama model
- `generate_answer(question: str, context_docs: List[Document]) -> str` - Generate an answer based on the question and context documents

---

### `retriever.py`

#### `OllamaEmbedder`
A simple Ollama embedder to embed documents and queries.

**Methods:**
- `__init__(model: str = "nomic-embed-text", base_url: str = "http://localhost:11434")` - Initialize the embedder with model name and base URL
- `embed_query(text: str) -> List[float]` - Embed a single query text and return the embedding vector
- `embed_documents(texts: List[str]) -> List[List[float]]` - Embed multiple documents and return a list of embedding vectors

#### `Retriever`
Retrieves relevant documents from a vector store based on a query.

**Methods:**
- `__init__(vector_store: VectorStore, embedder: OllamaEmbedder)` - Initialize the retriever with a vector store and embedder
- `retrieve(query: str, k: int = 5, filter: dict = None) -> List[Document]` - Retrieve top-k most relevant documents for a given query

---

### `data_models.py`

#### `Document`
A simple data class to store document content and metadata.

**Methods:**
- `__init__(page_content: str, metadata: Dict[str, Any] = None)` - Initialize a document with content and optional metadata
- `__repr__()` - Return a string representation of the document
