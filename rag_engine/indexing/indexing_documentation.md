# Indexing Module Documentation

## Overview

The indexing module is responsible for converting documents into searchable vector representations. It consists of three main components that work together to enable semantic search capabilities.

---

## Components

### 1. Embedder (`embedder.py`)

The Embedder converts text into numerical vector representations (embeddings) that capture semantic meaning.

#### Class: `Embedder`

**Purpose:** Transform text into embeddings using HuggingFace models with custom preprocessing.

**Initialization:**

```python
embedder = Embedder(model_name="all-MiniLM-L6-v2")
```

**Parameters:**

- `model_name` (str): HuggingFace model identifier. Default: "all-MiniLM-L6-v2"

**Custom Features:**

- Text preprocessing for consistency
- Batch processing with progress tracking
- Efficient embedding generation

#### Methods

##### `embed_query(text: str) -> List[float]`

Embeds a single text query into a vector representation.

**How it works:**

1. Preprocesses the text (removes extra whitespace, strips edges)
2. Generates embedding using the HuggingFace model
3. Returns a list of floating-point numbers (typically 384 dimensions)

**Parameters:**

- `text` (str): Text to embed

**Returns:**

- List[float]: Embedding vector

**Example:**

```python
embedder = Embedder()
embedding = embedder.embed_query("What is machine learning?")
print(f"Embedding dimension: {len(embedding)}")  # 384
print(f"First 5 values: {embedding[:5]}")
```

**Custom Logic:**

- `_preprocess_text()`: Normalizes whitespace and removes leading/trailing spaces for consistent embeddings

---

##### `embed_documents(texts: List[str], show_progress: bool = False) -> List[List[float]]`

Embeds multiple documents in batch with optional progress tracking.

**How it works:**

1. Iterates through each text in the list
2. Calls `embed_query()` for each text
3. Optionally displays progress every 10 documents
4. Returns list of all embeddings

**Parameters:**

- `texts` (List[str]): List of texts to embed
- `show_progress` (bool): Whether to show progress updates. Default: False

**Returns:**

- List[List[float]]: List of embedding vectors

**Example:**

```python
texts = ["Document 1", "Document 2", "Document 3"]
embeddings = embedder.embed_documents(texts, show_progress=True)
# Output: "Embedding progress: 3/3 documents"
#         "Batch complete: 3 documents embedded"
```

**Custom Logic:**

- Progress tracking for large batches
- Displays updates every 10 documents
- Final completion message

---

##### `get_embedding_dimension() -> int`

Returns the dimensionality of the embedding vectors.

**How it works:**

1. Generates a test embedding with the word "test"
2. Returns the length of the resulting vector

**Returns:**

- int: Embedding dimension (typically 384 for all-MiniLM-L6-v2)

**Example:**

```python
dim = embedder.get_embedding_dimension()
print(f"This model produces {dim}-dimensional embeddings")
```

---

### 2. VectorStore (`vector_store.py`)

The VectorStore manages the storage and retrieval of document embeddings using ChromaDB.

#### Class: `VectorStore`

**Purpose:** Store and search document embeddings with custom metadata tracking and analytics.

**Initialization:**

```python
vs = VectorStore(
    embedding_model=embedder.model,
    persist_dir="./my_vector_db"
)
```

**Parameters:**

- `embedding_model`: The embedding function from an Embedder instance
- `persist_dir` (str, optional): Directory for persistent storage. Default: None (in-memory)

**Custom Features:**

- Automatic document ID assignment
- Metadata indexing and tracking
- Source-based organization
- Search analytics
- Relevance scoring

#### Internal State

The VectorStore maintains several custom tracking structures:

- `_doc_count`: Total number of documents indexed
- `_metadata_index`: Dictionary mapping doc_id to metadata (source, timestamp, content_length)
- `_source_index`: Dictionary mapping source to list of doc_ids
- `_search_stats`: Analytics on searches (total searches, unique queries, popular sources)

#### Methods

##### `add_documents(docs: List[Document]) -> List[str]`

Adds documents to the vector store with custom metadata tracking.

**How it works:**

1. Assigns unique document ID (format: `doc_000001`, `doc_000002`, etc.)
2. Extracts source from document metadata
3. Records provenance information (source, timestamp, content length)
4. Indexes documents by source for fast filtering
5. Adds custom metadata to each document (doc_id, indexed_at timestamp)
6. Stores in ChromaDB vector database
7. Returns list of assigned document IDs

**Parameters:**

- `docs` (List[Document]): List of LangChain Document objects to add

**Returns:**

- List[str]: Assigned document IDs

**Example:**

```python
from langchain_core.documents import Document

docs = [
    Document(
        page_content="Python is a programming language",
        metadata={"source": "intro.txt"}
    )
]

doc_ids = vs.add_documents(docs)
print(doc_ids)  # ['doc_000000']
```

**Custom Logic:**

- Sequential ID generation ensures unique identifiers
- Metadata indexing enables fast lookups by source
- Timestamp tracking for provenance
- Content length tracking for analytics

---

##### `search(query: str, k: int = 5, filter: dict = None) -> List[Document]`

Searches for documents similar to the query with custom analytics.

**How it works:**

1. Records search query for analytics
2. Performs similarity search using ChromaDB
3. Enhances results with ranking metadata:
   - `search_rank`: Position in results (1, 2, 3, ...)
   - `relevance_score`: Inverse rank score (1.0, 0.5, 0.33, ...)
4. Updates source popularity statistics
5. Returns enhanced documents

**Parameters:**

- `query` (str): Search query text
- `k` (int): Number of results to return. Default: 5
- `filter` (dict, optional): Metadata filter for results

**Returns:**

- List[Document]: Ranked list of relevant documents

**Example:**

```python
results = vs.search("machine learning", k=3)

for doc in results:
    print(f"Rank {doc.metadata['search_rank']}: {doc.page_content[:50]}")
    print(f"Relevance: {doc.metadata['relevance_score']}")
```

**Custom Logic:**

- Tracks all search queries for analytics
- Adds ranking metadata to results
- Monitors which sources are most frequently retrieved
- Simple relevance scoring (1/rank)

---

##### `search_by_source(query: str, source: str, k: int = 5) -> List[Document]`

Searches within documents from a specific source.

**How it works:**

1. Creates a metadata filter for the specified source
2. Calls `search()` with the source filter
3. Returns only documents from that source

**Parameters:**

- `query` (str): Search query
- `source` (str): Source identifier to filter by
- `k` (int): Number of results. Default: 5

**Returns:**

- List[Document]: Documents from the specified source

**Example:**

```python
# Search only in documents from 'report.pdf'
results = vs.search_by_source("findings", source="report.pdf", k=3)
```

**Custom Logic:**

- Leverages the source index for efficient filtering

---

##### `get_sources() -> List[str]`

Returns all unique sources in the vector store.

**How it works:**

- Returns keys from the internal `_source_index` dictionary

**Returns:**

- List[str]: List of source identifiers

**Example:**

```python
sources = vs.get_sources()
print(f"Documents from: {sources}")
# Output: ['intro.txt', 'report.pdf', 'data.csv']
```

---

##### `get_document_count_by_source(source: str) -> int`

Returns the number of documents from a specific source.

**How it works:**

- Looks up the source in `_source_index`
- Returns the length of the document ID list

**Parameters:**

- `source` (str): Source identifier

**Returns:**

- int: Number of documents from that source

**Example:**

```python
count = vs.get_document_count_by_source("report.pdf")
print(f"report.pdf has {count} documents")
```

---

##### `get_statistics() -> Dict`

Returns comprehensive analytics about the vector store.

**How it works:**

1. Calculates source distribution (documents per source)
2. Identifies top 5 most-searched sources
3. Computes average document length
4. Aggregates all search statistics

**Returns:**

- Dict: Statistics dictionary with keys:
  - `total_documents`: Total number of indexed documents
  - `unique_sources`: Number of unique sources
  - `source_distribution`: Dict mapping source to document count
  - `total_searches`: Total search queries performed
  - `unique_queries`: Number of unique queries
  - `avg_results_per_search`: Average results returned per search
  - `top_searched_sources`: Most frequently retrieved sources
  - `avg_document_length`: Average characters per document
  - `persist_directory`: Storage location

**Example:**

```python
stats = vs.get_statistics()
print(f"Total documents: {stats['total_documents']}")
print(f"Total searches: {stats['total_searches']}")
print(f"Average doc length: {stats['avg_document_length']}")
```

**Custom Logic:**

- Aggregates multiple metrics from internal tracking
- Provides insights into usage patterns
- Helps identify popular content

---

##### `get_document_info(doc_id: str) -> Optional[Dict]`

Retrieves detailed metadata for a specific document.

**Parameters:**

- `doc_id` (str): Document identifier

**Returns:**

- Optional[Dict]: Metadata dictionary or None if not found

**Example:**

```python
info = vs.get_document_info("doc_000001")
print(f"Source: {info['source']}")
print(f"Indexed at: {info['timestamp']}")
print(f"Length: {info['content_length']} chars")
```

---

##### `list_documents_by_source(source: str) -> List[str]`

Lists all document IDs from a specific source.

**Parameters:**

- `source` (str): Source identifier

**Returns:**

- List[str]: Document IDs from that source

**Example:**

```python
doc_ids = vs.list_documents_by_source("report.pdf")
print(f"Document IDs from report.pdf: {doc_ids}")
```

---

### 3. IndexingEngine (`index_engine.py`)

The IndexingEngine orchestrates the complete indexing pipeline, coordinating all components.

#### Class: `IndexingEngine`

**Purpose:** High-level API for indexing documents with validation, progress tracking, and error handling.

**Initialization:**

```python
indexer = IndexingEngine(
    persist_dir="./my_index",
    embedding_model="all-MiniLM-L6-v2",
    chunk_size=1000,
    overlap=200
)
```

**Parameters:**

- `persist_dir` (str): Directory for vector database. Default: "./datasage_store"
- `embedding_model` (str): HuggingFace model name. Default: "all-MiniLM-L6-v2"
- `chunk_size` (int): Maximum characters per chunk. Default: 1000
- `overlap` (int): Overlapping characters between chunks. Default: 200

**Custom Features:**

- Automatic file type detection
- File validation (existence, readability, size)
- Duplicate detection
- Progress tracking with verbose output
- Batch processing with error recovery
- Complete indexing history

#### Internal State

- `_indexed_files`: Set of successfully indexed file paths
- `_failed_files`: Dictionary mapping failed files to error messages
- `_indexing_history`: List of all indexing operations with details
- `_supported_extensions`: Set of supported file types (.pdf, .csv, .txt)

#### Methods

##### `index(file_path: str, metadata: Optional[dict] = None, force_reindex: bool = False, verbose: bool = True) -> List[Document]`

Indexes a single file through the complete pipeline.

**How it works:**

1. **Validation**: Checks file existence, readability, type, and size
2. **Duplicate Check**: Verifies if file was already indexed
3. **Loading**: Selects appropriate loader (PDF/CSV/TXT) and loads documents
4. **Metadata**: Applies custom metadata to all documents
5. **Chunking**: Splits documents into smaller chunks
6. **Embedding & Storage**: Embeds chunks and stores in vector database
7. **Tracking**: Records operation in history with timestamp and duration

**Parameters:**

- `file_path` (str): Path to file to index
- `metadata` (dict, optional): Additional metadata to attach
- `force_reindex` (bool): Re-index even if already processed. Default: False
- `verbose` (bool): Print progress messages. Default: True

**Returns:**

- List[Document]: Created document chunks

**Example:**

```python
chunks = indexer.index(
    "report.pdf",
    metadata={"department": "research"},
    verbose=True
)
print(f"Created {len(chunks)} chunks")
```

**Custom Logic:**

- Pre-flight validation prevents processing invalid files
- Duplicate detection saves time and prevents redundancy
- Progress tracking provides visibility into long operations
- Complete history enables auditing

---

##### `batch_index(file_paths: List[str], metadata: Optional[dict] = None, continue_on_error: bool = True, verbose: bool = True) -> Dict[str, List[Document]]`

Indexes multiple files with error recovery.

**How it works:**

1. Iterates through each file
2. Attempts to index each one
3. On error: logs failure and continues (if `continue_on_error=True`)
4. Returns dictionary mapping file paths to their chunks
5. Prints summary statistics

**Parameters:**

- `file_paths` (List[str]): List of file paths
- `metadata` (dict, optional): Metadata for all files
- `continue_on_error` (bool): Continue if a file fails. Default: True
- `verbose` (bool): Print progress. Default: True

**Returns:**

- Dict[str, List[Document]]: Maps file paths to created chunks

**Example:**

```python
files = ["doc1.pdf", "doc2.csv", "doc3.txt"]
results = indexer.batch_index(files, verbose=True)

for file, chunks in results.items():
    print(f"{file}: {len(chunks)} chunks")
```

**Custom Logic:**

- Error recovery allows batch operations to complete even with failures
- Per-file progress tracking
- Aggregated statistics

---

##### `search(query: str, k: int = 5, filter: Optional[dict] = None) -> List[Document]`

Convenience method to search indexed documents.

**How it works:**

- Delegates to `vector_store.search()`

**Parameters:**

- `query` (str): Search query
- `k` (int): Number of results. Default: 5
- `filter` (dict, optional): Metadata filter

**Returns:**

- List[Document]: Search results

**Example:**

```python
results = indexer.search("machine learning", k=3)
```

---

##### `get_indexed_files() -> List[str]`

Returns list of successfully indexed files.

**Returns:**

- List[str]: File paths

---

##### `get_failed_files() -> Dict[str, str]`

Returns dictionary of failed files and their errors.

**Returns:**

- Dict[str, str]: Maps file path to error message

**Example:**

```python
failed = indexer.get_failed_files()
for file, error in failed.items():
    print(f"{file} failed: {error}")
```

---

##### `get_indexing_history() -> List[Dict]`

Returns complete history of indexing operations.

**Returns:**

- List[Dict]: List of operation records containing:
  - `file_path`: File that was indexed
  - `timestamp`: When operation started
  - `status`: "success" or "failed"
  - `duration_seconds`: How long it took
  - `documents_loaded`: Number of documents loaded
  - `chunks_created`: Number of chunks created
  - `error`: Error message (if failed)

**Example:**

```python
history = indexer.get_indexing_history()
for entry in history:
    print(f"{entry['file_path']}: {entry['status']}")
```

---

##### `get_system_statistics() -> Dict`

Returns comprehensive system-wide statistics.

**How it works:**

1. Aggregates indexing history statistics
2. Includes configuration details
3. Pulls vector store statistics
4. Combines into single report

**Returns:**

- Dict: Statistics with keys:
  - `indexing`: Files indexed, failed, total chunks, timing
  - `configuration`: Chunk size, overlap, model info
  - `vector_store`: All vector store statistics

**Example:**

```python
stats = indexer.get_system_statistics()
print(f"Files indexed: {stats['indexing']['files_indexed']}")
print(f"Total chunks: {stats['indexing']['total_chunks_created']}")
print(f"Model: {stats['configuration']['embedding_model']}")
```

**Custom Logic:**

- Aggregates metrics across all components
- Provides single view of system health
- Useful for monitoring and debugging

---

##### `reset_history()`

Clears indexing history (keeps indexed files tracked).

**Example:**

```python
indexer.reset_history()
```

---

## Usage Flow

### Complete Example

```python
from rag_engine.indexing.embedder import Embedder
from rag_engine.indexing.vector_store import VectorStore
from rag_engine.indexing.index_engine import IndexingEngine

# 1. Initialize components
indexer = IndexingEngine(
    persist_dir="./my_database",
    chunk_size=500,
    overlap=50
)

# 2. Index documents
chunks = indexer.index("research_paper.pdf", verbose=True)
print(f"Indexed: {len(chunks)} chunks")

# 3. Search
results = indexer.search("neural networks", k=5)
for i, doc in enumerate(results, 1):
    print(f"{i}. {doc.page_content[:100]}...")

# 4. Get statistics
stats = indexer.get_system_statistics()
print(f"Total documents: {stats['vector_store']['total_documents']}")
```

---

## Custom Logic Summary

### What Makes This OOP (Not Just Library Wrappers)?

**Embedder:**

- Custom preprocessing pipeline
- Batch progress tracking
- Dimension introspection

**VectorStore:**

- Automatic document ID generation
- Metadata indexing by source
- Search analytics and tracking
- Relevance scoring
- Source-based filtering

**IndexingEngine:**

- File validation (type, existence, size, permissions)
- Duplicate detection
- Automatic loader selection
- Complete operation history
- Batch processing with error recovery
- Progress tracking
- System-wide statistics aggregation

These custom features add significant value beyond the underlying libraries (LangChain, ChromaDB, HuggingFace), demonstrating real OOP design with state management, business logic, and orchestration.
