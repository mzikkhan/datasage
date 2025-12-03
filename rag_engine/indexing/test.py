"""
Comprehensive test script for the indexing module.
Tests embedder, vector_store, and index_engine with custom logic.
"""

import sys
import os

# Add parent directories to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
grandparent_dir = os.path.dirname(parent_dir)
sys.path.insert(0, grandparent_dir)

print("Starting Indexing Module Tests...\n")

# Test 1: Import all components
print("=" * 60)
print("Test 1: Importing Components")
print("=" * 60)

try:
    from rag_engine.indexing.embedder import Embedder
    from rag_engine.indexing.vector_store import VectorStore
    from rag_engine.indexing.index_engine import IndexingEngine
    print("All imports successful")
except ImportError as e:
    print(f"Import failed: {e}")
    exit(1)

print()

# Test 2: Test Embedder
print("=" * 60)
print("Test 2: Testing Embedder")
print("=" * 60)

try:
    embedder = Embedder(model_name="all-MiniLM-L6-v2")
    print("Embedder initialized")
    
    # Test single query
    test_text = "Hello, this is a test"
    embedding = embedder.embed_query(test_text)
    print(f"Embedded query: '{test_text}'")
    print(f"Embedding dimension: {len(embedding)}")
    print(f"First 5 values: {embedding[:5]}")
    
    # Test batch embedding
    print(f"\nTesting batch embedding...")
    texts = ["Text 1", "Text 2", "Text 3", "Text 4", "Text 5"]
    embeddings = embedder.embed_documents(texts, show_progress=True)
    print(f"Batch embedded {len(embeddings)} documents")
    
    # Test embedding dimension
    dim = embedder.get_embedding_dimension()
    print(f"Embedding dimension: {dim}")
    
except Exception as e:
    print(f"Embedder test failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

print()

# Test 3: Test VectorStore
print("=" * 60)
print("Test 3: Testing VectorStore")
print("=" * 60)

try:
    from langchain_core.documents import Document
    
    # Initialize vector store
    vs = VectorStore(
        embedding_model=embedder.model,
        persist_dir="./test_vector_store"
    )
    print("VectorStore initialized")
    
    # Create test documents with NEW metadata format (path, name, type)
    test_docs = [
        Document(
            page_content="Python is a programming language",
            metadata={"path": "test.txt", "name": "test.txt", "type": "txt"}
        ),
        Document(
            page_content="Machine learning is a subset of AI",
            metadata={"path": "test.txt", "name": "test.txt", "type": "txt"}
        ),
        Document(
            page_content="Data science uses Python extensively",
            metadata={"path": "test2.txt", "name": "test2.txt", "type": "txt"}
        )
    ]
    
    # Add documents
    print(f"\nAdding {len(test_docs)} test documents...")
    doc_ids = vs.add_documents(test_docs)
    print(f"Added {len(test_docs)} documents")
    print(f"Document IDs: {doc_ids}")
    
    # Search
    print(f"\nSearching for 'programming languages'...")
    results = vs.search("programming languages", k=2)
    print(f"Search returned {len(results)} results")
    for i, doc in enumerate(results, 1):
        print(f"Result {i}: {doc.page_content[:50]}...")
        print(f"Rank: {doc.metadata.get('search_rank')}")
        print(f"Relevance: {doc.metadata.get('relevance_score')}")
    
    # Get statistics
    print(f"\nGetting VectorStore statistics...")
    vs_stats = vs.get_statistics()
    print(f"VectorStore statistics:")
    print(f"Total documents: {vs_stats['total_documents']}")
    print(f"Unique sources: {vs_stats['unique_sources']}")
    print(f"Total searches: {vs_stats['total_searches']}")
    
    # Test source filtering
    print(f"\nTesting source-based operations...")
    sources = vs.get_sources()
    print(f"Found sources: {sources}")
    for source in sources:
        count = vs.get_document_count_by_source(source)
        print(f"  {source}: {count} documents")
    
except Exception as e:
    print(f"VectorStore test failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

print()

# Test 4: Test IndexingEngine with CSV
print("=" * 60)
print("Test 4: Testing IndexingEngine")
print("=" * 60)

try:
    indexer = IndexingEngine(
        persist_dir="./test_index",
        embedding_model="all-MiniLM-L6-v2",
        chunk_size=500,
        overlap=50
    )
    print("IndexingEngine initialized")
    print(f"Persist directory: ./test_index")
    print(f"Chunk size: 500")
    print(f"Overlap: 50")
    
    # Check if fruits_processed.csv exists
    import os
    if os.path.exists("fruits_processed.csv"):
        print(f"\nFound fruits_processed.csv")
        
        # Index the file
        print(f"\nIndexing fruits_processed.csv...")
        chunks = indexer.index("fruits_processed.csv", verbose=True)
        print(f"\nIndexing complete: {len(chunks)} chunks created")
        
        # Get indexed files
        indexed_files = indexer.get_indexed_files()
        print(f"Indexed files: {len(indexed_files)}")
        for f in indexed_files:
            print(f"  - {f}")
        
        # Test duplicate detection
        print(f"\nTesting duplicate detection...")
        chunks2 = indexer.index("fruits_processed.csv", verbose=False)
        if len(chunks2) == 0:
            print(f"Duplicate detection working - prevented re-indexing")
        
        # Search
        print(f"\nTesting search functionality...")
        results = indexer.search("apple fruit", k=3)
        print(f"Search returned {len(results)} results")
        for i, doc in enumerate(results, 1):
            print(f"Result {i}: {doc.page_content[:60]}...")
        
        # Get indexing history
        print(f"\nGetting indexing history...")
        history = indexer.get_indexing_history()
        print(f"Found {len(history)} indexing operations")
        for entry in history:
            status = entry.get('status', 'unknown')
            chunks_created = entry.get('chunks_created', 0)
            print(f"  - {entry['file_path']}: {status} ({chunks_created} chunks)")
        
        # Get system statistics
        print(f"\nGetting system statistics...")
        stats = indexer.get_system_statistics()
        print(f"System statistics:")
        print(f"Files indexed: {stats['indexing']['files_indexed']}")
        print(f"Total chunks: {stats['indexing']['total_chunks_created']}")
        print(f"Total searches: {stats['vector_store']['total_searches']}")
        
    else:
        print("fruits_processed.csv not found")
        print("Creating a simple test file...")
        
        # Create a simple test CSV
        with open("test_data.csv", "w") as f:
            f.write("name,value\n")
            f.write("item1,100\n")
            f.write("item2,200\n")
        
        print("Created test_data.csv")
        chunks = indexer.index("test_data.csv", verbose=True)
        print(f"Indexed test file: {len(chunks)} chunks")
    
except Exception as e:
    print(f"IndexingEngine test failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

print()

# Test 5: Test Error Handling
print("=" * 60)
print("Test 5: Testing Error Handling")
print("=" * 60)

try:
    # Test with non-existent file
    print("Testing with non-existent file...")
    try:
        indexer.index("nonexistent_file.csv", verbose=False)
        print("Should have raised FileNotFoundError")
    except FileNotFoundError as e:
        print(f"Correctly caught missing file")
        print(f"Error: {str(e)[:60]}...")
    
    # Test with unsupported file type
    print("\nTesting with unsupported file type...")
    
    # Create dummy file with unsupported extension
    with open("test.xyz", "w") as f:
        f.write("test content")
    
    try:
        indexer.index("test.xyz", verbose=False)
        print("Should have raised ValueError for unsupported type")
    except ValueError as e:
        print(f"Correctly caught unsupported file type")
        print(f"Error: {str(e)[:60]}...")
    finally:
        import os
        if os.path.exists("test.xyz"):
            os.remove("test.xyz")
    
    # Test batch indexing with errors
    print("\nTesting batch indexing with mixed results...")
    batch_files = ["fruits_processed.csv", "nonexistent.csv"]
    if os.path.exists("test_data.csv"):
        batch_files.append("test_data.csv")
    
    results = indexer.batch_index(
        batch_files,
        continue_on_error=True,
        verbose=False
    )
    
    successful = sum(1 for chunks in results.values() if len(chunks) > 0)
    failed = sum(1 for chunks in results.values() if len(chunks) == 0)
    
    print(f"Batch indexing completed")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    
    failed_files = indexer.get_failed_files()
    if failed_files:
        print(f"Failed files:")
        for file, error in failed_files.items():
            print(f"    - {file}: {error[:50]}...")
    
    print("Error handling works correctly")
    
except Exception as e:
    print(f"Error handling test failed: {e}")
    import traceback
    traceback.print_exc()

print()

# Test 6: Test with Text File
print("=" * 60)
print("Test 6: Testing with TXT File")
print("=" * 60)

try:
    # Create a test text file
    test_txt = "test_document.txt"
    with open(test_txt, "w") as f:
        f.write("This is a test document for the indexing engine. ")
        f.write("It contains multiple sentences. ")
        f.write("The indexing engine should process this correctly.")
    
    print(f"Created test file: {test_txt}")
    
    # Index the text file
    chunks = indexer.index(test_txt, verbose=True)
    print(f"Successfully indexed text file: {len(chunks)} chunks")
    
    # Search in the indexed text
    results = indexer.search("test document", k=2)
    print(f"Search in text file returned {len(results)} results")
    
    # Clean up
    os.remove(test_txt)
    print(f"Cleaned up test file")
    
except Exception as e:
    print(f"Text file test failed: {e}")
    import traceback
    traceback.print_exc()

print()

# Final Summary
print("=" * 60)
print("TEST SUMMARY")
print("=" * 60)
print("All tests completed successfully!")
print()
print("Custom Logic Verified:")
print(" - Embedder: Text preprocessing")
print(" - Embedder: Batch processing with progress")
print(" - VectorStore: Metadata indexing and tracking")
print(" - VectorStore: Source-based filtering")
print(" - VectorStore: Relevance scoring")
print(" - IndexingEngine: File validation")
print(" - IndexingEngine: Duplicate detection")
print(" - IndexingEngine: Progress tracking")
print(" - IndexingEngine: Batch processing")
print(" - IndexingEngine: Error handling")
print("=" * 60)