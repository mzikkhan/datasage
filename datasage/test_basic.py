"""Test basic indexing functionality."""
print("Starting import test...")

try:
    import datasage
    print(f"datasage module imported")
    print(f"Available in datasage: {dir(datasage)}")
except Exception as e:
    print(f"Failed to import datasage: {e}")
    import traceback
    traceback.print_exc()

try:
    from datasage import IndexingEngine
    print(f"IndexingEngine imported: {IndexingEngine}")
except Exception as e:
    print(f"Failed to import IndexingEngine: {e}")
    import traceback
    traceback.print_exc()

# Only run this if IndexingEngine imported successfully
try:
    # Initialize
    indexer = IndexingEngine(
        persist_dir="./my_index",
        embedding_model="all-MiniLM-L6-v2",
        chunk_size=500,
        overlap=50,
    )

    # Index
    print("Indexing fruits_processed.csv...")
    chunks = indexer.index("fruits_processed.csv")
    print(f"✓ Created {len(chunks)} chunks\n")

    # Search
    print("Searching...")
    results = indexer.search("What fruits are in the dataset?", k=3)

    for i, doc in enumerate(results, 1):
        print(f"--- Result {i} ---")
        print(f"Content: {doc.page_content[:150]}...")
        print(f"Source: {doc.metadata.get('source')}\n")
except NameError:
    print("Skipping test - IndexingEngine not available")
except Exception as e:
    print(f"Error during test: {e}")
    import traceback
    traceback.print_exc()