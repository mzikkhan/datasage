import sys
import os

# Get the directory where test.py is located
current_dir = os.path.dirname(os.path.abspath(__file__))
# Add the inner premature-datasage-v1.0-main folder to path
sys.path.insert(0, os.path.join(current_dir, 'premature-datasage-v1.0-main'))

print(f"Added to path: {os.path.join(current_dir, 'premature-datasage-v1.0-main')}")
print(f"sys.path: {sys.path[0]}")

from datasage.indexing.index_engine import IndexingEngine

# Initialize the indexing engine
indexer = IndexingEngine(
    persist_dir="./my_index",
    embedding_model="all-MiniLM-L6-v2",
    chunk_size=500,
    overlap=50,
)

# Index your CSV file
print("Indexing fruits_processed.csv...")
chunks = indexer.index("fruits_processed.csv")
print(f"Created {len(chunks)} chunks")

# Search the index
print("\nSearching for fruits...")
results = indexer.search("What fruits are in the dataset?", k=3)

for i, doc in enumerate(results, 1):
    print(f"\n--- Result {i} ---")
    print(f"Content: {doc.page_content[:200]}...")
    print(f"Metadata: {doc.metadata}")