"""Test RAG pipeline with manual component setup."""
from datasage import IndexingEngine, Retriever, LLMGenerator, Embedder

# 1. Index documents
print("Step 1: Indexing...")
indexer = IndexingEngine(persist_dir="./my_index")
indexer.index("fruits_processed.csv")
print("✓ Indexed\n")

# 2. Setup retriever
print("Step 2: Setting up retriever...")
embedder = Embedder()
retriever = Retriever(
    vector_store=indexer.vector_store,
    embedder=embedder
)
print("✓ Retriever ready\n")

# 3. Setup generator
print("Step 3: Setting up generator...")
generator = LLMGenerator(model="llama3.1")
print("✓ Generator ready\n")

# 4. Query
question = "What fruits are in the dataset?"
print(f"{question}")

docs = retriever.retrieve(question, k=3)
print(f"Retrieved {len(docs)} documents")

answer = generator.generate_answer(question, docs)
print(f"{answer}")