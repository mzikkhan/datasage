import sys
import os

# Add the inner folder to path (same as test.py)
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(current_dir, 'premature-datasage-v1.0-main'))

from datasage.indexing.index_engine import IndexingEngine
from datasage.query.retriever import Retriever
from datasage.indexing.embedder import Embedder
from langchain_ollama import OllamaLLM
from typing import List
from langchain_core.documents import Document

# Simple LLM Generator using LangChain
class LLMGenerator:
    def __init__(self, model: str = "llama3.1"):
        self.llm = OllamaLLM(model=model)
    
    def generate_answer(self, question: str, context_docs: List[Document]) -> str:
        context_str = "\n\n".join(
            f"Content: {doc.page_content}" 
            for doc in context_docs
        )
        prompt = f"""Use the following context to answer the question. Be concise and specific.

Context:
{context_str}

Question: {question}

Answer:"""
        return self.llm.invoke(prompt)

# 1. Index your data (reuse existing index if available)
print("Step 1: Setting up indexing engine...")
indexer = IndexingEngine(
    persist_dir="./my_index",
    embedding_model="all-MiniLM-L6-v2",
)
print("✓ Indexing engine ready\n")

# 2. Set up retrieval
print("Step 2: Setting up retriever...")
embedder = Embedder()
retriever = Retriever(
    vector_store=indexer.vector_store,
    embedder=embedder
)
print("✓ Retriever ready\n")

# 3. Set up generation with Ollama
print("Step 3: Setting up LLM generator with Ollama...")
generator = LLMGenerator(model="llama3.1")
print("✓ Generator ready\n")

# 4. Ask questions with RAG!
print("=" * 80)
print("RAG QUESTION ANSWERING")
print("=" * 80 + "\n")

questions = [
    "What fruits are in the dataset?",
    "What is the mass of the apple?",
    "Which fruit has a rating of 8?",
]

for question in questions:
    print(f"❓ Question: {question}")
    
    # Retrieve relevant context
    docs = retriever.retrieve(question, k=3)
    print(f"   Retrieved {len(docs)} relevant chunks")
    
    # Generate answer using LLM
    answer = generator.generate_answer(question, docs)
    print(f"🤖 Answer: {answer}\n")
    print("-" * 80 + "\n")

print("✅ RAG pipeline complete!")