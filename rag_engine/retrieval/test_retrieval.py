"""
Test and demonstration file for the retrieval subpackage.

This file demonstrates the usage of the retrieval subpackage components:
- Document class for data storage
- OllamaEmbedder for generating embeddings
- Ollama for LLM completions
- LLMGenerator for context-aware answer generation

Note: This requires Ollama to be running locally with the specified models installed.
"""

from typing import List
from rag_engine.retrieval.data_models import Document
from rag_engine.retrieval.retriever import OllamaEmbedder
from rag_engine.retrieval.generator import Ollama, LLMGenerator


def test_document_creation():
    """Test basic Document creation and representation."""
    print("=" * 60)
    print("Test 1: Document Creation")
    print("=" * 60)
    
    doc = Document(
        page_content="Python is a high-level, interpreted programming language known for its simplicity and readability.",
        metadata={"source": "python_intro.txt", "page": 1, "author": "Guido van Rossum"}
    )
    
    print(f"Document created: {doc}")
    print(f"Content: {doc.page_content}")
    print(f"Metadata: {doc.metadata}")
    print()


def test_ollama_embedder():
    """Test OllamaEmbedder for generating embeddings."""
    print("=" * 60)
    print("Test 2: OllamaEmbedder")
    print("=" * 60)
    
    try:
        embedder = OllamaEmbedder(model="nomic-embed-text")
        
        # Test single query embedding
        query = "What is machine learning?"
        print(f"Embedding query: '{query}'")
        query_embedding = embedder.embed_query(query)
        print(f"Embedding dimension: {len(query_embedding)}")
        print(f"First 5 values: {query_embedding[:5]}")
        print()
        
        # Test batch document embedding
        documents = [
            "Machine learning is a subset of artificial intelligence.",
            "Neural networks are inspired by biological neurons.",
            "Deep learning uses multiple layers of neural networks."
        ]
        print(f"Embedding {len(documents)} documents...")
        doc_embeddings = embedder.embed_documents(documents)
        print(f"Generated {len(doc_embeddings)} embeddings")
        print(f"Each embedding has {len(doc_embeddings[0])} dimensions")
        print()
        
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure Ollama is running and 'nomic-embed-text' model is installed.")
        print("Install with: ollama pull nomic-embed-text")
        print()


def test_ollama_completion():
    """Test basic Ollama completion."""
    print("=" * 60)
    print("Test 3: Ollama Completion")
    print("=" * 60)
    
    try:
        ollama = Ollama(model="llama3.1")
        
        prompt = "In one sentence, what is Python programming language?"
        print(f"Prompt: {prompt}")
        print("Generating response...")
        response = ollama.complete(prompt)
        print(f"Response: {response}")
        print()
        
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure Ollama is running and 'llama3.1' model is installed.")
        print("Install with: ollama pull llama3.1")
        print()


def test_llm_generator():
    """Test LLMGenerator with context documents."""
    print("=" * 60)
    print("Test 4: LLMGenerator with Context")
    print("=" * 60)
    
    try:
        generator = LLMGenerator(model="llama3.1")
        
        # Create sample context documents
        context_docs = [
            Document(
                page_content="Python was created by Guido van Rossum and first released in 1991.",
                metadata={"source": "history.txt"}
            ),
            Document(
                page_content="Python emphasizes code readability with significant indentation.",
                metadata={"source": "features.txt"}
            ),
            Document(
                page_content="Python supports multiple programming paradigms including procedural, object-oriented, and functional programming.",
                metadata={"source": "paradigms.txt"}
            )
        ]
        
        question = "Who created Python and when?"
        print(f"Question: {question}")
        print(f"Context: {len(context_docs)} documents")
        print("Generating answer...")
        
        answer = generator.generate_answer(question, context_docs)
        print(f"Answer: {answer}")
        print()
        
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure Ollama is running and 'llama3.1' model is installed.")
        print()


def test_complete_rag_workflow():
    """Demonstrate a complete RAG workflow simulation."""
    print("=" * 60)
    print("Test 5: Complete RAG Workflow (Simulated)")
    print("=" * 60)
    
    try:
        # Initialize components
        embedder = OllamaEmbedder(model="nomic-embed-text")
        generator = LLMGenerator(model="llama3.1")
        
        # Sample document collection
        documents = [
            Document(
                page_content="Machine learning is a method of data analysis that automates analytical model building.",
                metadata={"source": "ml_basics.txt", "section": "intro"}
            ),
            Document(
                page_content="Supervised learning algorithms learn from labeled training data to make predictions.",
                metadata={"source": "ml_basics.txt", "section": "supervised"}
            ),
            Document(
                page_content="Unsupervised learning finds hidden patterns in unlabeled data.",
                metadata={"source": "ml_basics.txt", "section": "unsupervised"}
            ),
            Document(
                page_content="Deep learning is a subset of machine learning using neural networks with multiple layers.",
                metadata={"source": "deep_learning.txt", "section": "overview"}
            )
        ]
        
        # Simulate retrieval (in real scenario, this would use vector similarity)
        query = "What is supervised learning?"
        print(f"Query: {query}")
        
        # Simple keyword-based retrieval for demonstration
        # In production, this would use embedder + vector store
        print("Embedding query...")
        query_embedding = embedder.embed_query(query)
        print(f"Query embedded to {len(query_embedding)} dimensions")
        
        # Simulate retrieving top 2 relevant documents
        relevant_docs = [doc for doc in documents if "supervised" in doc.page_content.lower()][:2]
        print(f"Retrieved {len(relevant_docs)} relevant documents")
        
        # Generate answer
        print("Generating answer...")
        answer = generator.generate_answer(query, relevant_docs)
        print(f"\nFinal Answer: {answer}")
        print()
        
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure Ollama is running with required models installed.")
        print()


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("RETRIEVAL SUBPACKAGE - TEST AND DEMONSTRATION")
    print("=" * 60)
    print()
    
    test_document_creation()
    test_ollama_embedder()
    test_ollama_completion()
    test_llm_generator()
    test_complete_rag_workflow()
    
    print("=" * 60)
    print("All tests completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
