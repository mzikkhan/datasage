"""Test full RAG pipeline with high-level API."""
from datasage import RagEngine

# Initialize RAG engine
print("Initializing RAG engine...")
rag = RagEngine(
    data_files="fruits_processed.csv",
    model_name="llama3.1"
)
print("RAG engine ready\n")

# Ask questions
questions = [
    "What fruits are in the dataset?",
    "What is the mass of the apple?",
    "Which fruit has the highest rating?",
]

for question in questions:
    print(f"{question}")
    answer = rag.query(question, top_k=3)
    print(f"DatasageAI response: {answer}\n")
    print("-" * 80 + "\n")