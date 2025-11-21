import os
from rag_engine import RagEngine

def main():

    # Path to your CSV file
    csv_path = "fruits_processed.csv"
    
    # Optional metadata for the document
    metadata = {
        "dataset_name": "Processed Fruits Dataset",
        "description": "Contains fruit categories, quantities, prices, and other attributes."
    }

    # Initialize the RAG Engine
    print("\n🔧 Initializing RAG Engine...")
    engine = RagEngine(csv_path, metadata=metadata)

    # Define your analysis question
    query = "Provide a detailed summary of the key insights from this dataset."

    print("\n🤖 Asking the model for insights...")
    response = engine.query(query)

    print("\n📊 Summary Insights:\n")
    print(response)


if __name__ == "__main__":
    main()
