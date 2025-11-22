import os
from datasage import RagEngine

def main():

    # Global Path to your CSV file
    csv_path = "/Users/zaedkhan/Desktop/data533_project_experiments/datasage/fruits_processed.csv"
    
    # Optional metadata for the document
    metadata = {
        "dataset_name": "Processed Fruits Dataset",
        "description": "Contains fruit categories, quantities, prices, and other attributes."
    }

    # Initialize the RAG Engine
    print("\nInitializing RAG Engine...")
    engine = RagEngine(csv_path, metadata=metadata, model_name="tinyllama:1.1b")

    # Define your analysis question
    query = "Provide a detailed summary of the key insights from this dataset."

    print("\nAsking the model for insights...")
    response = engine.query(query)

    print("\nSummary Insights:\n")
    print(response)


if __name__ == "__main__":
    main()