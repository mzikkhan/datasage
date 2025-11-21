from typing import List
from langchain_core.documents import Document
from llama_index import LLMPredictor
from llama_index.llms import Ollama

class LLMGenerator:
    def __init__(self, model: str = "llama3.1:latest"):
        from llama_index import LLMPredictor
        from llama_index.llms import Ollama
        self.llm_predictor = LLMPredictor(llm=Ollama(model=model))

    def generate_answer(self, question: str, context_docs: List[Document]) -> str:
        # Build a prompt with context
        context_str = "\n\n".join(f"Source: {doc.metadata}\nContent: {doc.page_content}" for doc in context_docs)
        prompt = f"Use the following context to answer the question:\n\n{context_str}\n\nQuestion: {question}\nAnswer:"
        return self.llm_predictor.predict(prompt)
