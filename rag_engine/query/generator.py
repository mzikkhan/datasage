from typing import List
from langchain_core.documents import Document

class LLMGenerator:
    def __init__(self, model: str = "llama3.1:latest"):
        # Move imports here to avoid module-level import errors
        from llama_index.core import Settings
        from llama_index.llms.ollama import Ollama
        
        self.llm = Ollama(model=model)
        Settings.llm = self.llm

    def generate_answer(self, question: str, context_docs: List[Document]) -> str:
        # Build a prompt with context
        context_str = "\n\n".join(
            f"Source: {doc.metadata}\nContent: {doc.page_content}" 
            for doc in context_docs
        )
        prompt = f"Use the following context to answer the question:\n\n{context_str}\n\nQuestion: {question}\nAnswer:"
        
        response = self.llm.complete(prompt)
        return str(response)