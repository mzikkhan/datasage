from typing import List
from langchain_core.documents import Document
from langchain_ollama import OllamaLLM

class LLMGenerator:
    def __init__(self, model: str = "llama3.1"):
        """Initialize with Ollama through LangChain."""
        self.llm = OllamaLLM(model=model)

    def generate_answer(self, question: str, context_docs: List[Document]) -> str:
        """Generate an answer using the LLM with retrieved context."""
        context_str = "\n\n".join(
            f"Content: {doc.page_content}" 
            for doc in context_docs
        )
        prompt = f"""Use the following context to answer the question. Be concise and specific.

Context:
{context_str}

Question: {question}

Answer:"""
        
        response = self.llm.invoke(prompt)
        return response