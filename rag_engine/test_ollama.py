from langchain_ollama import OllamaLLM

# Initialize
llm = OllamaLLM(model="llama3.1")

# Test it
response = llm.invoke("What is Python? Answer in one sentence.")
print(response)