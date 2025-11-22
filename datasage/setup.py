from setuptools import setup, find_packages

with open("readme.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="datasage",
    version="0.1.0",
    author="Your Team Name",
    author_email="your.email@example.com",
    description="A fully local RAG engine for document Q&A",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/datasage",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "langchain>=0.1.0",
        "langchain-huggingface>=0.0.1",
        "langchain-chroma>=0.1.0",
        "langchain-ollama>=0.1.0",
        "langchain-core>=0.1.0",
        "chromadb>=0.4.0",
        "sentence-transformers>=2.2.0",
        "pypdf>=3.0.0",
        "langchain-text-splitters>=0.0.1",
    ],
)