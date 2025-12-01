import unittest
from unittest.mock import MagicMock
import sys
import os
import importlib.util

# Add project root to sys.path to allow imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Mock langchain_core and other dependencies that might cause import errors
sys.modules["langchain_core"] = MagicMock()
sys.modules["langchain_core.documents"] = MagicMock()
sys.modules["langchain_huggingface"] = MagicMock()
sys.modules["langchain_text_splitters"] = MagicMock()
sys.modules["rag_engine.indexing.index_engine"] = MagicMock()
sys.modules["rag_engine.indexing.embedder"] = MagicMock()
sys.modules["rag_engine.indexing.vector_store"] = MagicMock()

# Now we can import normally
from rag_engine.data_models import Document
from rag_engine.retrieval.retriever import Retriever

class TestRetriever(unittest.TestCase):
    def test_retrieve_conversion(self):
        # Mock VectorStore and Embedder
        mock_vs = MagicMock()
        mock_embedder = MagicMock()
        
        # Mock result from vector store (simulating an object with page_content and metadata)
        mock_doc = MagicMock()
        mock_doc.page_content = "Test content"
        mock_doc.metadata = {"id": 1}
        mock_vs.search.return_value = [mock_doc]
        
        retriever = Retriever(vector_store=mock_vs, embedder=mock_embedder)
        
        results = retriever.retrieve("query")
        
        self.assertEqual(len(results), 1)
        self.assertIsInstance(results[0], Document)
        self.assertEqual(results[0].page_content, "Test content")
        self.assertEqual(results[0].metadata, {"id": 1})
        
        # Verify vector store was called correctly
        mock_vs.search.assert_called_once_with("query", k=5, filter=None)

if __name__ == '__main__':
    unittest.main()
