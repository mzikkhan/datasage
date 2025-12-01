from typing import List
from langchain_core.documents import Document
import csv

class DocumentLoader:
    def load(self, file_path: str) -> List[Document]:
        """Load and return a list of Document objects from a file."""
        raise NotImplementedError


class PDFLoader(DocumentLoader):
    def load(self, file_path: str) -> List[Document]:
        try:
            from pypdf import PdfReader
            
            reader = PdfReader(file_path)
            documents = []
            
            for page_num, page in enumerate(reader.pages):
                text = page.extract_text()
                if text.strip():
                    doc = Document(
                        page_content=text,
                        metadata={
                            "source": file_path,
                            "page": page_num + 1
                        }
                    )
                    documents.append(doc)
            
            return documents
        except ImportError:
            raise ImportError("pypdf is required for PDF loading. Install it with: pip install pypdf")


class CSVLoader(DocumentLoader):
    def load(self, file_path: str) -> List[Document]:
        documents = []
        
        with open(file_path, 'r', encoding='utf-8') as f:
            csv_reader = csv.DictReader(f)
            
            for row_num, row in enumerate(csv_reader):
                # Convert row to a readable text format
                content = "\n".join(f"{key}: {value}" for key, value in row.items())
                
                doc = Document(
                    page_content=content,
                    metadata={
                        "source": file_path,
                        "row": row_num + 1,
                        **row  # Include all CSV fields in metadata
                    }
                )
                documents.append(doc)
        
        return documents


class TextLoader(DocumentLoader):
    def load(self, file_path: str) -> List[Document]:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        return [Document(
            page_content=content,
            metadata={"source": file_path}
        )]