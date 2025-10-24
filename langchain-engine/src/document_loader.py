"""
Document loading utilities for renewable energy documents
"""
from typing import List
from pathlib import Path
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    Docx2txtLoader
)
from langchain.schema import Document

class DocumentLoader:
    """Load documents from various formats"""
    
    SUPPORTED_EXTENSIONS = {'.pdf', '.txt', '.docx'}
    
    @staticmethod
    def load_document(file_path: str) -> List[Document]:
        """
        Load a single document based on file extension
        
        Args:
            file_path: Path to the document
            
        Returns:
            List of Document objects
        """
        path = Path(file_path)
        
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        extension = path.suffix.lower()
        
        if extension not in DocumentLoader.SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported file type: {extension}. "
                f"Supported: {DocumentLoader.SUPPORTED_EXTENSIONS}"
            )
        
        # Load based on extension
        if extension == '.pdf':
            loader = PyPDFLoader(file_path)
        elif extension == '.txt':
            loader = TextLoader(file_path)
        elif extension == '.docx':
            loader = Docx2txtLoader(file_path)
        
        documents = loader.load()
        
        # Add metadata
        for doc in documents:
            doc.metadata.update({
                'source': path.name,
                'file_type': extension[1:],
                'file_path': str(path.absolute())
            })
        
        return documents
    
    @staticmethod
    def load_directory(directory_path: str) -> List[Document]:
        """
        Load all supported documents from a directory
        
        Args:
            directory_path: Path to directory
            
        Returns:
            List of all loaded documents
        """
        path = Path(directory_path)
        
        if not path.is_dir():
            raise ValueError(f"Not a directory: {directory_path}")
        
        all_documents = []
        
        for file_path in path.rglob('*'):
            if file_path.suffix.lower() in DocumentLoader.SUPPORTED_EXTENSIONS:
                try:
                    docs = DocumentLoader.load_document(str(file_path))
                    all_documents.extend(docs)
                    print(f"✓ Loaded: {file_path.name} ({len(docs)} chunks)")
                except Exception as e:
                    print(f"✗ Error loading {file_path.name}: {e}")
        
        return all_documents