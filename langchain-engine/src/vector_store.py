"""
Vector store management
"""
from typing import List, Optional
from pathlib import Path
from langchain_community.vectorstores import Chroma
from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings
from .config import settings

class VectorStoreManager:
    """Manage vector store operations"""
    
    def __init__(self, persist_directory: str = None):
        """
        Initialize vector store manager
        
        Args:
            persist_directory: Directory to persist vector store
        """
        if persist_directory is None:
            persist_directory = settings.CHROMA_PERSIST_DIR
            
        self.persist_directory = persist_directory
        
        # Initialize embeddings directly
        self.embeddings = OpenAIEmbeddings(
            openai_api_key=settings.OPENAI_API_KEY,
            model=settings.EMBEDDING_MODEL
        )
        
        self.vectorstore: Optional[Chroma] = None
    
    def create_vectorstore(self, documents: List[Document]) -> Chroma:
        """
        Create vector store from documents
        
        Args:
            documents: List of documents to index
            
        Returns:
            Chroma vector store
        """
        self.vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings,
            persist_directory=self.persist_directory
        )
        return self.vectorstore
    
    def load_vectorstore(self) -> Optional[Chroma]:
        """
        Load existing vector store
        
        Returns:
            Chroma vector store or None if doesn't exist
        """
        if not Path(self.persist_directory).exists():
            return None
        
        try:
            self.vectorstore = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embeddings
            )
            return self.vectorstore
        except Exception as e:
            print(f"Error loading vector store: {e}")
            return None
    
    def add_documents(self, documents: List[Document]):
        """
        Add documents to existing vector store
        
        Args:
            documents: Documents to add
        """
        if self.vectorstore is None:
            self.vectorstore = self.load_vectorstore()
        
        if self.vectorstore is None:
            self.vectorstore = self.create_vectorstore(documents)
        else:
            self.vectorstore.add_documents(documents)
    
    def similarity_search(
        self,
        query: str,
        k: int = None
    ) -> List[Document]:
        """
        Search for similar documents
        
        Args:
            query: Search query
            k: Number of results
            
        Returns:
            List of relevant documents
        """
        if k is None:
            k = settings.TOP_K_RESULTS
            
        if self.vectorstore is None:
            self.vectorstore = self.load_vectorstore()
        
        if self.vectorstore is None:
            return []
        
        return self.vectorstore.similarity_search(query, k=k)
    
    def clear(self):
        """Clear vector store"""
        if Path(self.persist_directory).exists():
            import shutil
            shutil.rmtree(self.persist_directory)
        self.vectorstore = None