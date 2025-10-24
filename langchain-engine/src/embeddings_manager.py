"""
Embedding generation and management
"""
from langchain_openai import OpenAIEmbeddings
from .config import settings

class EmbeddingsManager:
    """Manage embedding generation"""
    
    def __init__(self, api_key: str = None):
        """
        Initialize embeddings manager
        
        Args:
            api_key: OpenAI API key (optional, reads from settings)
        """
        self.api_key = api_key or settings.OPENAI_API_KEY
        
        if not self.api_key:
            raise ValueError("OpenAI API key not provided")
        
        self.embeddings = OpenAIEmbeddings(
            openai_api_key=self.api_key,
            model=settings.EMBEDDING_MODEL
        )
    
    def get_embeddings(self):
        """Return embeddings instance"""
        return self.embeddings