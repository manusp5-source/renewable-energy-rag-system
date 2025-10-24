"""
Configuration management for RAG system
"""
from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    """Application settings"""
    
    # OpenAI Configuration
    OPENAI_API_KEY: str = ""
    EMBEDDING_MODEL: str = "text-embedding-ada-002"
    LLM_MODEL: str = "gpt-4"
    LLM_TEMPERATURE: float = 0.0
    
    # Vector Store Configuration
    VECTOR_STORE_TYPE: str = "chroma"  # or "pinecone"
    CHROMA_PERSIST_DIR: str = "./chroma_db"
    
    # Retrieval Configuration
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200
    TOP_K_RESULTS: int = 4
    
    # API Configuration
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    
    class Config:
        env_file = ".env"
        case_sensitive = True

settings = Settings()