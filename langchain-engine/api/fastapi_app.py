"""
FastAPI application for RAG system
"""
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import sys
import os
import tempfile
from pathlib import Path

# Adjust Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

# Now import
try:
    from src.document_loader import DocumentLoader
    from src.text_processor import TextProcessor
    from src.vector_store import VectorStoreManager
    from src.rag_chain import RAGChain
    from src.config import settings
except ImportError as e:
    print(f"Import error: {e}")
    print(f"Current directory: {os.getcwd()}")
    print(f"Parent directory: {parent_dir}")
    print(f"sys.path: {sys.path}")
    raise

# Initialize FastAPI app
app = FastAPI(
    title="Renewable Energy RAG API",
    description="AI-powered knowledge retrieval for renewable energy documentation",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global instances
vector_store_manager = VectorStoreManager()
rag_chain = RAGChain()
text_processor = TextProcessor()

# Pydantic models
class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    question: str
    answer: str
    sources: List[Dict[str, Any]]

class IngestDirectoryRequest(BaseModel):
    directory_path: str

class IngestResponse(BaseModel):
    message: str
    documents_processed: int
    chunks_created: int

# Routes
@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "online",
        "service": "Renewable Energy RAG API",
        "version": "1.0.0"
    }

@app.post("/api/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """
    Query the RAG system
    
    Args:
        request: Query request with question
        
    Returns:
        Answer with sources
    """
    try:
        result = rag_chain.query(request.question)
        return QueryResponse(**result)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query error: {str(e)}")

@app.post("/api/ingest", response_model=IngestResponse)
async def ingest_file(file: UploadFile = File(...)):
    """
    Ingest a single file
    
    Args:
        file: Uploaded file (PDF, DOCX, TXT)
        
    Returns:
        Ingestion result
    """
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name
        
        # Load document
        documents = DocumentLoader.load_file(tmp_path)
        
        # Process and chunk
        chunks = text_processor.split_documents(documents)
        
        # Add to vector store
        vector_store_manager.add_documents(chunks)
        
        # Clean up
        os.unlink(tmp_path)
        
        return IngestResponse(
            message=f"Successfully ingested {file.filename}",
            documents_processed=len(documents),
            chunks_created=len(chunks)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingestion error: {str(e)}")

@app.post("/api/ingest-directory", response_model=IngestResponse)
async def ingest_directory(request: IngestDirectoryRequest):
    """
    Ingest all documents from a directory
    
    Args:
        request: Directory path
        
    Returns:
        Ingestion result
    """
    try:
        # Load all documents
        documents = DocumentLoader.load_directory(request.directory_path)
        
        if not documents:
            raise HTTPException(status_code=404, detail="No documents found in directory")
        
        # Process and chunk
        chunks = text_processor.split_documents(documents)
        
        # Create or update vector store
        vector_store_manager.create_vectorstore(chunks)
        
        return IngestResponse(
            message=f"Successfully ingested documents from {request.directory_path}",
            documents_processed=len(documents),
            chunks_created=len(chunks)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingestion error: {str(e)}")

@app.get("/api/sources")
async def get_sources():
    """Get list of indexed sources"""
    try:
        vectorstore = vector_store_manager.load_vectorstore()
        if vectorstore is None:
            return {"sources": [], "count": 0}
        
        # Get sample of documents to extract sources
        sample_docs = vectorstore.similarity_search("", k=100)
        sources = list(set([doc.metadata.get("source", "Unknown") for doc in sample_docs]))
        
        return {
            "sources": sources,
            "count": len(sources)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving sources: {str(e)}")

@app.get("/api/metrics")
async def get_metrics():
    """Get system metrics (placeholder)"""
    return {
        "total_queries": 0,
        "avg_response_time": 0,
        "documents_indexed": 0,
        "most_queried_topics": []
    }

@app.delete("/api/clear")
async def clear_vectorstore():
    """Clear the vector store"""
    try:
        vector_store_manager.clear()
        return {"message": "Vector store cleared successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error clearing vector store: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host=settings.API_HOST,
        port=settings.API_PORT
    )