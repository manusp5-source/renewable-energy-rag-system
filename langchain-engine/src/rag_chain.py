"""
RAG chain implementation
"""
from typing import Dict, Any, List
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.schema import Document
from .vector_store import VectorStoreManager
from .config import settings

class RAGChain:
    """RAG chain for question answering"""
    
    def __init__(self):
        """Initialize RAG chain"""
        self.vector_store_manager = VectorStoreManager()
        
        self.llm = ChatOpenAI(
            openai_api_key=settings.OPENAI_API_KEY,
            model_name=settings.LLM_MODEL,
            temperature=settings.LLM_TEMPERATURE
        )
        
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer"
        )
        
        self.chain = None
    
    def initialize_chain(self):
        """Initialize the conversational retrieval chain"""
        vectorstore = self.vector_store_manager.load_vectorstore()
        
        if vectorstore is None:
            raise ValueError("Vector store not initialized. Please ingest documents first.")
        
        self.chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=vectorstore.as_retriever(
                search_kwargs={"k": settings.TOP_K_RESULTS}
            ),
            memory=self.memory,
            return_source_documents=True,
            verbose=False
        )
    
    def query(self, question: str) -> Dict[str, Any]:
        """
        Query the RAG system
        
        Args:
            question: User question
            
        Returns:
            Dictionary with answer and sources
        """
        if self.chain is None:
            self.initialize_chain()
        
        result = self.chain({"question": question})
        
        # Format response
        return {
            "question": question,
            "answer": result["answer"],
            "sources": self._format_sources(result.get("source_documents", []))
        }
    
    def _format_sources(self, documents: List[Document]) -> List[Dict[str, Any]]:
        """
        Format source documents
        
        Args:
            documents: Source documents
            
        Returns:
            List of formatted sources
        """
        sources = []
        for doc in documents:
            sources.append({
                "content": doc.page_content,
                "source": doc.metadata.get("source", "Unknown"),
                "page": doc.metadata.get("page", "N/A")
            })
        return sources