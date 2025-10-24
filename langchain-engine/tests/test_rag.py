"""
Tests for RAG system components
"""
import pytest
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.document_loader import DocumentLoader
from src.text_processor import TextProcessor
from src.embeddings_manager import EmbeddingsManager
from src.vector_store import VectorStoreManager
from src.config import settings

# Sample test data
SAMPLE_TEXT = """
Solar photovoltaic (PV) systems convert sunlight directly into electricity.
The optimal tilt angle for solar panels depends on latitude and season.
In Spain, a tilt angle of 30-35 degrees typically maximizes annual energy production.
Wind turbines convert kinetic energy from wind into electrical power.
"""

class TestDocumentLoader:
    """Test document loading functionality"""
    
    def test_supported_extensions(self):
        """Test that supported extensions are correct"""
        expected = {'.pdf', '.txt', '.docx'}
        assert DocumentLoader.SUPPORTED_EXTENSIONS == expected
    
    def test_unsupported_file_raises_error(self):
        """Test that unsupported file types raise error"""
        with pytest.raises(ValueError):
            DocumentLoader.load_document("test.xlsx")

class TestTextProcessor:
    """Test text processing functionality"""
    
    def test_text_processor_initialization(self):
        """Test text processor initializes correctly"""
        processor = TextProcessor(chunk_size=500, chunk_overlap=50)
        assert processor.text_splitter.chunk_size == 500
        assert processor.text_splitter.chunk_overlap == 50
    
    def test_clean_text(self):
        """Test text cleaning"""
        dirty_text = "This  has   extra    spaces"
        clean = TextProcessor.clean_text(dirty_text)
        assert clean == "This has extra spaces"

class TestRAGSystem:
    """Integration tests for RAG system"""
    
    @pytest.fixture
    def sample_documents(self):
        """Create sample documents for testing"""
        from langchain.schema import Document
        return [
            Document(
                page_content=SAMPLE_TEXT,
                metadata={"source": "test.txt", "page": 1}
            )
        ]
    
    def test_text_processing(self, sample_documents):
        """Test document chunking"""
        processor = TextProcessor(chunk_size=100, chunk_overlap=20)
        chunks = processor.process_documents(sample_documents)
        
        assert len(chunks) > 0
        assert all(hasattr(chunk, 'page_content') for chunk in chunks)
        assert all('chunk_id' in chunk.metadata for chunk in chunks)

# Evaluation metrics for RAG
class RAGEvaluator:
    """Evaluate RAG system performance"""
    
    @staticmethod
    def calculate_relevance_score(
        query: str,
        retrieved_docs: list,
        ground_truth: str
    ) -> float:
        """
        Calculate relevance score for retrieved documents
        
        Args:
            query: User query
            retrieved_docs: Documents retrieved by RAG
            ground_truth: Expected answer keywords
            
        Returns:
            Relevance score (0-1)
        """
        # Simple keyword matching (can be improved with embeddings)
        keywords = set(ground_truth.lower().split())
        retrieved_text = " ".join([doc.page_content.lower() for doc in retrieved_docs])
        
        matches = sum(1 for keyword in keywords if keyword in retrieved_text)
        score = matches / len(keywords) if keywords else 0.0
        
        return score
    
    @staticmethod
    def evaluate_test_queries(rag_chain, test_cases: list) -> dict:
        """
        Evaluate RAG system on test queries
        
        Args:
            rag_chain: RAG chain instance
            test_cases: List of (query, expected_keywords) tuples
            
        Returns:
            Evaluation metrics
        """
        scores = []
        
        for query, expected_keywords in test_cases:
            result = rag_chain.query(query)
            answer = result["answer"].lower()
            
            # Calculate keyword overlap
            keywords = set(expected_keywords.lower().split())
            matches = sum(1 for keyword in keywords if keyword in answer)
            score = matches / len(keywords) if keywords else 0.0
            scores.append(score)
        
        return {
            "average_score": sum(scores) / len(scores) if scores else 0.0,
            "total_tests": len(test_cases),
            "scores": scores
        }

# Sample test queries for renewable energy
TEST_QUERIES = [
    ("What is the optimal tilt angle for solar panels in Spain?", "30 35 degrees spain"),
    ("How do solar panels work?", "photovoltaic sunlight electricity convert"),
    ("What is wind energy?", "wind turbine kinetic energy electrical"),
]

if __name__ == "__main__":
    print("Running RAG system tests...")
    pytest.main([__file__, "-v"])