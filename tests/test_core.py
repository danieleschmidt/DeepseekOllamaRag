"""Tests for core functionality."""

import pytest
import tempfile
import os
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

from core import DocumentProcessor, EmbeddingManager, LLMManager, RAGPipeline
from config import config
from exceptions import DocumentProcessingError, EmbeddingGenerationError, LLMError
from langchain.schema import Document


class TestDocumentProcessor:
    """Test document processing functionality."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.processor = DocumentProcessor()
    
    def test_supported_formats(self):
        """Test supported file formats."""
        assert ".pdf" in self.processor.supported_formats
    
    @pytest.fixture
    def sample_pdf_path(self):
        """Create a temporary PDF file for testing."""
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            # Create a minimal PDF content
            pdf_content = b"""%PDF-1.4
1 0 obj
<<
/Type /Catalog
/Pages 2 0 R
>>
endobj
2 0 obj
<<
/Type /Pages
/Kids [3 0 R]
/Count 1
>>
endobj
3 0 obj
<<
/Type /Page
/Parent 2 0 R
/MediaBox [0 0 612 792]
/Contents 4 0 R
>>
endobj
4 0 obj
<<
/Length 44
>>
stream
BT
/F1 12 Tf
72 720 Td
(Hello World) Tj
ET
endstream
endobj
xref
0 5
0000000000 65535 f 
0000000009 00000 n 
0000000058 00000 n 
0000000115 00000 n 
0000000205 00000 n 
trailer
<<
/Size 5
/Root 1 0 R
>>
startxref
298
%%EOF"""
            f.write(pdf_content)
            f.flush()
            yield f.name
        
        # Cleanup
        try:
            os.unlink(f.name)
        except FileNotFoundError:
            pass
    
    def test_load_document_file_not_found(self):
        """Test loading non-existent file."""
        with pytest.raises(DocumentProcessingError, match="File not found"):
            self.processor.load_document("/nonexistent/file.pdf")
    
    def test_load_document_unsupported_format(self):
        """Test loading unsupported file format."""
        with tempfile.NamedTemporaryFile(suffix=".txt") as f:
            with pytest.raises(DocumentProcessingError, match="Unsupported file format"):
                self.processor.load_document(f.name)
    
    @patch('core.PDFPlumberLoader')
    def test_load_document_success(self, mock_loader):
        """Test successful document loading."""
        # Setup mock
        mock_doc = Document(page_content="Test content", metadata={"source": "test.pdf"})
        mock_loader_instance = Mock()
        mock_loader_instance.load.return_value = [mock_doc]
        mock_loader.return_value = mock_loader_instance
        
        # Test
        result = self.processor.load_document("test.pdf")
        
        # Assertions
        assert len(result) == 1
        assert result[0].page_content == "Test content"
        mock_loader.assert_called_once_with("test.pdf")
    
    @patch('core.PDFPlumberLoader')
    def test_load_document_empty_content(self, mock_loader):
        """Test loading document with no content."""
        mock_loader_instance = Mock()
        mock_loader_instance.load.return_value = []
        mock_loader.return_value = mock_loader_instance
        
        with pytest.raises(DocumentProcessingError, match="No content extracted"):
            self.processor.load_document("test.pdf")
    
    @patch('core.HuggingFaceEmbeddings')
    @patch('core.RecursiveCharacterTextSplitter')
    def test_split_documents_small(self, mock_splitter, mock_embeddings):
        """Test splitting small documents."""
        # Setup
        mock_doc = Document(page_content="Small test content", metadata={})
        mock_splitter_instance = Mock()
        mock_splitter_instance.split_documents.return_value = [
            Document(page_content="Small test", metadata={}),
            Document(page_content="content", metadata={})
        ]
        mock_splitter.return_value = mock_splitter_instance
        
        # Test
        result = self.processor.split_documents([mock_doc])
        
        # Assertions
        assert len(result) == 2
        mock_splitter.assert_called()
    
    def test_generate_content_hash(self):
        """Test content hash generation."""
        contents = ["content1", "content2"]
        hash1 = self.processor._generate_content_hash(contents)
        hash2 = self.processor._generate_content_hash(contents)
        hash3 = self.processor._generate_content_hash(["different", "content"])
        
        assert hash1 == hash2  # Same content should produce same hash
        assert hash1 != hash3  # Different content should produce different hash
        assert len(hash1) == 64  # SHA256 hash length


class TestEmbeddingManager:
    """Test embedding management functionality."""
    
    def setup_method(self):
        """Setup test fixtures."""
        with patch('core.HuggingFaceEmbeddings'):
            self.manager = EmbeddingManager()
    
    @patch('core.HuggingFaceEmbeddings')
    def test_initialize_embeddings_success(self, mock_embeddings):
        """Test successful embedding initialization."""
        mock_embeddings_instance = Mock()
        mock_embeddings.return_value = mock_embeddings_instance
        
        manager = EmbeddingManager()
        assert manager.embeddings == mock_embeddings_instance
    
    @patch('core.HuggingFaceEmbeddings')
    def test_initialize_embeddings_failure(self, mock_embeddings):
        """Test embedding initialization failure."""
        mock_embeddings.side_effect = Exception("Model not found")
        
        with pytest.raises(EmbeddingGenerationError):
            EmbeddingManager()
    
    def test_create_vector_store_empty_documents(self):
        """Test creating vector store with empty documents."""
        with pytest.raises(VectorStoreError, match="No documents provided"):
            self.manager.create_vector_store([])
    
    @patch('core.FAISS.from_documents')
    def test_create_vector_store_success(self, mock_faiss):
        """Test successful vector store creation."""
        mock_store = Mock()
        mock_faiss.return_value = mock_store
        
        documents = [Document(page_content="test", metadata={})]
        result = self.manager.create_vector_store(documents)
        
        assert result == mock_store
        mock_faiss.assert_called_once()
    
    @patch('core.FAISS.from_documents')
    def test_create_vector_store_failure(self, mock_faiss):
        """Test vector store creation failure."""
        mock_faiss.side_effect = Exception("FAISS error")
        
        documents = [Document(page_content="test", metadata={})]
        with pytest.raises(VectorStoreError):
            self.manager.create_vector_store(documents)
    
    def test_get_retriever_success(self):
        """Test successful retriever creation."""
        mock_store = Mock()
        mock_retriever = Mock()
        mock_store.as_retriever.return_value = mock_retriever
        
        result = self.manager.get_retriever(mock_store)
        assert result == mock_retriever


class TestLLMManager:
    """Test LLM management functionality."""
    
    def setup_method(self):
        """Setup test fixtures."""
        with patch('core.check_ollama_connection', return_value=True), \
             patch('core.get_available_models', return_value=['deepseek-r1:1.5b']), \
             patch('core.Ollama') as mock_ollama:
            
            mock_llm = Mock()
            mock_llm.invoke.return_value = "Hello response"
            mock_ollama.return_value = mock_llm
            
            self.manager = LLMManager()
    
    @patch('core.check_ollama_connection', return_value=False)
    def test_initialize_llm_connection_failure(self, mock_check):
        """Test LLM initialization with connection failure."""
        with pytest.raises(LLMError):
            LLMManager()
    
    @patch('core.check_ollama_connection', return_value=True)
    @patch('core.get_available_models', return_value=['other-model'])
    def test_initialize_llm_model_not_found(self, mock_models, mock_check):
        """Test LLM initialization with unavailable model."""
        with pytest.raises(LLMError):
            LLMManager()
    
    def test_create_qa_chain_success(self):
        """Test successful QA chain creation."""
        mock_retriever = Mock()
        result = self.manager.create_qa_chain(mock_retriever)
        
        assert result is not None
        assert hasattr(result, '__call__')  # Should be callable
    
    @patch('core.RetrievalQA')
    def test_create_qa_chain_failure(self, mock_qa):
        """Test QA chain creation failure."""
        mock_qa.side_effect = Exception("Chain creation failed")
        mock_retriever = Mock()
        
        with pytest.raises(LLMError):
            self.manager.create_qa_chain(mock_retriever)
    
    def test_ask_question_success(self):
        """Test successful question processing."""
        mock_qa_chain = Mock()
        mock_qa_chain.return_value = {
            "result": "Test answer",
            "source_documents": []
        }
        
        result = self.manager.ask_question(mock_qa_chain, "Test question?", "test_hash")
        
        assert result["answer"] == "Test answer"
        assert result["question"] == "Test question?"
        assert "timestamp" in result
    
    def test_ask_question_failure(self):
        """Test question processing failure."""
        mock_qa_chain = Mock()
        mock_qa_chain.side_effect = Exception("Processing failed")
        
        with pytest.raises(LLMError):
            self.manager.ask_question(mock_qa_chain, "Test question?", "test_hash")


class TestRAGPipeline:
    """Test complete RAG pipeline."""
    
    def setup_method(self):
        """Setup test fixtures."""
        with patch('core.DocumentProcessor'), \
             patch('core.EmbeddingManager'), \
             patch('core.LLMManager'):
            self.pipeline = RAGPipeline()
    
    @patch('core.DocumentProcessor')
    @patch('core.EmbeddingManager')  
    @patch('core.LLMManager')
    def test_process_document_success(self, mock_llm, mock_embedding, mock_doc):
        """Test successful document processing pipeline."""
        # Setup mocks
        mock_doc_instance = Mock()
        mock_doc_instance.load_document.return_value = [Document(page_content="test", metadata={})]
        mock_doc_instance.split_documents.return_value = [Document(page_content="test", metadata={})]
        mock_doc.return_value = mock_doc_instance
        
        mock_embedding_instance = Mock()
        mock_vector_store = Mock()
        mock_retriever = Mock()
        mock_embedding_instance.create_vector_store.return_value = mock_vector_store
        mock_embedding_instance.get_retriever.return_value = mock_retriever
        mock_embedding.return_value = mock_embedding_instance
        
        mock_llm_instance = Mock()
        mock_qa_chain = Mock()
        mock_llm_instance.create_qa_chain.return_value = mock_qa_chain
        mock_llm.return_value = mock_llm_instance
        
        pipeline = RAGPipeline()
        
        # Test
        vector_store, qa_chain, doc_hash = pipeline.process_document("test.pdf")
        
        # Assertions
        assert vector_store == mock_vector_store
        assert qa_chain == mock_qa_chain
        assert isinstance(doc_hash, str)
        assert len(doc_hash) == 64  # SHA256 hash length
    
    def test_ask_question_delegates_to_llm_manager(self):
        """Test that ask_question delegates to LLM manager."""
        mock_qa_chain = Mock()
        mock_result = {"answer": "test", "question": "test"}
        
        self.pipeline.llm_manager.ask_question = Mock(return_value=mock_result)
        
        result = self.pipeline.ask_question(mock_qa_chain, "test question", "test_hash")
        
        assert result == mock_result
        self.pipeline.llm_manager.ask_question.assert_called_once_with(
            mock_qa_chain, "test question", "test_hash"
        )


class TestIntegration:
    """Integration tests for the complete system."""
    
    @pytest.mark.integration
    def test_full_pipeline_with_mocks(self):
        """Test full pipeline with comprehensive mocking."""
        with patch('core.check_ollama_connection', return_value=True), \
             patch('core.get_available_models', return_value=['deepseek-r1:1.5b']), \
             patch('core.PDFPlumberLoader') as mock_loader, \
             patch('core.HuggingFaceEmbeddings') as mock_embeddings, \
             patch('core.FAISS') as mock_faiss, \
             patch('core.Ollama') as mock_ollama:
            
            # Setup comprehensive mocks
            mock_doc = Document(page_content="Test document content for processing", metadata={"source": "test.pdf"})
            mock_loader_instance = Mock()
            mock_loader_instance.load.return_value = [mock_doc]
            mock_loader.return_value = mock_loader_instance
            
            mock_embeddings_instance = Mock()
            mock_embeddings.return_value = mock_embeddings_instance
            
            mock_vector_store = Mock()
            mock_retriever = Mock()
            mock_vector_store.as_retriever.return_value = mock_retriever
            mock_faiss.from_documents.return_value = mock_vector_store
            
            mock_llm = Mock()
            mock_llm.invoke.return_value = "Hello response"
            mock_ollama.return_value = mock_llm
            
            # Create pipeline and test
            pipeline = RAGPipeline()
            
            # Process document
            vector_store, qa_chain, doc_hash = pipeline.process_document("test.pdf")
            
            # Verify results
            assert vector_store is not None
            assert qa_chain is not None
            assert isinstance(doc_hash, str)
            
            # Test question asking
            mock_qa_response = {
                "result": "This is a test answer",
                "source_documents": [mock_doc]
            }
            
            with patch.object(qa_chain, '__call__', return_value=mock_qa_response):
                result = pipeline.ask_question(qa_chain, "What is this document about?", doc_hash)
                
                assert result["answer"] == "This is a test answer"
                assert result["question"] == "What is this document about?"
                assert len(result["source_documents"]) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])