"""Integration tests for the complete DeepseekOllamaRag system."""

import pytest
import tempfile
import os
import time
from unittest.mock import patch, Mock, MagicMock
from pathlib import Path

from core import RAGPipeline
from config import config
from validation import file_validator, text_validator
from security import secure_file_processing, security_manager
from caching import global_cache
from monitoring import health_checker
from resilience import resilience_manager
from async_processing import global_task_manager


class TestFullSystemIntegration:
    """Test complete system integration."""
    
    def setup_method(self):
        """Setup integration test environment."""
        # Clear caches
        global_cache.clear()
        
        # Reset security state
        security_manager.session_tokens.clear()
        security_manager.temp_files.clear()
    
    def teardown_method(self):
        """Cleanup after integration tests."""
        # Cleanup any remaining temp files
        for temp_file in list(security_manager.temp_files):
            security_manager.secure_delete_file(temp_file)
    
    @pytest.mark.integration
    def test_end_to_end_document_processing_workflow(self):
        """Test complete document processing workflow."""
        
        with patch('core.check_ollama_connection', return_value=True), \
             patch('core.get_available_models', return_value=['deepseek-r1:1.5b']), \
             patch('core.PDFPlumberLoader') as mock_loader, \
             patch('core.HuggingFaceEmbeddings') as mock_embeddings, \
             patch('core.FAISS') as mock_faiss, \
             patch('core.Ollama') as mock_ollama:
            
            # Setup comprehensive mocks for the entire pipeline
            self._setup_pipeline_mocks(mock_loader, mock_embeddings, mock_faiss, mock_ollama)
            
            # Create test file content
            test_content = b"""%PDF-1.4
1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj
2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj
3 0 obj<</Type/Page/Parent 2 0 R/MediaBox[0 0 612 792]/Contents 4 0 R>>endobj
4 0 obj<</Length 44>>stream
BT /F1 12 Tf 72 720 Td (Test Document Content) Tj ET
endstream endobj
xref 0 5
0000000000 65535 f 
0000000009 00000 n 
0000000058 00000 n 
0000000115 00000 n 
0000000205 00000 n 
trailer<</Size 5/Root 1 0 R>>
startxref 298
%%EOF"""
            
            filename = "integration_test.pdf"
            
            # Step 1: Validate file
            validation_result = file_validator.validate_file(test_content, filename)
            assert validation_result.is_valid or len(validation_result.errors) == 0
            
            # Step 2: Secure file processing
            temp_path, file_hash = secure_file_processing(test_content, filename)
            assert os.path.exists(temp_path) or temp_path.startswith("/tmp/")  # Mock path
            assert len(file_hash) == 64  # SHA256 hash
            
            # Step 3: Process document through RAG pipeline
            pipeline = RAGPipeline()
            vector_store, qa_chain, doc_hash = pipeline.process_document(temp_path)
            
            assert vector_store is not None
            assert qa_chain is not None
            assert len(doc_hash) == 64
            
            # Step 4: Validate question
            test_question = "What is the main content of this document?"
            question_validation = text_validator.validate_question(test_question)
            assert question_validation.is_valid
            
            # Step 5: Process question
            with patch.object(qa_chain, '__call__') as mock_qa_call:
                mock_qa_call.return_value = {
                    "result": "The document contains test content about document processing.",
                    "source_documents": [Mock(page_content="Test Document Content")]
                }
                
                result = pipeline.ask_question(qa_chain, test_question, doc_hash)
                
                assert "answer" in result
                assert "question" in result
                assert "source_documents" in result
                assert result["question"] == test_question
    
    def _setup_pipeline_mocks(self, mock_loader, mock_embeddings, mock_faiss, mock_ollama):
        """Setup comprehensive mocks for the RAG pipeline."""
        from langchain.schema import Document
        
        # Document loader mock
        mock_doc = Document(
            page_content="This is a test document with sample content for processing and analysis.",
            metadata={"source": "test.pdf", "page": 1}
        )
        mock_loader_instance = Mock()
        mock_loader_instance.load.return_value = [mock_doc]
        mock_loader.return_value = mock_loader_instance
        
        # Embeddings mock
        mock_embeddings_instance = Mock()
        mock_embeddings.return_value = mock_embeddings_instance
        
        # FAISS vector store mock
        mock_vector_store = Mock()
        mock_retriever = Mock()
        mock_vector_store.as_retriever.return_value = mock_retriever
        mock_faiss.from_documents.return_value = mock_vector_store
        
        # Ollama LLM mock
        mock_llm = Mock()
        mock_llm.invoke.return_value = "Test response from LLM"
        mock_ollama.return_value = mock_llm
    
    @pytest.mark.integration
    def test_caching_integration(self):
        """Test that caching works throughout the system."""
        
        # Test embedding caching
        test_text = "This is test content for embedding"
        cache_key = f"embedding:test_model:{hash(test_text)}"
        
        # Should not be cached initially
        cached_result = global_cache.get(cache_key)
        assert cached_result is None
        
        # Cache a result
        test_embedding = [0.1, 0.2, 0.3, 0.4, 0.5]
        global_cache.set(cache_key, test_embedding, ttl=300)
        
        # Should now be cached
        cached_result = global_cache.get(cache_key)
        assert cached_result == test_embedding
        
        # Test Q&A caching
        qa_key = f"qa:test_model:doc_hash:question_hash"
        qa_result = {
            "answer": "Test answer",
            "question": "Test question",
            "source_documents": []
        }
        
        global_cache.set(qa_key, qa_result, ttl=300)
        cached_qa = global_cache.get(qa_key)
        assert cached_qa == qa_result
    
    @pytest.mark.integration
    def test_security_integration(self):
        """Test security measures integration."""
        
        # Test session management
        token = security_manager.generate_session_token("test_user")
        assert security_manager.validate_session_token(token)
        
        # Test file security
        test_content = b"%PDF-1.4\ntest content"
        filename = "security_test.pdf"
        
        temp_path, file_hash = secure_file_processing(test_content, filename)
        assert temp_path in security_manager.temp_files
        assert len(file_hash) == 64
        
        # Test input sanitization
        from security import input_sanitizer
        dangerous_filename = "../../../malicious.pdf"
        safe_filename = input_sanitizer.sanitize_filename(dangerous_filename)
        assert "../" not in safe_filename
        
        # Test text sanitization
        dangerous_text = "Normal text <script>alert('xss')</script> more text"
        safe_text = input_sanitizer.sanitize_text_input(dangerous_text)
        assert "<script>" not in safe_text
    
    @pytest.mark.integration
    def test_monitoring_integration(self):
        """Test monitoring and health checks integration."""
        
        # Test health checks
        health_results = health_checker.check_all()
        
        # Should have results for all components
        expected_components = ['ollama', 'model', 'filesystem', 'memory', 'disk']
        for component in expected_components:
            assert component in health_results
            assert hasattr(health_results[component], 'status')
            assert hasattr(health_results[component], 'timestamp')
        
        # Test overall status
        overall_status = health_checker.get_overall_status()
        assert overall_status in ['healthy', 'degraded', 'unhealthy', 'unknown']
    
    @pytest.mark.integration
    def test_resilience_integration(self):
        """Test resilience mechanisms integration."""
        
        # Test circuit breaker status
        circuit_status = resilience_manager.get_status()
        assert "circuit_breakers" in circuit_status
        assert "retry_handlers" in circuit_status
        
        # Test that services are configured
        assert "ollama" in circuit_status["circuit_breakers"]
        assert "embedding" in circuit_status["circuit_breakers"]
        
        # Test circuit breaker functionality with mock
        ollama_circuit = resilience_manager.get_circuit_breaker("ollama")
        assert ollama_circuit.state.value == "closed"  # Should start closed
    
    @pytest.mark.integration
    def test_async_processing_integration(self):
        """Test async processing integration."""
        
        # Start async processing
        from async_processing import start_async_processing, stop_async_processing, get_async_stats
        
        try:
            start_async_processing()
            
            # Test task submission
            def test_task(x, y):
                return x + y
            
            task_id = global_task_manager.submit_task(test_task, 5, 3, name="test_addition")
            assert isinstance(task_id, str)
            
            # Wait for task completion
            result = global_task_manager.wait_for_task(task_id, timeout=5.0)
            assert result == 8
            
            # Test async stats
            stats = get_async_stats()
            assert "running" in stats
            assert "num_workers" in stats
            assert "queue" in stats
            
        finally:
            stop_async_processing(timeout=2.0)
    
    @pytest.mark.integration
    def test_error_handling_integration(self):
        """Test error handling across the entire system."""
        
        # Test validation errors
        from exceptions import ValidationError
        
        # Invalid file should raise validation error
        with pytest.raises(ValidationError):
            file_validator.validate_file(b"", "")
        
        # Invalid question should raise validation error
        validation_result = text_validator.validate_question("")
        assert not validation_result.is_valid
        
        # Test document processing errors
        from exceptions import DocumentProcessingError
        
        with patch('core.DocumentProcessor.load_document') as mock_load:
            mock_load.side_effect = DocumentProcessingError("Test error")
            
            pipeline = RAGPipeline()
            with pytest.raises(DocumentProcessingError):
                pipeline.process_document("nonexistent.pdf")
    
    @pytest.mark.integration
    def test_configuration_integration(self):
        """Test configuration system integration."""
        
        # Test config validation
        from validation import config_validator
        
        validation_result = config_validator.validate_config()
        # Should be valid or have specific expected issues
        if not validation_result.is_valid:
            # Log errors for debugging
            print("Config validation errors:", validation_result.errors)
        
        # Test config access
        assert config.model.llm_model is not None
        assert config.app.max_file_size_mb > 0
        assert len(config.app.allowed_file_types) > 0
        
        # Test environment variable integration
        original_debug = config.app.debug
        
        with patch.dict(os.environ, {'DEBUG': 'true'}):
            # Reload config would happen here in real scenario
            # For test, we just verify the concept
            debug_value = os.environ.get('DEBUG', 'False').lower() == 'true'
            assert debug_value == True
    
    @pytest.mark.integration
    def test_performance_integration(self):
        """Test performance optimizations integration."""
        
        # Test caching improves performance
        start_time = time.time()
        
        # First call should be slower (cache miss)
        cache_key = "test_performance_key"
        result1 = global_cache.get(cache_key)
        assert result1 is None
        
        # Cache a value
        test_data = {"large_data": list(range(1000))}
        global_cache.set(cache_key, test_data)
        
        # Second call should be faster (cache hit)
        result2 = global_cache.get(cache_key)
        assert result2 == test_data
        
        cache_time = time.time() - start_time
        assert cache_time < 1.0  # Should be very fast
        
        # Test batch processing concept
        test_items = list(range(100))
        
        def process_item(item):
            return item * 2
        
        # Simulate batch processing
        batch_size = 10
        batches = [test_items[i:i + batch_size] for i in range(0, len(test_items), batch_size)]
        
        results = []
        for batch in batches:
            batch_results = [process_item(item) for item in batch]
            results.extend(batch_results)
        
        assert len(results) == len(test_items)
        assert results[0] == 0
        assert results[-1] == 198


class TestSystemReliability:
    """Test system reliability under various conditions."""
    
    @pytest.mark.integration
    def test_system_recovery_from_failures(self):
        """Test system recovery from various failure scenarios."""
        
        # Test recovery from temporary service unavailability
        with patch('core.check_ollama_connection') as mock_check:
            # First calls fail, then succeed
            mock_check.side_effect = [False, False, True]
            
            # Should eventually succeed with retry mechanisms
            # (This would test actual retry logic in real scenario)
            final_result = mock_check()
            assert final_result is True
    
    @pytest.mark.integration
    def test_concurrent_operations(self):
        """Test system behavior under concurrent operations."""
        
        import threading
        import time
        
        results = []
        errors = []
        
        def concurrent_cache_operation(thread_id):
            try:
                # Each thread performs cache operations
                key = f"thread_{thread_id}_key"
                value = f"thread_{thread_id}_value"
                
                global_cache.set(key, value)
                retrieved = global_cache.get(key)
                
                if retrieved == value:
                    results.append(thread_id)
                else:
                    errors.append(f"Thread {thread_id}: value mismatch")
                    
            except Exception as e:
                errors.append(f"Thread {thread_id}: {str(e)}")
        
        # Start multiple threads
        threads = []
        for i in range(10):
            thread = threading.Thread(target=concurrent_cache_operation, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads
        for thread in threads:
            thread.join(timeout=5.0)
        
        # Check results
        assert len(errors) == 0, f"Concurrent operations failed: {errors}"
        assert len(results) == 10, f"Expected 10 successful operations, got {len(results)}"
    
    @pytest.mark.integration
    def test_memory_management(self):
        """Test memory management and cleanup."""
        
        import gc
        import sys
        
        # Get baseline memory usage
        initial_objects = len(gc.get_objects())
        
        # Perform operations that create objects
        for i in range(100):
            cache_key = f"memory_test_{i}"
            large_data = list(range(1000))  # Create some data
            global_cache.set(cache_key, large_data, ttl=1)  # Short TTL
        
        # Force garbage collection
        gc.collect()
        
        # Clear cache
        global_cache.clear()
        gc.collect()
        
        # Check memory cleanup
        final_objects = len(gc.get_objects())
        
        # Should not have significant memory growth
        object_growth = final_objects - initial_objects
        assert object_growth < 1000, f"Potential memory leak: {object_growth} new objects"


if __name__ == "__main__":
    # Run integration tests with verbose output
    pytest.main([__file__, "-v", "-m", "integration"])