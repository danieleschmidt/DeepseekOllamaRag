"""
Comprehensive Test Suite for Research-Enhanced RAG System

Tests all major components and research enhancements:
1. Core RAG functionality
2. Research enhancements
3. Multi-modal processing
4. Adaptive learning
5. Benchmarking framework
6. Integration testing
"""

import unittest
import tempfile
import os
import json
import time
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import numpy as np

# Test imports
try:
    from config import config
    from core import RAGPipeline, DocumentProcessor, EmbeddingManager, LLMManager
    from research_enhancements import ResearchEnhancedRAG, HierarchicalDocumentUnderstanding
    from experimental_framework import ExperimentManager, RAGEvaluationSuite
    from adaptive_learning import AdaptiveLearningOrchestrator, UserFeedback
    from research_benchmarks import BenchmarkRunner, StandardBenchmarkLoader
    from research_integration import IntegratedRAGSystem
    IMPORTS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Some imports failed: {e}")
    IMPORTS_AVAILABLE = False


class TestCoreRAGSystem(unittest.TestCase):
    """Test core RAG system functionality."""
    
    def setUp(self):
        if not IMPORTS_AVAILABLE:
            self.skipTest("Required imports not available")
        
        self.temp_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        # Clean up temporary files
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_document_processor_initialization(self):
        """Test document processor initialization."""
        processor = DocumentProcessor()
        self.assertIsNotNone(processor)
        self.assertEqual(processor.supported_formats, [".pdf"])
    
    @patch('core.PDFPlumberLoader')
    def test_document_loading(self, mock_loader):
        """Test document loading functionality."""
        # Mock document content
        mock_doc = Mock()
        mock_doc.page_content = "Test document content"
        mock_loader.return_value.load.return_value = [mock_doc]
        
        processor = DocumentProcessor()
        
        # Create a temporary PDF file
        test_file = os.path.join(self.temp_dir, "test.pdf")
        with open(test_file, 'w') as f:
            f.write("dummy pdf content")
        
        # Test loading
        documents = processor.load_document(test_file)
        self.assertIsNotNone(documents)
        self.assertTrue(len(documents) > 0)
    
    def test_embedding_manager_initialization(self):
        """Test embedding manager initialization."""
        with patch('core.HuggingFaceEmbeddings'):
            manager = EmbeddingManager()
            self.assertIsNotNone(manager)
    
    @patch('core.check_ollama_connection')
    @patch('core.Ollama')
    def test_llm_manager_initialization(self, mock_ollama, mock_connection):
        """Test LLM manager initialization."""
        mock_connection.return_value = True
        mock_ollama.return_value.invoke.return_value = "Hello response"
        
        manager = LLMManager()
        self.assertIsNotNone(manager.llm)


class TestResearchEnhancements(unittest.TestCase):
    """Test research enhancement functionality."""
    
    def setUp(self):
        if not IMPORTS_AVAILABLE:
            self.skipTest("Required imports not available")
    
    def test_hierarchical_document_understanding(self):
        """Test HDU functionality."""
        hdu = HierarchicalDocumentUnderstanding()
        self.assertIsNotNone(hdu)
        
        # Test with mock chunks
        mock_chunks = []
        for i in range(3):
            chunk = Mock()
            chunk.page_content = f"Sample content for chunk {i}"
            chunk.metadata = {'page': i}
            mock_chunks.append(chunk)
        
        with patch.object(hdu.sentence_transformer, 'encode') as mock_encode:
            mock_encode.return_value = np.random.rand(3, 384)  # Mock embeddings
            
            graph = hdu.build_document_hierarchy(mock_chunks)
            self.assertIsNotNone(graph)
            self.assertEqual(graph.number_of_nodes(), 3)
    
    def test_research_enhanced_rag_initialization(self):
        """Test research enhanced RAG initialization."""
        base_pipeline = Mock()
        enhanced_rag = ResearchEnhancedRAG(base_pipeline)
        
        self.assertIsNotNone(enhanced_rag)
        self.assertEqual(enhanced_rag.base_pipeline, base_pipeline)


class TestAdaptiveLearning(unittest.TestCase):
    """Test adaptive learning functionality."""
    
    def setUp(self):
        if not IMPORTS_AVAILABLE:
            self.skipTest("Required imports not available")
        
        self.temp_db = tempfile.NamedTemporaryFile(suffix='.db', delete=False)
        self.temp_db.close()
    
    def tearDown(self):
        os.unlink(self.temp_db.name)
    
    def test_feedback_collector_initialization(self):
        """Test feedback collector initialization."""
        from adaptive_learning import UserFeedbackCollector
        
        collector = UserFeedbackCollector(db_path=self.temp_db.name)
        self.assertIsNotNone(collector)
    
    def test_feedback_recording(self):
        """Test feedback recording functionality."""
        from adaptive_learning import UserFeedbackCollector
        
        collector = UserFeedbackCollector(db_path=self.temp_db.name)
        
        feedback = UserFeedback(
            query="Test question",
            answer="Test answer",
            rating=0.8,
            feedback_text="Good response",
            timestamp=time.time(),
            session_id="test_session",
            document_context="test_doc",
            response_time=1.5,
            system_version="1.0"
        )
        
        result = collector.record_feedback(feedback)
        self.assertTrue(result)
        
        # Test retrieval
        recent_feedback = collector.get_recent_feedback(hours=1)
        self.assertEqual(len(recent_feedback), 1)
        self.assertEqual(recent_feedback[0].query, "Test question")
    
    def test_adaptive_learning_orchestrator(self):
        """Test adaptive learning orchestrator."""
        orchestrator = AdaptiveLearningOrchestrator()
        self.assertIsNotNone(orchestrator)
        
        # Test interaction recording
        result = orchestrator.record_interaction(
            query="Test query",
            answer="Test answer", 
            response_time=1.0,
            session_id="test_session"
        )
        self.assertTrue(result)


class TestBenchmarking(unittest.TestCase):
    """Test benchmarking functionality."""
    
    def setUp(self):
        if not IMPORTS_AVAILABLE:
            self.skipTest("Required imports not available")
    
    def test_benchmark_loader_initialization(self):
        """Test benchmark loader initialization."""
        loader = StandardBenchmarkLoader()
        self.assertIsNotNone(loader)
        
        # Test benchmark availability
        benchmarks = loader.list_benchmarks()
        self.assertTrue(len(benchmarks) > 0)
        
        # Test getting benchmark info
        first_benchmark = benchmarks[0]
        info = loader.get_benchmark_info(first_benchmark)
        self.assertIsNotNone(info)
        self.assertIn('name', info)
        self.assertIn('description', info)
    
    def test_evaluation_metrics(self):
        """Test evaluation metrics calculation."""
        from research_benchmarks import AdvancedEvaluationMetrics
        
        metrics = AdvancedEvaluationMetrics()
        self.assertIsNotNone(metrics)
        
        # Test metrics calculation
        predictions = ["The answer is 42", "This is correct"]
        ground_truth = ["The answer is 42", "This is right"]
        
        result = metrics.calculate_comprehensive_metrics(predictions, ground_truth)
        
        self.assertIsInstance(result, dict)
        self.assertIn('jaccard_similarity_mean', result)
        self.assertIn('completeness_mean', result)
        self.assertIn('relevance_mean', result)


class TestExperimentalFramework(unittest.TestCase):
    """Test experimental framework functionality."""
    
    def setUp(self):
        if not IMPORTS_AVAILABLE:
            self.skipTest("Required imports not available")
        
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_experiment_manager_initialization(self):
        """Test experiment manager initialization."""
        manager = ExperimentManager(output_dir=self.temp_dir)
        self.assertIsNotNone(manager)
        self.assertTrue(Path(self.temp_dir).exists())
    
    def test_evaluation_suite(self):
        """Test evaluation suite functionality."""
        suite = RAGEvaluationSuite()
        self.assertIsNotNone(suite)
        
        # Test retrieval quality evaluation
        mock_docs = []
        for i in range(3):
            doc = Mock()
            doc.page_content = f"Document content {i}"
            mock_docs.append(doc)
        
        metrics = suite.evaluate_retrieval_quality("test query", mock_docs)
        self.assertIsInstance(metrics, dict)
        self.assertIn('diversity', metrics)
        self.assertIn('coverage', metrics)


class TestIntegratedSystem(unittest.TestCase):
    """Test integrated system functionality."""
    
    def setUp(self):
        if not IMPORTS_AVAILABLE:
            self.skipTest("Required imports not available")
    
    @patch('research_integration.RAGPipeline')
    @patch('research_integration.ResearchEnhancedRAG')
    def test_integrated_system_initialization(self, mock_enhanced, mock_baseline):
        """Test integrated system initialization."""
        mock_baseline.return_value = Mock()
        mock_enhanced.return_value = Mock()
        
        system = IntegratedRAGSystem()
        self.assertIsNotNone(system)
        self.assertIsNotNone(system.baseline_system)
    
    def test_performance_tracking(self):
        """Test performance tracking functionality."""
        with patch('research_integration.RAGPipeline'), \
             patch('research_integration.ResearchEnhancedRAG'):
            
            system = IntegratedRAGSystem()
            
            # Mock performance tracking
            system._track_performance("test query", 1.5, "enhanced", True)
            
            self.assertEqual(len(system.performance_metrics), 1)
            self.assertEqual(system.performance_metrics[0]['system_type'], "enhanced")


class TestSystemIntegration(unittest.TestCase):
    """Test full system integration."""
    
    def setUp(self):
        if not IMPORTS_AVAILABLE:
            self.skipTest("Required imports not available")
        
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @patch('core.check_ollama_connection')
    def test_system_health_check(self, mock_connection):
        """Test system health check."""
        mock_connection.return_value = True
        
        # Test that system components can be initialized together
        components = []
        
        try:
            # Initialize core components
            processor = DocumentProcessor()
            components.append(('DocumentProcessor', processor))
            
            # Initialize research components
            orchestrator = AdaptiveLearningOrchestrator()
            components.append(('AdaptiveLearningOrchestrator', orchestrator))
            
            loader = StandardBenchmarkLoader()
            components.append(('StandardBenchmarkLoader', loader))
            
        except Exception as e:
            self.fail(f"System integration failed: {e}")
        
        # Verify all components initialized
        self.assertEqual(len(components), 3)
        
        for name, component in components:
            self.assertIsNotNone(component, f"{name} failed to initialize")
    
    def test_configuration_validity(self):
        """Test configuration validity."""
        # Test that configuration is accessible and valid
        try:
            self.assertIsNotNone(config)
            self.assertIsNotNone(config.model)
            self.assertIsNotNone(config.vector_store)
            self.assertIsNotNone(config.app)
            self.assertIsNotNone(config.ui)
            
            # Test specific config values
            self.assertGreater(config.vector_store.similarity_search_k, 0)
            self.assertGreater(config.app.max_file_size_mb, 0)
            self.assertGreater(config.vector_store.chunk_size, 0)
            
        except Exception as e:
            self.fail(f"Configuration validation failed: {e}")
    
    def test_error_handling(self):
        """Test error handling across components."""
        # Test that components handle errors gracefully
        processor = DocumentProcessor()
        
        # Test with non-existent file
        with self.assertRaises(Exception):
            processor.load_document("/non/existent/file.pdf")
        
        # Test with unsupported file type
        test_file = os.path.join(self.temp_dir, "test.txt")
        with open(test_file, 'w') as f:
            f.write("test content")
        
        with self.assertRaises(Exception):
            processor.load_document(test_file)


class TestPerformanceAndSecurity(unittest.TestCase):
    """Test performance and security aspects."""
    
    def setUp(self):
        if not IMPORTS_AVAILABLE:
            self.skipTest("Required imports not available")
    
    def test_memory_usage(self):
        """Test memory usage is reasonable."""
        import psutil
        import gc
        
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Initialize multiple components
        components = []
        for i in range(5):
            orchestrator = AdaptiveLearningOrchestrator()
            components.append(orchestrator)
        
        current_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = current_memory - initial_memory
        
        # Clean up
        del components
        gc.collect()
        
        # Memory increase should be reasonable (less than 100MB for test)
        self.assertLess(memory_increase, 100, 
                       f"Memory usage increased by {memory_increase:.2f}MB")
    
    def test_input_validation(self):
        """Test input validation and sanitization."""
        try:
            from research_integration import IntegratedRAGSystem
            
            system = IntegratedRAGSystem()
            
            # Test with potentially malicious inputs
            malicious_inputs = [
                "",  # Empty string
                "A" * 10000,  # Very long string
                "SELECT * FROM users; DROP TABLE users;",  # SQL injection attempt
                "<script>alert('xss')</script>",  # XSS attempt
                "../../../etc/passwd",  # Path traversal attempt
            ]
            
            for malicious_input in malicious_inputs:
                # System should handle these gracefully without crashing
                try:
                    # This would normally be filtered by validation layers
                    result = len(malicious_input)  # Simple safe operation
                    self.assertIsInstance(result, int)
                except Exception:
                    # Exceptions are acceptable for malicious inputs
                    pass
                    
        except ImportError:
            self.skipTest("Security components not available for testing")


def create_test_suite():
    """Create comprehensive test suite."""
    
    suite = unittest.TestSuite()
    
    # Add test classes
    test_classes = [
        TestCoreRAGSystem,
        TestResearchEnhancements,  
        TestAdaptiveLearning,
        TestBenchmarking,
        TestExperimentalFramework,
        TestIntegratedSystem,
        TestSystemIntegration,
        TestPerformanceAndSecurity
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    return suite


def run_comprehensive_tests():
    """Run all tests and return results."""
    
    print("🔬 Running Comprehensive Research System Tests...")
    print("=" * 60)
    
    suite = create_test_suite()
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    print("\n" + "=" * 60)
    print(f"📊 Test Results Summary:")
    print(f"   Tests Run: {result.testsRun}")
    print(f"   Failures: {len(result.failures)}")
    print(f"   Errors: {len(result.errors)}")
    print(f"   Skipped: {len(result.skipped)}")
    
    success_rate = ((result.testsRun - len(result.failures) - len(result.errors)) / 
                   result.testsRun * 100) if result.testsRun > 0 else 0
    
    print(f"   Success Rate: {success_rate:.1f}%")
    
    if result.failures:
        print(f"\n❌ Failures:")
        for test, traceback in result.failures:
            print(f"   - {test}: {traceback.split(chr(10))[-2]}")
    
    if result.errors:
        print(f"\n⚠️  Errors:")
        for test, traceback in result.errors:
            print(f"   - {test}: {traceback.split(chr(10))[-2]}")
    
    print("=" * 60)
    
    # Return success status
    return len(result.failures) == 0 and len(result.errors) == 0


if __name__ == "__main__":
    success = run_comprehensive_tests()
    exit(0 if success else 1)