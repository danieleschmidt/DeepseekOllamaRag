"""Performance benchmarks and tests for DeepseekOllamaRag application."""

import pytest
import time
import threading
import concurrent.futures
from unittest.mock import patch, Mock
import statistics
from typing import List, Dict, Any

from core import DocumentProcessor, EmbeddingManager, RAGPipeline
from caching import global_cache, SmartCache
from async_processing import global_task_manager
from monitoring import performance_monitor
from langchain.schema import Document


class PerformanceBenchmark:
    """Base class for performance benchmarks."""
    
    def __init__(self, name: str):
        self.name = name
        self.results: List[float] = []
        self.metadata: Dict[str, Any] = {}
    
    def run_benchmark(self, iterations: int = 10) -> Dict[str, Any]:
        """Run benchmark multiple times and collect statistics."""
        self.results = []
        
        for i in range(iterations):
            start_time = time.time()
            self._run_single_iteration(i)
            end_time = time.time()
            self.results.append(end_time - start_time)
        
        return self._calculate_statistics()
    
    def _run_single_iteration(self, iteration: int):
        """Override this method in subclasses."""
        raise NotImplementedError
    
    def _calculate_statistics(self) -> Dict[str, Any]:
        """Calculate performance statistics."""
        if not self.results:
            return {}
        
        return {
            "benchmark_name": self.name,
            "iterations": len(self.results),
            "mean_time": statistics.mean(self.results),
            "median_time": statistics.median(self.results),
            "min_time": min(self.results),
            "max_time": max(self.results),
            "std_dev": statistics.stdev(self.results) if len(self.results) > 1 else 0,
            "total_time": sum(self.results),
            "throughput_per_second": len(self.results) / sum(self.results),
            "metadata": self.metadata
        }


class DocumentProcessingBenchmark(PerformanceBenchmark):
    """Benchmark document processing performance."""
    
    def __init__(self):
        super().__init__("Document Processing")
        self.test_documents = self._create_test_documents()
    
    def _create_test_documents(self) -> List[Document]:
        """Create test documents of various sizes."""
        documents = []
        
        # Small document
        small_content = "This is a small test document. " * 50
        documents.append(Document(page_content=small_content, metadata={"size": "small"}))
        
        # Medium document
        medium_content = "This is a medium test document with more content. " * 500
        documents.append(Document(page_content=medium_content, metadata={"size": "medium"}))
        
        # Large document
        large_content = "This is a large test document with extensive content. " * 2000
        documents.append(Document(page_content=large_content, metadata={"size": "large"}))
        
        return documents
    
    @patch('core.HuggingFaceEmbeddings')
    def _run_single_iteration(self, iteration: int):
        """Run document processing for one iteration."""
        processor = DocumentProcessor()
        
        # Use a different document each time (cycle through)
        doc = self.test_documents[iteration % len(self.test_documents)]
        
        # Mock the embedding model to avoid actual model loading
        with patch('core.RecursiveCharacterTextSplitter') as mock_splitter:
            mock_chunks = [
                Document(page_content=doc.page_content[:100], metadata={}),
                Document(page_content=doc.page_content[100:200], metadata={}),
                Document(page_content=doc.page_content[200:300], metadata={})
            ]
            mock_splitter_instance = Mock()
            mock_splitter_instance.split_documents.return_value = mock_chunks
            mock_splitter.return_value = mock_splitter_instance
            
            chunks = processor.split_documents([doc])
            self.metadata[f"iteration_{iteration}_chunks"] = len(chunks)


class CachingBenchmark(PerformanceBenchmark):
    """Benchmark caching performance."""
    
    def __init__(self):
        super().__init__("Caching Performance")
        self.cache = SmartCache(memory_max_size=1000, disk_max_size_mb=100)
        self.test_data = self._create_test_data()
    
    def _create_test_data(self) -> List[Dict[str, Any]]:
        """Create test data of various sizes."""
        data = []
        
        # Small data
        small_data = {"type": "small", "data": list(range(100))}
        data.append(small_data)
        
        # Medium data
        medium_data = {"type": "medium", "data": list(range(10000))}
        data.append(medium_data)
        
        # Large data
        large_data = {"type": "large", "data": list(range(100000))}
        data.append(large_data)
        
        return data
    
    def _run_single_iteration(self, iteration: int):
        """Run caching operations for one iteration."""
        data = self.test_data[iteration % len(self.test_data)]
        key = f"benchmark_key_{iteration}"
        
        # Test cache set
        start_set = time.time()
        self.cache.set(key, data)
        set_time = time.time() - start_set
        
        # Test cache get
        start_get = time.time()
        retrieved = self.cache.get(key)
        get_time = time.time() - start_get
        
        self.metadata[f"iteration_{iteration}_set_time"] = set_time
        self.metadata[f"iteration_{iteration}_get_time"] = get_time
        self.metadata[f"iteration_{iteration}_data_size"] = len(str(data))
        
        assert retrieved == data, "Cache data integrity check failed"


class ConcurrentAccessBenchmark(PerformanceBenchmark):
    """Benchmark concurrent access performance."""
    
    def __init__(self, num_threads: int = 10):
        super().__init__(f"Concurrent Access ({num_threads} threads)")
        self.num_threads = num_threads
        self.cache = SmartCache(memory_max_size=1000)
    
    def _run_single_iteration(self, iteration: int):
        """Run concurrent operations for one iteration."""
        results = []
        errors = []
        
        def worker_task(worker_id: int):
            try:
                # Each worker performs multiple cache operations
                for i in range(10):
                    key = f"worker_{worker_id}_key_{i}"
                    value = f"worker_{worker_id}_value_{i}"
                    
                    # Set and get operations
                    self.cache.set(key, value)
                    retrieved = self.cache.get(key)
                    
                    if retrieved != value:
                        errors.append(f"Worker {worker_id}: Data mismatch")
                    else:
                        results.append(worker_id)
                        
            except Exception as e:
                errors.append(f"Worker {worker_id}: {str(e)}")
        
        # Run concurrent workers
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_threads) as executor:
            futures = [executor.submit(worker_task, i) for i in range(self.num_threads)]
            concurrent.futures.wait(futures, timeout=10.0)
        
        self.metadata[f"iteration_{iteration}_successes"] = len(results)
        self.metadata[f"iteration_{iteration}_errors"] = len(errors)
        
        if errors:
            raise Exception(f"Concurrent access errors: {errors[:5]}")  # Show first 5 errors


class MemoryUsageBenchmark(PerformanceBenchmark):
    """Benchmark memory usage patterns."""
    
    def __init__(self):
        super().__init__("Memory Usage")
        self.cache = SmartCache(memory_max_size=100, disk_max_size_mb=50)
    
    def _run_single_iteration(self, iteration: int):
        """Run memory usage test for one iteration."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Create and cache large amounts of data
        large_data_sets = []
        for i in range(50):
            data = {
                "id": i,
                "content": list(range(1000)),
                "metadata": {"created": time.time(), "size": "large"}
            }
            large_data_sets.append(data)
            self.cache.set(f"memory_test_{iteration}_{i}", data)
        
        # Check memory after operations
        peak_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Clear cache to test cleanup
        self.cache.clear()
        
        # Check memory after cleanup
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        self.metadata[f"iteration_{iteration}_initial_memory_mb"] = initial_memory
        self.metadata[f"iteration_{iteration}_peak_memory_mb"] = peak_memory
        self.metadata[f"iteration_{iteration}_final_memory_mb"] = final_memory
        self.metadata[f"iteration_{iteration}_memory_growth_mb"] = peak_memory - initial_memory
        self.metadata[f"iteration_{iteration}_memory_cleanup_mb"] = peak_memory - final_memory


class AsyncProcessingBenchmark(PerformanceBenchmark):
    """Benchmark async processing performance."""
    
    def __init__(self):
        super().__init__("Async Processing")
    
    def _run_single_iteration(self, iteration: int):
        """Run async processing test for one iteration."""
        from async_processing import start_async_processing, stop_async_processing
        
        # Simple task function
        def compute_task(n):
            return sum(range(n))
        
        try:
            start_async_processing()
            
            # Submit multiple tasks
            task_ids = []
            num_tasks = 20
            
            for i in range(num_tasks):
                task_id = global_task_manager.submit_task(
                    compute_task, 
                    1000 + i * 100,  # Different workloads
                    name=f"benchmark_task_{i}"
                )
                task_ids.append(task_id)
            
            # Wait for all tasks to complete
            results = []
            for task_id in task_ids:
                try:
                    result = global_task_manager.wait_for_task(task_id, timeout=5.0)
                    results.append(result)
                except Exception as e:
                    self.metadata[f"iteration_{iteration}_task_error"] = str(e)
            
            self.metadata[f"iteration_{iteration}_tasks_submitted"] = num_tasks
            self.metadata[f"iteration_{iteration}_tasks_completed"] = len(results)
            
        finally:
            stop_async_processing(timeout=2.0)


class TestPerformanceBenchmarks:
    """Test class for running performance benchmarks."""
    
    @pytest.mark.performance
    def test_document_processing_performance(self):
        """Test document processing performance."""
        benchmark = DocumentProcessingBenchmark()
        results = benchmark.run_benchmark(iterations=5)
        
        # Performance assertions
        assert results["mean_time"] < 5.0, f"Document processing too slow: {results['mean_time']:.2f}s"
        assert results["std_dev"] < 2.0, f"Document processing too inconsistent: {results['std_dev']:.2f}s"
        
        print(f"Document Processing Performance: {results['mean_time']:.3f}±{results['std_dev']:.3f}s")
    
    @pytest.mark.performance
    def test_caching_performance(self):
        """Test caching performance."""
        benchmark = CachingBenchmark()
        results = benchmark.run_benchmark(iterations=10)
        
        # Performance assertions
        assert results["mean_time"] < 1.0, f"Caching too slow: {results['mean_time']:.2f}s"
        assert results["throughput_per_second"] > 5.0, f"Caching throughput too low: {results['throughput_per_second']:.1f} ops/s"
        
        print(f"Caching Performance: {results['throughput_per_second']:.1f} ops/s")
    
    @pytest.mark.performance
    def test_concurrent_access_performance(self):
        """Test concurrent access performance."""
        benchmark = ConcurrentAccessBenchmark(num_threads=5)
        results = benchmark.run_benchmark(iterations=3)
        
        # Performance assertions
        assert results["mean_time"] < 3.0, f"Concurrent access too slow: {results['mean_time']:.2f}s"
        
        # Check for errors in metadata
        total_errors = sum(
            results["metadata"].get(f"iteration_{i}_errors", 0) 
            for i in range(3)
        )
        assert total_errors == 0, f"Concurrent access errors: {total_errors}"
        
        print(f"Concurrent Access Performance: {results['mean_time']:.3f}s average")
    
    @pytest.mark.performance
    def test_memory_usage_performance(self):
        """Test memory usage patterns."""
        benchmark = MemoryUsageBenchmark()
        results = benchmark.run_benchmark(iterations=3)
        
        # Memory usage assertions
        max_growth = max(
            results["metadata"].get(f"iteration_{i}_memory_growth_mb", 0)
            for i in range(3)
        )
        assert max_growth < 500, f"Memory growth too high: {max_growth:.1f}MB"
        
        # Check memory cleanup
        avg_cleanup = statistics.mean([
            results["metadata"].get(f"iteration_{i}_memory_cleanup_mb", 0)
            for i in range(3)
        ])
        assert avg_cleanup > 0, "Memory not being cleaned up properly"
        
        print(f"Memory Usage: Peak growth {max_growth:.1f}MB, Average cleanup {avg_cleanup:.1f}MB")
    
    @pytest.mark.performance
    @pytest.mark.slow
    def test_async_processing_performance(self):
        """Test async processing performance."""
        benchmark = AsyncProcessingBenchmark()
        results = benchmark.run_benchmark(iterations=3)
        
        # Performance assertions
        assert results["mean_time"] < 10.0, f"Async processing too slow: {results['mean_time']:.2f}s"
        
        # Check task completion rate
        total_submitted = sum(
            results["metadata"].get(f"iteration_{i}_tasks_submitted", 0)
            for i in range(3)
        )
        total_completed = sum(
            results["metadata"].get(f"iteration_{i}_tasks_completed", 0)
            for i in range(3)
        )
        
        completion_rate = total_completed / total_submitted if total_submitted > 0 else 0
        assert completion_rate > 0.9, f"Task completion rate too low: {completion_rate:.2%}"
        
        print(f"Async Processing: {completion_rate:.1%} completion rate, {results['mean_time']:.3f}s average")
    
    @pytest.mark.performance
    def test_system_performance_monitoring(self):
        """Test system performance monitoring."""
        from monitoring import performance_monitor
        
        # Start monitoring
        performance_monitor.start_monitoring(interval_seconds=1)
        
        try:
            # Perform some operations
            time.sleep(2)  # Let it collect some metrics
            
            # Get metrics
            metrics = performance_monitor.collect_metrics()
            summary = performance_monitor.get_metrics_summary(hours=1)
            
            # Assertions
            assert metrics.cpu_percent >= 0.0
            assert metrics.memory_percent >= 0.0
            assert metrics.disk_usage_percent >= 0.0
            
            if summary:  # May be empty if no history
                assert "samples_count" in summary
                print(f"System Monitoring: {summary}")
            
        finally:
            performance_monitor.stop_monitoring()


class PerformanceReport:
    """Generate comprehensive performance report."""
    
    @staticmethod
    def run_all_benchmarks() -> Dict[str, Any]:
        """Run all performance benchmarks and generate report."""
        benchmarks = [
            DocumentProcessingBenchmark(),
            CachingBenchmark(),
            ConcurrentAccessBenchmark(num_threads=5),
            MemoryUsageBenchmark(),
            AsyncProcessingBenchmark()
        ]
        
        report = {
            "timestamp": time.time(),
            "benchmarks": {},
            "summary": {}
        }
        
        total_time = 0
        
        for benchmark in benchmarks:
            print(f"Running benchmark: {benchmark.name}")
            start_time = time.time()
            
            try:
                results = benchmark.run_benchmark(iterations=5)
                report["benchmarks"][benchmark.name] = results
                total_time += time.time() - start_time
                
            except Exception as e:
                report["benchmarks"][benchmark.name] = {
                    "error": str(e),
                    "benchmark_name": benchmark.name
                }
        
        report["summary"] = {
            "total_benchmarks": len(benchmarks),
            "successful_benchmarks": len([b for b in report["benchmarks"].values() if "error" not in b]),
            "total_execution_time": total_time
        }
        
        return report


if __name__ == "__main__":
    # Run performance tests
    pytest.main([__file__, "-v", "-m", "performance"])
    
    # Generate performance report
    print("\\n" + "="*50)
    print("PERFORMANCE REPORT")
    print("="*50)
    
    report = PerformanceReport.run_all_benchmarks()
    
    for name, results in report["benchmarks"].items():
        if "error" in results:
            print(f"{name}: ERROR - {results['error']}")
        else:
            print(f"{name}: {results['mean_time']:.3f}±{results['std_dev']:.3f}s "
                  f"({results['throughput_per_second']:.1f} ops/s)")
    
    print(f"\\nTotal execution time: {report['summary']['total_execution_time']:.2f}s")
    print(f"Successful benchmarks: {report['summary']['successful_benchmarks']}/{report['summary']['total_benchmarks']}")