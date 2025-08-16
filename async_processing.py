"""Asynchronous processing capabilities for DeepseekOllamaRag application."""

import asyncio
import concurrent.futures
import threading
import queue
import time
from typing import Callable, Any, Dict, Optional, List, Tuple, Union
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum
import uuid
from functools import wraps

from logging_config import global_logger as logger
from exceptions import DeepSeekRAGException


class TaskStatus(Enum):
    """Task execution status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class AsyncTask:
    """Asynchronous task representation."""
    id: str
    name: str
    func: Callable
    args: tuple
    kwargs: dict
    status: TaskStatus = TaskStatus.PENDING
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    result: Any = None
    error: Optional[str] = None
    progress: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class TaskQueue:
    """Thread-safe task queue with priority support."""
    
    def __init__(self, max_size: int = 1000):
        self.queue = queue.PriorityQueue(maxsize=max_size)
        self.tasks: Dict[str, AsyncTask] = {}
        self.lock = threading.RLock()
        
    def add_task(self, task: AsyncTask, priority: int = 0) -> str:
        """Add task to queue with priority (lower number = higher priority)."""
        with self.lock:
            try:
                # Priority queue item: (priority, timestamp, task_id)
                self.queue.put((priority, time.time(), task.id), block=False)
                self.tasks[task.id] = task
                logger.debug(f"Added task {task.id} to queue with priority {priority}")
                return task.id
            except queue.Full:
                raise DeepSeekRAGException("Task queue is full")
    
    def get_task(self, timeout: Optional[float] = None) -> Optional[AsyncTask]:
        """Get next task from queue."""
        try:
            priority, timestamp, task_id = self.queue.get(timeout=timeout)
            with self.lock:
                if task_id in self.tasks:
                    task = self.tasks[task_id]
                    if task.status == TaskStatus.PENDING:
                        return task
                    else:
                        # Task was cancelled, get next one
                        return self.get_task(timeout=0.1)
                return None
        except queue.Empty:
            return None
    
    def get_task_status(self, task_id: str) -> Optional[AsyncTask]:
        """Get task status by ID."""
        with self.lock:
            return self.tasks.get(task_id)
    
    def cancel_task(self, task_id: str) -> bool:
        """Cancel pending task."""
        with self.lock:
            if task_id in self.tasks:
                task = self.tasks[task_id]
                if task.status == TaskStatus.PENDING:
                    task.status = TaskStatus.CANCELLED
                    task.completed_at = datetime.now()
                    return True
            return False
    
    def get_queue_stats(self) -> Dict[str, Any]:
        """Get queue statistics."""
        with self.lock:
            status_counts = {}
            for task in self.tasks.values():
                status_counts[task.status.value] = status_counts.get(task.status.value, 0) + 1
            
            return {
                "total_tasks": len(self.tasks),
                "queue_size": self.queue.qsize(),
                "status_counts": status_counts
            }


class AsyncWorker:
    """Worker thread for processing async tasks."""
    
    def __init__(self, worker_id: str, task_queue: TaskQueue, max_concurrent: int = 3):
        self.worker_id = worker_id
        self.task_queue = task_queue
        self.max_concurrent = max_concurrent
        self.running = False
        self.thread = None
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_concurrent)
        self.active_tasks: Dict[str, concurrent.futures.Future] = {}
        
    def start(self):
        """Start the worker thread."""
        if self.running:
            return
        
        self.running = True
        self.thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.thread.start()
        logger.info(f"Started async worker {self.worker_id}")
    
    def stop(self, timeout: float = 10.0):
        """Stop the worker thread."""
        if not self.running:
            return
        
        self.running = False
        
        # Cancel active tasks
        for future in self.active_tasks.values():
            future.cancel()
        
        # Shutdown executor
        self.executor.shutdown(wait=True, timeout=timeout)
        
        # Join worker thread
        if self.thread:
            self.thread.join(timeout=timeout)
        
        logger.info(f"Stopped async worker {self.worker_id}")
    
    def _worker_loop(self):
        """Main worker loop."""
        while self.running:
            try:
                # Clean up completed tasks
                self._cleanup_completed_tasks()
                
                # Check if we can accept more tasks
                if len(self.active_tasks) >= self.max_concurrent:
                    time.sleep(0.1)
                    continue
                
                # Get next task
                task = self.task_queue.get_task(timeout=1.0)
                if task is None:
                    continue
                
                # Submit task for execution
                future = self.executor.submit(self._execute_task, task)
                self.active_tasks[task.id] = future
                
            except Exception as e:
                logger.error(f"Error in worker loop {self.worker_id}: {str(e)}")
                time.sleep(1.0)
    
    def _execute_task(self, task: AsyncTask) -> Any:
        """Execute a single task."""
        try:
            task.status = TaskStatus.RUNNING
            task.started_at = datetime.now()
            
            logger.info(f"Executing task {task.id}: {task.name}")
            
            # Execute the function
            result = task.func(*task.args, **task.kwargs)
            
            task.result = result
            task.status = TaskStatus.COMPLETED
            task.progress = 1.0
            task.completed_at = datetime.now()
            
            duration = (task.completed_at - task.started_at).total_seconds()
            logger.info(f"Task {task.id} completed in {duration:.2f}s")
            
            return result
            
        except Exception as e:
            task.status = TaskStatus.FAILED
            task.error = str(e)
            task.completed_at = datetime.now()
            
            logger.error(f"Task {task.id} failed: {str(e)}")
            raise
    
    def _cleanup_completed_tasks(self):
        """Clean up completed task futures."""
        completed_tasks = []
        for task_id, future in self.active_tasks.items():
            if future.done():
                completed_tasks.append(task_id)
        
        for task_id in completed_tasks:
            del self.active_tasks[task_id]
    
    def get_worker_stats(self) -> Dict[str, Any]:
        """Get worker statistics."""
        return {
            "worker_id": self.worker_id,
            "running": self.running,
            "active_tasks": len(self.active_tasks),
            "max_concurrent": self.max_concurrent
        }


class AsyncTaskManager:
    """Main async task management system."""
    
    def __init__(self, num_workers: int = 2, max_queue_size: int = 1000):
        self.task_queue = TaskQueue(max_queue_size)
        self.workers: List[AsyncWorker] = []
        self.num_workers = num_workers
        self.running = False
        
        # Create workers
        for i in range(num_workers):
            worker = AsyncWorker(f"worker-{i}", self.task_queue)
            self.workers.append(worker)
    
    def start(self):
        """Start all workers."""
        if self.running:
            return
        
        self.running = True
        for worker in self.workers:
            worker.start()
        
        logger.info(f"Started async task manager with {self.num_workers} workers")
    
    def stop(self, timeout: float = 10.0):
        """Stop all workers."""
        if not self.running:
            return
        
        self.running = False
        for worker in self.workers:
            worker.stop(timeout)
        
        logger.info("Stopped async task manager")
    
    def submit_task(self, func: Callable, *args, name: Optional[str] = None, 
                   priority: int = 0, metadata: Optional[Dict[str, Any]] = None, **kwargs) -> str:
        """Submit task for async execution."""
        task_id = str(uuid.uuid4())
        task_name = name or func.__name__
        
        task = AsyncTask(
            id=task_id,
            name=task_name,
            func=func,
            args=args,
            kwargs=kwargs,
            metadata=metadata or {}
        )
        
        self.task_queue.add_task(task, priority)
        logger.debug(f"Submitted task {task_id}: {task_name}")
        
        return task_id
    
    def get_task_status(self, task_id: str) -> Optional[AsyncTask]:
        """Get task status."""
        return self.task_queue.get_task_status(task_id)
    
    def wait_for_task(self, task_id: str, timeout: Optional[float] = None) -> Any:
        """Wait for task completion and return result."""
        start_time = time.time()
        
        while True:
            task = self.get_task_status(task_id)
            if task is None:
                raise DeepSeekRAGException(f"Task {task_id} not found")
            
            if task.status == TaskStatus.COMPLETED:
                return task.result
            elif task.status == TaskStatus.FAILED:
                raise DeepSeekRAGException(f"Task failed: {task.error}")
            elif task.status == TaskStatus.CANCELLED:
                raise DeepSeekRAGException("Task was cancelled")
            
            if timeout and (time.time() - start_time) > timeout:
                raise DeepSeekRAGException("Task timeout")
            
            time.sleep(0.1)
    
    def cancel_task(self, task_id: str) -> bool:
        """Cancel task."""
        return self.task_queue.cancel_task(task_id)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get system statistics."""
        worker_stats = [worker.get_worker_stats() for worker in self.workers]
        queue_stats = self.task_queue.get_queue_stats()
        
        return {
            "running": self.running,
            "num_workers": self.num_workers,
            "workers": worker_stats,
            "queue": queue_stats,
            "timestamp": datetime.now().isoformat()
        }


# Progress tracking for long-running tasks
class ProgressTracker:
    """Track progress of long-running tasks."""
    
    def __init__(self, task_id: str, total_steps: int):
        self.task_id = task_id
        self.total_steps = total_steps
        self.current_step = 0
        self.step_names: List[str] = []
    
    def update(self, step_name: str = "", increment: int = 1):
        """Update progress."""
        self.current_step += increment
        if step_name:
            self.step_names.append(step_name)
        
        progress = min(1.0, self.current_step / self.total_steps)
        
        # Update task progress if task manager is available
        if hasattr(global_task_manager, 'task_queue'):
            task = global_task_manager.get_task_status(self.task_id)
            if task:
                task.progress = progress
                task.metadata.update({
                    "current_step": self.current_step,
                    "total_steps": self.total_steps,
                    "latest_step": step_name
                })
    
    def get_progress(self) -> Dict[str, Any]:
        """Get current progress."""
        return {
            "progress": min(1.0, self.current_step / self.total_steps),
            "current_step": self.current_step,
            "total_steps": self.total_steps,
            "step_names": self.step_names
        }


# Decorators for async processing
def async_task(priority: int = 0, name: Optional[str] = None):
    """Decorator to make function async-processable."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            task_name = name or func.__name__
            return global_task_manager.submit_task(
                func, *args, name=task_name, priority=priority, **kwargs
            )
        
        # Add sync version
        wrapper.sync = func
        return wrapper
    return decorator


def with_progress_tracking(total_steps: int):
    """Decorator to add progress tracking to functions."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Try to get task_id from kwargs or generate one
            task_id = kwargs.pop('task_id', str(uuid.uuid4()))
            
            # Create progress tracker
            tracker = ProgressTracker(task_id, total_steps)
            
            # Add tracker to kwargs
            kwargs['progress_tracker'] = tracker
            
            return func(*args, **kwargs)
        return wrapper
    return decorator


# Async context manager for batch processing
class AsyncBatchProcessor:
    """Process multiple items asynchronously in batches."""
    
    def __init__(self, batch_size: int = 10, max_concurrent: int = 3):
        self.batch_size = batch_size
        self.max_concurrent = max_concurrent
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_concurrent)
    
    async def process_batch(self, items: List[Any], processor_func: Callable) -> List[Any]:
        """Process items in batches asynchronously."""
        results = []
        
        # Split items into batches
        batches = [items[i:i + self.batch_size] for i in range(0, len(items), self.batch_size)]
        
        # Process batches concurrently
        loop = asyncio.get_event_loop()
        
        for batch in batches:
            # Process items in current batch concurrently
            tasks = []
            for item in batch:
                task = loop.run_in_executor(self.executor, processor_func, item)
                tasks.append(task)
            
            # Wait for batch completion
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            results.extend(batch_results)
        
        return results
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.executor.shutdown(wait=True)


# Global async task manager
global_task_manager = AsyncTaskManager(num_workers=2, max_queue_size=100)


def start_async_processing():
    """Start global async processing."""
    global_task_manager.start()
    logger.info("Async processing started")


def stop_async_processing(timeout: float = 10.0):
    """Stop global async processing."""
    global_task_manager.stop(timeout)
    logger.info("Async processing stopped")


def get_async_stats() -> Dict[str, Any]:
    """Get async processing statistics."""
    return global_task_manager.get_stats()