"""Health checks and monitoring for DeepseekOllamaRag application."""

import time
import psutil
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from threading import Thread, Event
import json
from pathlib import Path

from config import config
from logging_config import global_logger as logger, log_system_health
from exceptions import OllamaConnectionError, ModelNotFoundError


@dataclass
class HealthStatus:
    """Health status information."""
    component: str
    status: str  # "healthy", "unhealthy", "degraded", "unknown"
    timestamp: datetime
    response_time_ms: Optional[float] = None
    details: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None


@dataclass
class SystemMetrics:
    """System performance metrics."""
    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    memory_used_mb: float
    memory_available_mb: float
    disk_usage_percent: float
    disk_free_gb: float
    active_connections: int = 0
    response_time_ms: Optional[float] = None


class HealthChecker:
    """Comprehensive health checking system."""
    
    def __init__(self):
        self.checks = {
            'ollama': self._check_ollama,
            'model': self._check_model,
            'filesystem': self._check_filesystem,
            'memory': self._check_memory,
            'disk': self._check_disk,
        }
        self.status_history = []
        self.max_history = 100
    
    def check_all(self) -> Dict[str, HealthStatus]:
        """Run all health checks."""
        results = {}
        
        for check_name, check_func in self.checks.items():
            try:
                start_time = time.time()
                status = check_func()
                response_time = (time.time() - start_time) * 1000
                
                if status.response_time_ms is None:
                    status.response_time_ms = response_time
                
                results[check_name] = status
                
            except Exception as e:
                results[check_name] = HealthStatus(
                    component=check_name,
                    status="unhealthy",
                    timestamp=datetime.now(),
                    error_message=str(e)
                )
                logger.error(f"Health check failed for {check_name}: {str(e)}")
        
        # Store in history
        self._update_history(results)
        
        return results
    
    def _check_ollama(self) -> HealthStatus:
        """Check Ollama service health."""
        try:
            response = requests.get(
                f"{config.model.ollama_base_url}/api/tags",
                timeout=5
            )
            
            if response.status_code == 200:
                models = response.json().get("models", [])
                return HealthStatus(
                    component="ollama",
                    status="healthy",
                    timestamp=datetime.now(),
                    details={
                        "available_models": len(models),
                        "url": config.model.ollama_base_url
                    }
                )
            else:
                return HealthStatus(
                    component="ollama",
                    status="unhealthy",
                    timestamp=datetime.now(),
                    error_message=f"HTTP {response.status_code}"
                )
                
        except requests.exceptions.ConnectionError:
            return HealthStatus(
                component="ollama",
                status="unhealthy",
                timestamp=datetime.now(),
                error_message="Connection refused"
            )
        except requests.exceptions.Timeout:
            return HealthStatus(
                component="ollama",
                status="degraded",
                timestamp=datetime.now(),
                error_message="Request timeout"
            )
        except Exception as e:
            return HealthStatus(
                component="ollama",
                status="unhealthy",
                timestamp=datetime.now(),
                error_message=str(e)
            )
    
    def _check_model(self) -> HealthStatus:
        """Check if configured model is available."""
        try:
            response = requests.get(
                f"{config.model.ollama_base_url}/api/tags",
                timeout=5
            )
            
            if response.status_code == 200:
                models = response.json().get("models", [])
                model_names = [model["name"] for model in models]
                
                if config.model.llm_model in model_names:
                    return HealthStatus(
                        component="model",
                        status="healthy",
                        timestamp=datetime.now(),
                        details={
                            "model_name": config.model.llm_model,
                            "total_models": len(models)
                        }
                    )
                else:
                    return HealthStatus(
                        component="model",
                        status="unhealthy",
                        timestamp=datetime.now(),
                        error_message=f"Model '{config.model.llm_model}' not found",
                        details={"available_models": model_names}
                    )
            else:
                return HealthStatus(
                    component="model",
                    status="unknown",
                    timestamp=datetime.now(),
                    error_message=f"Cannot check models: HTTP {response.status_code}"
                )
                
        except Exception as e:
            return HealthStatus(
                component="model",
                status="unknown",
                timestamp=datetime.now(),
                error_message=str(e)
            )
    
    def _check_filesystem(self) -> HealthStatus:
        """Check filesystem health."""
        try:
            # Check required directories
            required_dirs = [config.app.upload_dir, config.app.temp_dir, "logs"]
            missing_dirs = []
            
            for dir_path in required_dirs:
                if not Path(dir_path).exists():
                    missing_dirs.append(dir_path)
            
            if missing_dirs:
                return HealthStatus(
                    component="filesystem",
                    status="degraded",
                    timestamp=datetime.now(),
                    error_message=f"Missing directories: {missing_dirs}"
                )
            
            # Check write permissions
            test_file = Path(config.app.temp_dir) / "health_check.tmp"
            try:
                test_file.write_text("health check")
                test_file.unlink()
                
                return HealthStatus(
                    component="filesystem",
                    status="healthy",
                    timestamp=datetime.now(),
                    details={"directories_checked": len(required_dirs)}
                )
            except Exception as e:
                return HealthStatus(
                    component="filesystem",
                    status="unhealthy",
                    timestamp=datetime.now(),
                    error_message=f"Write permission error: {str(e)}"
                )
                
        except Exception as e:
            return HealthStatus(
                component="filesystem",
                status="unhealthy",
                timestamp=datetime.now(),
                error_message=str(e)
            )
    
    def _check_memory(self) -> HealthStatus:
        """Check memory usage."""
        try:
            memory = psutil.virtual_memory()
            
            if memory.percent > 90:
                status = "unhealthy"
            elif memory.percent > 80:
                status = "degraded"
            else:
                status = "healthy"
            
            return HealthStatus(
                component="memory",
                status=status,
                timestamp=datetime.now(),
                details={
                    "usage_percent": memory.percent,
                    "used_mb": memory.used / (1024 * 1024),
                    "available_mb": memory.available / (1024 * 1024),
                    "total_mb": memory.total / (1024 * 1024)
                }
            )
            
        except Exception as e:
            return HealthStatus(
                component="memory",
                status="unknown",
                timestamp=datetime.now(),
                error_message=str(e)
            )
    
    def _check_disk(self) -> HealthStatus:
        """Check disk usage."""
        try:
            disk = psutil.disk_usage('/')
            usage_percent = (disk.used / disk.total) * 100
            
            if usage_percent > 95:
                status = "unhealthy"
            elif usage_percent > 85:
                status = "degraded"
            else:
                status = "healthy"
            
            return HealthStatus(
                component="disk",
                status=status,
                timestamp=datetime.now(),
                details={
                    "usage_percent": usage_percent,
                    "free_gb": disk.free / (1024 ** 3),
                    "total_gb": disk.total / (1024 ** 3)
                }
            )
            
        except Exception as e:
            return HealthStatus(
                component="disk",
                status="unknown",
                timestamp=datetime.now(),
                error_message=str(e)
            )
    
    def _update_history(self, results: Dict[str, HealthStatus]):
        """Update health check history."""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "results": {name: asdict(status) for name, status in results.items()}
        }
        
        self.status_history.append(entry)
        
        # Keep only recent history
        if len(self.status_history) > self.max_history:
            self.status_history = self.status_history[-self.max_history:]
    
    def get_overall_status(self) -> str:
        """Get overall system health status."""
        results = self.check_all()
        
        statuses = [status.status for status in results.values()]
        
        if "unhealthy" in statuses:
            return "unhealthy"
        elif "degraded" in statuses:
            return "degraded"
        elif "unknown" in statuses:
            return "unknown"
        elif all(status == "healthy" for status in statuses):
            return "healthy"
        else:
            return "unknown"


class PerformanceMonitor:
    """Performance monitoring and metrics collection."""
    
    def __init__(self):
        self.metrics_history = []
        self.max_history = 1000
        self.monitoring_active = False
        self.monitor_thread = None
        self.stop_event = Event()
    
    def start_monitoring(self, interval_seconds: int = 30):
        """Start continuous performance monitoring."""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.stop_event.clear()
        
        self.monitor_thread = Thread(
            target=self._monitor_loop,
            args=(interval_seconds,),
            daemon=True
        )
        self.monitor_thread.start()
        logger.info(f"Performance monitoring started (interval: {interval_seconds}s)")
    
    def stop_monitoring(self):
        """Stop continuous monitoring."""
        if not self.monitoring_active:
            return
        
        self.monitoring_active = False
        self.stop_event.set()
        
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        
        logger.info("Performance monitoring stopped")
    
    def _monitor_loop(self, interval_seconds: int):
        """Main monitoring loop."""
        while not self.stop_event.wait(interval_seconds):
            try:
                metrics = self.collect_metrics()
                self._update_metrics_history(metrics)
                
                # Log system health periodically
                if len(self.metrics_history) % 10 == 0:  # Every 10 intervals
                    log_system_health({
                        "cpu_percent": metrics.cpu_percent,
                        "memory_percent": metrics.memory_percent,
                        "disk_usage_percent": metrics.disk_usage_percent,
                        "monitoring_active": self.monitoring_active
                    })\n                    \n            except Exception as e:\n                logger.error(f\"Error in monitoring loop: {str(e)}\")\n    \n    def collect_metrics(self) -> SystemMetrics:\n        \"\"\"Collect current system metrics.\"\"\"\n        try:\n            # CPU usage\n            cpu_percent = psutil.cpu_percent(interval=1)\n            \n            # Memory usage\n            memory = psutil.virtual_memory()\n            \n            # Disk usage\n            disk = psutil.disk_usage('/')\n            disk_usage_percent = (disk.used / disk.total) * 100\n            \n            return SystemMetrics(\n                timestamp=datetime.now(),\n                cpu_percent=cpu_percent,\n                memory_percent=memory.percent,\n                memory_used_mb=memory.used / (1024 * 1024),\n                memory_available_mb=memory.available / (1024 * 1024),\n                disk_usage_percent=disk_usage_percent,\n                disk_free_gb=disk.free / (1024 ** 3)\n            )\n            \n        except Exception as e:\n            logger.error(f\"Error collecting metrics: {str(e)}\")\n            return SystemMetrics(\n                timestamp=datetime.now(),\n                cpu_percent=0.0,\n                memory_percent=0.0,\n                memory_used_mb=0.0,\n                memory_available_mb=0.0,\n                disk_usage_percent=0.0,\n                disk_free_gb=0.0\n            )\n    \n    def _update_metrics_history(self, metrics: SystemMetrics):\n        \"\"\"Update metrics history.\"\"\"\n        self.metrics_history.append(asdict(metrics))\n        \n        # Keep only recent history\n        if len(self.metrics_history) > self.max_history:\n            self.metrics_history = self.metrics_history[-self.max_history:]\n    \n    def get_metrics_summary(self, hours: int = 1) -> Dict[str, Any]:\n        \"\"\"Get metrics summary for the last N hours.\"\"\"\n        if not self.metrics_history:\n            return {}\n        \n        cutoff_time = datetime.now() - timedelta(hours=hours)\n        recent_metrics = [\n            m for m in self.metrics_history\n            if datetime.fromisoformat(m['timestamp']) > cutoff_time\n        ]\n        \n        if not recent_metrics:\n            return {}\n        \n        # Calculate averages\n        cpu_values = [m['cpu_percent'] for m in recent_metrics]\n        memory_values = [m['memory_percent'] for m in recent_metrics]\n        disk_values = [m['disk_usage_percent'] for m in recent_metrics]\n        \n        return {\n            \"period_hours\": hours,\n            \"samples_count\": len(recent_metrics),\n            \"cpu_percent\": {\n                \"avg\": sum(cpu_values) / len(cpu_values),\n                \"min\": min(cpu_values),\n                \"max\": max(cpu_values)\n            },\n            \"memory_percent\": {\n                \"avg\": sum(memory_values) / len(memory_values),\n                \"min\": min(memory_values),\n                \"max\": max(memory_values)\n            },\n            \"disk_usage_percent\": {\n                \"avg\": sum(disk_values) / len(disk_values),\n                \"min\": min(disk_values),\n                \"max\": max(disk_values)\n            }\n        }\n    \n    def export_metrics(self, file_path: str):\n        \"\"\"Export metrics to JSON file.\"\"\"\n        try:\n            with open(file_path, 'w') as f:\n                json.dump({\n                    \"exported_at\": datetime.now().isoformat(),\n                    \"metrics_count\": len(self.metrics_history),\n                    \"metrics\": self.metrics_history\n                }, f, indent=2)\n            logger.info(f\"Metrics exported to {file_path}\")\n        except Exception as e:\n            logger.error(f\"Error exporting metrics: {str(e)}\")\n\n\n# Global instances\nhealth_checker = HealthChecker()\nperformance_monitor = PerformanceMonitor()\n\n\ndef get_system_status() -> Dict[str, Any]:\n    \"\"\"Get comprehensive system status.\"\"\"\n    health_results = health_checker.check_all()\n    current_metrics = performance_monitor.collect_metrics()\n    metrics_summary = performance_monitor.get_metrics_summary()\n    \n    return {\n        \"overall_status\": health_checker.get_overall_status(),\n        \"health_checks\": {name: asdict(status) for name, status in health_results.items()},\n        \"current_metrics\": asdict(current_metrics),\n        \"metrics_summary\": metrics_summary,\n        \"monitoring_active\": performance_monitor.monitoring_active,\n        \"timestamp\": datetime.now().isoformat()\n    }\n\n\ndef start_background_monitoring():\n    \"\"\"Start background monitoring services.\"\"\"\n    try:\n        performance_monitor.start_monitoring(interval_seconds=60)\n        logger.info(\"Background monitoring services started\")\n    except Exception as e:\n        logger.error(f\"Failed to start background monitoring: {str(e)}\")\n\n\ndef stop_background_monitoring():\n    \"\"\"Stop background monitoring services.\"\"\"\n    try:\n        performance_monitor.stop_monitoring()\n        logger.info(\"Background monitoring services stopped\")\n    except Exception as e:\n        logger.error(f\"Failed to stop background monitoring: {str(e)}\")"}, {"old_string": "                    log_system_health({\n                        \"cpu_percent\": metrics.cpu_percent,\n                        \"memory_percent\": metrics.memory_percent,\n                        \"disk_usage_percent\": metrics.disk_usage_percent,\n                        \"monitoring_active\": self.monitoring_active\n                    })", "new_string": "                    log_system_health({\n                        \"cpu_percent\": metrics.cpu_percent,\n                        \"memory_percent\": metrics.memory_percent,\n                        \"disk_usage_percent\": metrics.disk_usage_percent,\n                        \"monitoring_active\": self.monitoring_active\n                    })"}]