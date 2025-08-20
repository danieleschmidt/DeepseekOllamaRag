"""
System Monitoring and Health Checks

Provides comprehensive monitoring and health checking capabilities
for the RAG system components.
"""

import logging
import time
import json
import threading
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict
from pathlib import Path

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    logging.warning("psutil not available, system monitoring will be limited")

from config import config
from utils import setup_logging

logger = setup_logging()


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


@dataclass
class HealthStatus:
    """Health status for a component."""
    component: str
    status: str  # 'healthy', 'warning', 'critical'
    message: str
    last_check: datetime
    response_time_ms: float = 0.0


class HealthChecker:
    """Performs health checks on system components."""
    
    def __init__(self):
        self.checks = {}
        
    def register_check(self, name: str, check_function: callable, 
                      interval_seconds: int = 60):
        """Register a health check function."""
        self.checks[name] = {
            'function': check_function,
            'interval': interval_seconds,
            'last_run': None,
            'last_result': None
        }
    
    def check_component(self, name: str) -> HealthStatus:
        """Check health of a specific component."""
        if name not in self.checks:
            return HealthStatus(
                component=name,
                status="critical",
                message="Health check not registered",
                last_check=datetime.now()
            )
        
        check_info = self.checks[name]
        
        # Check if we need to run the check
        now = datetime.now()
        if (check_info['last_run'] is None or 
            (now - check_info['last_run']).total_seconds() >= check_info['interval']):
            
            start_time = time.time()
            try:
                result = check_info['function']()
                response_time = (time.time() - start_time) * 1000
                
                check_info['last_run'] = now
                check_info['last_result'] = HealthStatus(
                    component=name,
                    status=result.get('status', 'healthy'),
                    message=result.get('message', 'OK'),
                    last_check=now,
                    response_time_ms=response_time
                )
                
            except Exception as e:
                response_time = (time.time() - start_time) * 1000
                check_info['last_result'] = HealthStatus(
                    component=name,
                    status="critical",
                    message=f"Health check failed: {str(e)}",
                    last_check=now,
                    response_time_ms=response_time
                )
        
        return check_info['last_result']
    
    def check_all(self) -> Dict[str, HealthStatus]:
        """Run all registered health checks."""
        results = {}
        for name in self.checks:
            results[name] = self.check_component(name)
        return results
    
    def get_overall_status(self) -> str:
        """Get overall system health status."""
        results = self.check_all()
        
        if not results:
            return "unknown"
        
        statuses = [result.status for result in results.values()]
        
        if any(status == "critical" for status in statuses):
            return "critical"
        elif any(status == "warning" for status in statuses):
            return "warning"
        else:
            return "healthy"


class PerformanceMonitor:
    """Monitors system performance metrics."""
    
    def __init__(self, max_history: int = 1000):
        self.max_history = max_history
        self.metrics_history = []
        self.monitoring_active = False
        self.monitor_thread = None
    
    def start_monitoring(self, interval_seconds: int = 60):
        """Start background monitoring."""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.monitor_thread = threading.Thread(
            target=self._monitoring_loop,
            args=(interval_seconds,),
            daemon=True
        )
        self.monitor_thread.start()
        logger.info(f"Performance monitoring started (interval: {interval_seconds}s)")
    
    def stop_monitoring(self):
        """Stop background monitoring."""
        self.monitoring_active = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        logger.info("Performance monitoring stopped")
    
    def _monitoring_loop(self, interval_seconds: int):
        """Background monitoring loop."""
        while self.monitoring_active:
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
                    })
                    
            except Exception as e:
                logger.error(f"Error in monitoring loop: {str(e)}")
            
            time.sleep(interval_seconds)
    
    def collect_metrics(self) -> SystemMetrics:
        """Collect current system metrics."""
        try:
            if not PSUTIL_AVAILABLE:
                return SystemMetrics(
                    timestamp=datetime.now(),
                    cpu_percent=0.0,
                    memory_percent=0.0,
                    memory_used_mb=0.0,
                    memory_available_mb=0.0,
                    disk_usage_percent=0.0,
                    disk_free_gb=0.0
                )
            
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # Memory usage
            memory = psutil.virtual_memory()
            
            # Disk usage
            disk = psutil.disk_usage('/')
            disk_usage_percent = (disk.used / disk.total) * 100
            
            return SystemMetrics(
                timestamp=datetime.now(),
                cpu_percent=cpu_percent,
                memory_percent=memory.percent,
                memory_used_mb=memory.used / (1024 * 1024),
                memory_available_mb=memory.available / (1024 * 1024),
                disk_usage_percent=disk_usage_percent,
                disk_free_gb=disk.free / (1024 ** 3)
            )
            
        except Exception as e:
            logger.error(f"Error collecting metrics: {str(e)}")
            return SystemMetrics(
                timestamp=datetime.now(),
                cpu_percent=0.0,
                memory_percent=0.0,
                memory_used_mb=0.0,
                memory_available_mb=0.0,
                disk_usage_percent=0.0,
                disk_free_gb=0.0
            )
    
    def _update_metrics_history(self, metrics: SystemMetrics):
        """Update metrics history."""
        self.metrics_history.append(asdict(metrics))
        
        # Keep only recent history
        if len(self.metrics_history) > self.max_history:
            self.metrics_history = self.metrics_history[-self.max_history:]
    
    def get_metrics_summary(self, hours: int = 1) -> Dict[str, Any]:
        """Get metrics summary for the last N hours."""
        if not self.metrics_history:
            return {}
        
        cutoff_time = datetime.now() - timedelta(hours=hours)
        recent_metrics = [
            m for m in self.metrics_history
            if datetime.fromisoformat(m['timestamp']) > cutoff_time
        ]
        
        if not recent_metrics:
            return {}
        
        # Calculate averages
        cpu_values = [m['cpu_percent'] for m in recent_metrics]
        memory_values = [m['memory_percent'] for m in recent_metrics]
        disk_values = [m['disk_usage_percent'] for m in recent_metrics]
        
        return {
            "period_hours": hours,
            "samples_count": len(recent_metrics),
            "cpu_percent": {
                "avg": sum(cpu_values) / len(cpu_values),
                "min": min(cpu_values),
                "max": max(cpu_values)
            },
            "memory_percent": {
                "avg": sum(memory_values) / len(memory_values),
                "min": min(memory_values),
                "max": max(memory_values)
            },
            "disk_usage_percent": {
                "avg": sum(disk_values) / len(disk_values),
                "min": min(disk_values),
                "max": max(disk_values)
            }
        }
    
    def export_metrics(self, file_path: str):
        """Export metrics to JSON file."""
        try:
            with open(file_path, 'w') as f:
                json.dump({
                    "exported_at": datetime.now().isoformat(),
                    "metrics_count": len(self.metrics_history),
                    "metrics": self.metrics_history
                }, f, indent=2)
            logger.info(f"Metrics exported to {file_path}")
        except Exception as e:
            logger.error(f"Error exporting metrics: {str(e)}")


def log_system_health(metrics: Dict[str, Any]):
    """Log system health metrics."""
    logger.info(f"System Health - CPU: {metrics.get('cpu_percent', 0):.1f}%, "
               f"Memory: {metrics.get('memory_percent', 0):.1f}%, "
               f"Disk: {metrics.get('disk_usage_percent', 0):.1f}%")


# Global instances
health_checker = HealthChecker()
performance_monitor = PerformanceMonitor()


def get_system_status() -> Dict[str, Any]:
    """Get comprehensive system status."""
    health_results = health_checker.check_all()
    current_metrics = performance_monitor.collect_metrics()
    metrics_summary = performance_monitor.get_metrics_summary()
    
    return {
        "overall_status": health_checker.get_overall_status(),
        "health_checks": {name: asdict(status) for name, status in health_results.items()},
        "current_metrics": asdict(current_metrics),
        "metrics_summary": metrics_summary,
        "monitoring_active": performance_monitor.monitoring_active,
        "timestamp": datetime.now().isoformat()
    }


def start_background_monitoring():
    """Start background monitoring services."""
    try:
        performance_monitor.start_monitoring(interval_seconds=60)
        logger.info("Background monitoring services started")
    except Exception as e:
        logger.error(f"Failed to start background monitoring: {str(e)}")


def stop_background_monitoring():
    """Stop background monitoring services."""
    try:
        performance_monitor.stop_monitoring()
        logger.info("Background monitoring services stopped")
    except Exception as e:
        logger.error(f"Failed to stop background monitoring: {str(e)}")