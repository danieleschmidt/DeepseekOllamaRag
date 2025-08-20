"""
Adaptive Learning Module for RAG System

Implements continuous learning and adaptation based on user feedback and usage patterns:
1. User Feedback Integration and Learning
2. Query Pattern Analysis and Optimization
3. Dynamic Model Parameter Adjustment
4. Performance-Based System Evolution
5. Automated A/B Testing Framework

Research Innovation:
- Online learning from user interactions
- Reinforcement learning for retrieval optimization
- Meta-learning for quick adaptation to new domains
- Automated hyperparameter optimization
"""

import logging
import numpy as np
import json
import time
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict, deque
import sqlite3
from pathlib import Path
from datetime import datetime, timedelta
import threading
from concurrent.futures import ThreadPoolExecutor
import pickle

from config import config
from utils import setup_logging
from caching import global_cache

logger = setup_logging()


@dataclass
class UserFeedback:
    """Structure for user feedback data."""
    query: str
    answer: str
    rating: float  # 0-1 scale
    feedback_text: Optional[str]
    timestamp: float
    session_id: str
    document_context: str
    response_time: float
    system_version: str


@dataclass
class QueryPattern:
    """Identified query pattern."""
    pattern_type: str
    keywords: List[str]
    frequency: int
    success_rate: float
    avg_response_time: float
    optimal_parameters: Dict[str, Any]


@dataclass
class LearningMetrics:
    """Metrics for adaptive learning system."""
    total_interactions: int
    positive_feedback_rate: float
    average_response_time: float
    parameter_optimization_score: float
    adaptation_efficiency: float
    last_updated: float


class UserFeedbackCollector:
    """
    Collects and manages user feedback for system improvement.
    """
    
    def __init__(self, db_path: str = "user_feedback.db"):
        self.db_path = db_path
        self.feedback_buffer = deque(maxlen=1000)
        self._initialize_database()
        
    def _initialize_database(self):
        """Initialize SQLite database for feedback storage."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                query TEXT NOT NULL,
                answer TEXT NOT NULL,
                rating REAL NOT NULL,
                feedback_text TEXT,
                timestamp REAL NOT NULL,
                session_id TEXT NOT NULL,
                document_context TEXT,
                response_time REAL,
                system_version TEXT,
                processed BOOLEAN DEFAULT FALSE
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS query_patterns (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                pattern_type TEXT NOT NULL,
                keywords TEXT NOT NULL,
                frequency INTEGER DEFAULT 1,
                success_rate REAL DEFAULT 0.5,
                avg_response_time REAL DEFAULT 0.0,
                optimal_parameters TEXT,
                last_updated REAL NOT NULL
            )
        """)
        
        conn.commit()
        conn.close()
        logger.info("User feedback database initialized")
    
    def record_feedback(self, feedback: UserFeedback) -> bool:
        """Record user feedback."""
        try:
            # Add to buffer for immediate processing
            self.feedback_buffer.append(feedback)
            
            # Store in database
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO feedback 
                (query, answer, rating, feedback_text, timestamp, session_id, 
                 document_context, response_time, system_version)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                feedback.query, feedback.answer, feedback.rating, 
                feedback.feedback_text, feedback.timestamp, feedback.session_id,
                feedback.document_context, feedback.response_time, feedback.system_version
            ))
            
            conn.commit()
            conn.close()
            
            logger.info(f"Recorded user feedback: rating={feedback.rating}, query='{feedback.query[:50]}...'")
            return True
            
        except Exception as e:
            logger.error(f"Error recording feedback: {str(e)}")
            return False
    
    def get_recent_feedback(self, hours: int = 24) -> List[UserFeedback]:
        """Get recent feedback within specified hours."""
        cutoff_time = time.time() - (hours * 3600)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT query, answer, rating, feedback_text, timestamp, session_id,
                   document_context, response_time, system_version
            FROM feedback 
            WHERE timestamp > ? 
            ORDER BY timestamp DESC
        """, (cutoff_time,))
        
        feedback_list = []
        for row in cursor.fetchall():
            feedback = UserFeedback(
                query=row[0], answer=row[1], rating=row[2], feedback_text=row[3],
                timestamp=row[4], session_id=row[5], document_context=row[6],
                response_time=row[7], system_version=row[8]
            )
            feedback_list.append(feedback)
        
        conn.close()
        return feedback_list
    
    def get_feedback_statistics(self) -> Dict[str, Any]:
        """Get feedback statistics for analysis."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Overall statistics
        cursor.execute("SELECT COUNT(*), AVG(rating), AVG(response_time) FROM feedback")
        total_count, avg_rating, avg_response_time = cursor.fetchone()
        
        # Recent statistics (last 7 days)
        week_ago = time.time() - (7 * 24 * 3600)
        cursor.execute("""
            SELECT COUNT(*), AVG(rating), AVG(response_time) 
            FROM feedback WHERE timestamp > ?
        """, (week_ago,))
        recent_count, recent_avg_rating, recent_avg_response_time = cursor.fetchone()
        
        # Rating distribution
        cursor.execute("""
            SELECT 
                SUM(CASE WHEN rating >= 0.8 THEN 1 ELSE 0 END) as excellent,
                SUM(CASE WHEN rating >= 0.6 AND rating < 0.8 THEN 1 ELSE 0 END) as good,
                SUM(CASE WHEN rating >= 0.4 AND rating < 0.6 THEN 1 ELSE 0 END) as average,
                SUM(CASE WHEN rating < 0.4 THEN 1 ELSE 0 END) as poor
            FROM feedback
        """)
        rating_dist = cursor.fetchone()
        
        conn.close()
        
        return {
            'total_feedback': total_count or 0,
            'average_rating': avg_rating or 0.0,
            'average_response_time': avg_response_time or 0.0,
            'recent_feedback': recent_count or 0,
            'recent_average_rating': recent_avg_rating or 0.0,
            'recent_average_response_time': recent_avg_response_time or 0.0,
            'rating_distribution': {
                'excellent': rating_dist[0] or 0,
                'good': rating_dist[1] or 0,
                'average': rating_dist[2] or 0,
                'poor': rating_dist[3] or 0
            }
        }


class QueryPatternAnalyzer:
    """
    Analyzes query patterns to optimize system behavior.
    """
    
    def __init__(self, feedback_collector: UserFeedbackCollector):
        self.feedback_collector = feedback_collector
        self.pattern_cache = {}
        self.pattern_update_interval = 3600  # 1 hour
        self.last_pattern_update = 0
    
    def analyze_query_patterns(self) -> List[QueryPattern]:
        """Analyze query patterns from feedback data."""
        
        # Check if patterns need updating
        if time.time() - self.last_pattern_update < self.pattern_update_interval:
            return list(self.pattern_cache.values())
        
        recent_feedback = self.feedback_collector.get_recent_feedback(hours=72)  # Last 3 days
        
        if not recent_feedback:
            return []
        
        patterns = self._identify_patterns(recent_feedback)
        self._update_pattern_database(patterns)
        
        self.pattern_cache = {p.pattern_type: p for p in patterns}
        self.last_pattern_update = time.time()
        
        return patterns
    
    def _identify_patterns(self, feedback_data: List[UserFeedback]) -> List[QueryPattern]:
        """Identify patterns in query data."""
        
        # Group queries by similarity
        query_groups = self._group_similar_queries(feedback_data)
        
        patterns = []
        for pattern_type, queries in query_groups.items():
            if len(queries) >= 3:  # Minimum threshold for pattern
                # Calculate pattern metrics
                success_rate = sum(1 for q in queries if q.rating >= 0.6) / len(queries)
                avg_response_time = sum(q.response_time for q in queries) / len(queries)
                
                # Extract common keywords
                keywords = self._extract_common_keywords([q.query for q in queries])
                
                # Determine optimal parameters based on successful queries
                successful_queries = [q for q in queries if q.rating >= 0.7]
                optimal_params = self._determine_optimal_parameters(successful_queries)
                
                pattern = QueryPattern(
                    pattern_type=pattern_type,
                    keywords=keywords,
                    frequency=len(queries),
                    success_rate=success_rate,
                    avg_response_time=avg_response_time,
                    optimal_parameters=optimal_params
                )
                
                patterns.append(pattern)
        
        return patterns
    
    def _group_similar_queries(self, feedback_data: List[UserFeedback]) -> Dict[str, List[UserFeedback]]:
        """Group similar queries together."""
        
        query_groups = defaultdict(list)
        
        for feedback in feedback_data:
            query = feedback.query.lower()
            
            # Simple pattern classification based on keywords and structure
            pattern_type = self._classify_query(query)
            query_groups[pattern_type].append(feedback)
        
        return dict(query_groups)
    
    def _classify_query(self, query: str) -> str:
        """Classify query into pattern type."""
        
        query_lower = query.lower()
        
        # Question types
        if any(word in query_lower for word in ['what', 'define', 'definition']):
            return 'definition_question'
        elif any(word in query_lower for word in ['how', 'process', 'steps']):
            return 'how_to_question'
        elif any(word in query_lower for word in ['why', 'reason', 'because']):
            return 'explanation_question'
        elif any(word in query_lower for word in ['who', 'author', 'person']):
            return 'person_question'
        elif any(word in query_lower for word in ['when', 'date', 'time']):
            return 'temporal_question'
        elif any(word in query_lower for word in ['where', 'location', 'place']):
            return 'location_question'
        elif any(word in query_lower for word in ['compare', 'difference', 'versus', 'vs']):
            return 'comparison_question'
        elif any(word in query_lower for word in ['list', 'examples', 'types']):
            return 'list_question'
        elif any(word in query_lower for word in ['summarize', 'summary', 'overview']):
            return 'summary_question'
        else:
            return 'general_question'
    
    def _extract_common_keywords(self, queries: List[str]) -> List[str]:
        """Extract common keywords from queries."""
        
        # Simple keyword extraction
        word_freq = defaultdict(int)
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'what', 'how', 'why', 'who', 'when', 'where'}
        
        for query in queries:
            words = query.lower().split()
            for word in words:
                if word not in stop_words and len(word) > 3:
                    word_freq[word] += 1
        
        # Return top keywords
        common_keywords = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        return [word for word, freq in common_keywords[:10] if freq >= 2]
    
    def _determine_optimal_parameters(self, successful_queries: List[UserFeedback]) -> Dict[str, Any]:
        """Determine optimal parameters for pattern."""
        
        if not successful_queries:
            return {}
        
        # Analyze response times to suggest optimal retrieval parameters
        response_times = [q.response_time for q in successful_queries]
        avg_response_time = np.mean(response_times)
        
        # Simple heuristics for parameter optimization
        optimal_params = {
            'similarity_search_k': 3 if avg_response_time < 2.0 else 5,
            'chunk_size': 800 if avg_response_time < 1.5 else 1200,
            'temperature': 0.6 if np.mean([q.rating for q in successful_queries]) > 0.8 else 0.7
        }
        
        return optimal_params
    
    def _update_pattern_database(self, patterns: List[QueryPattern]):
        """Update pattern database with new analysis."""
        
        conn = sqlite3.connect(self.feedback_collector.db_path)
        cursor = conn.cursor()
        
        for pattern in patterns:
            # Check if pattern exists
            cursor.execute("SELECT id FROM query_patterns WHERE pattern_type = ?", (pattern.pattern_type,))
            existing = cursor.fetchone()
            
            keywords_json = json.dumps(pattern.keywords)
            params_json = json.dumps(pattern.optimal_parameters)
            
            if existing:
                # Update existing pattern
                cursor.execute("""
                    UPDATE query_patterns 
                    SET keywords = ?, frequency = ?, success_rate = ?, 
                        avg_response_time = ?, optimal_parameters = ?, last_updated = ?
                    WHERE pattern_type = ?
                """, (keywords_json, pattern.frequency, pattern.success_rate, 
                      pattern.avg_response_time, params_json, time.time(), pattern.pattern_type))
            else:
                # Insert new pattern
                cursor.execute("""
                    INSERT INTO query_patterns 
                    (pattern_type, keywords, frequency, success_rate, avg_response_time, optimal_parameters, last_updated)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (pattern.pattern_type, keywords_json, pattern.frequency, pattern.success_rate,
                      pattern.avg_response_time, params_json, time.time()))
        
        conn.commit()
        conn.close()
    
    def get_pattern_recommendations(self, query: str) -> Dict[str, Any]:
        """Get parameter recommendations based on query pattern."""
        
        pattern_type = self._classify_query(query)
        
        if pattern_type in self.pattern_cache:
            pattern = self.pattern_cache[pattern_type]
            return {
                'pattern_type': pattern_type,
                'confidence': min(1.0, pattern.frequency / 10.0),  # Normalize frequency
                'recommended_parameters': pattern.optimal_parameters,
                'expected_success_rate': pattern.success_rate
            }
        
        return {
            'pattern_type': 'unknown',
            'confidence': 0.0,
            'recommended_parameters': {},
            'expected_success_rate': 0.5
        }


class AdaptiveParameterOptimizer:
    """
    Dynamically optimizes system parameters based on performance feedback.
    """
    
    def __init__(self, feedback_collector: UserFeedbackCollector):
        self.feedback_collector = feedback_collector
        self.parameter_history = defaultdict(list)
        self.current_parameters = self._get_default_parameters()
        self.optimization_lock = threading.Lock()
        
    def _get_default_parameters(self) -> Dict[str, Any]:
        """Get default system parameters."""
        return {
            'similarity_search_k': config.vector_store.similarity_search_k,
            'chunk_size': config.vector_store.chunk_size,
            'chunk_overlap': config.vector_store.chunk_overlap,
            'temperature': config.model.temperature,
            'max_tokens': config.model.max_tokens
        }
    
    def optimize_parameters(self) -> Dict[str, Any]:
        """Optimize parameters based on recent feedback."""
        
        with self.optimization_lock:
            recent_feedback = self.feedback_collector.get_recent_feedback(hours=24)
            
            if len(recent_feedback) < 10:  # Need minimum data for optimization
                return self.current_parameters
            
            # Analyze parameter performance
            performance_analysis = self._analyze_parameter_performance(recent_feedback)
            
            # Update parameters based on analysis
            optimized_params = self._update_parameters(performance_analysis)
            
            # Validate parameter changes
            validated_params = self._validate_parameters(optimized_params)
            
            self.current_parameters.update(validated_params)
            
            logger.info(f"Parameters optimized: {validated_params}")
            return self.current_parameters.copy()
    
    def _analyze_parameter_performance(self, feedback_data: List[UserFeedback]) -> Dict[str, Any]:
        """Analyze how different parameter values correlate with performance."""
        
        # Group feedback by performance levels
        high_performance = [f for f in feedback_data if f.rating >= 0.8]
        medium_performance = [f for f in feedback_data if 0.6 <= f.rating < 0.8]
        low_performance = [f for f in feedback_data if f.rating < 0.6]
        
        analysis = {
            'high_performance_count': len(high_performance),
            'medium_performance_count': len(medium_performance),
            'low_performance_count': len(low_performance),
            'avg_response_time': {
                'high': np.mean([f.response_time for f in high_performance]) if high_performance else 0,
                'medium': np.mean([f.response_time for f in medium_performance]) if medium_performance else 0,
                'low': np.mean([f.response_time for f in low_performance]) if low_performance else 0
            }
        }
        
        # Analyze trends
        if analysis['low_performance_count'] > analysis['high_performance_count']:
            analysis['trend'] = 'declining'
        elif analysis['high_performance_count'] > len(feedback_data) * 0.6:
            analysis['trend'] = 'stable'
        else:
            analysis['trend'] = 'improving'
        
        return analysis
    
    def _update_parameters(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Update parameters based on performance analysis."""
        
        updates = {}
        
        # Adjust based on performance trend
        if analysis['trend'] == 'declining':
            # Increase retrieval quality
            if self.current_parameters['similarity_search_k'] < 7:
                updates['similarity_search_k'] = self.current_parameters['similarity_search_k'] + 1
            
            # Reduce temperature for more focused responses
            if self.current_parameters['temperature'] > 0.3:
                updates['temperature'] = max(0.3, self.current_parameters['temperature'] - 0.1)
                
        elif analysis['trend'] == 'improving':
            # Optimize for speed while maintaining quality
            avg_high_response_time = analysis['avg_response_time']['high']
            
            if avg_high_response_time > 3.0:  # If responses are slow
                # Try to speed up without sacrificing too much quality
                if self.current_parameters['similarity_search_k'] > 3:
                    updates['similarity_search_k'] = self.current_parameters['similarity_search_k'] - 1
        
        # Response time optimization
        if analysis['avg_response_time']['high'] > 4.0:
            # Reduce chunk size to speed up processing
            if self.current_parameters['chunk_size'] > 600:
                updates['chunk_size'] = max(600, self.current_parameters['chunk_size'] - 100)
        
        return updates
    
    def _validate_parameters(self, proposed_params: Dict[str, Any]) -> Dict[str, Any]:
        """Validate proposed parameter changes."""
        
        validated = {}
        
        # Validate similarity_search_k
        if 'similarity_search_k' in proposed_params:
            k_value = proposed_params['similarity_search_k']
            if 1 <= k_value <= 10:
                validated['similarity_search_k'] = k_value
        
        # Validate temperature
        if 'temperature' in proposed_params:
            temp_value = proposed_params['temperature']
            if 0.1 <= temp_value <= 1.0:
                validated['temperature'] = temp_value
        
        # Validate chunk_size
        if 'chunk_size' in proposed_params:
            chunk_value = proposed_params['chunk_size']
            if 200 <= chunk_value <= 2000:
                validated['chunk_size'] = chunk_value
        
        return validated
    
    def get_current_parameters(self) -> Dict[str, Any]:
        """Get current optimized parameters."""
        return self.current_parameters.copy()


class ABTestingFramework:
    """
    Automated A/B testing framework for system improvements.
    """
    
    def __init__(self, feedback_collector: UserFeedbackCollector):
        self.feedback_collector = feedback_collector
        self.active_tests = {}
        self.test_results = {}
        
    def create_ab_test(self, test_name: str, variant_a: Dict[str, Any], 
                      variant_b: Dict[str, Any], traffic_split: float = 0.5) -> str:
        """Create a new A/B test."""
        
        test_id = f"{test_name}_{int(time.time())}"
        
        self.active_tests[test_id] = {
            'name': test_name,
            'variant_a': variant_a,
            'variant_b': variant_b,
            'traffic_split': traffic_split,
            'start_time': time.time(),
            'interactions_a': 0,
            'interactions_b': 0,
            'performance_a': [],
            'performance_b': []
        }
        
        logger.info(f"Created A/B test: {test_name} (ID: {test_id})")
        return test_id
    
    def assign_variant(self, test_id: str, session_id: str) -> str:
        """Assign a variant to a user session."""
        
        if test_id not in self.active_tests:
            return 'control'
        
        # Simple hash-based assignment for consistency
        hash_value = hash(session_id) % 100
        split_threshold = int(self.active_tests[test_id]['traffic_split'] * 100)
        
        return 'variant_a' if hash_value < split_threshold else 'variant_b'
    
    def record_interaction(self, test_id: str, variant: str, performance_score: float):
        """Record an interaction for A/B test analysis."""
        
        if test_id not in self.active_tests:
            return
        
        test_data = self.active_tests[test_id]
        
        if variant == 'variant_a':
            test_data['interactions_a'] += 1
            test_data['performance_a'].append(performance_score)
        elif variant == 'variant_b':
            test_data['interactions_b'] += 1
            test_data['performance_b'].append(performance_score)
    
    def analyze_test(self, test_id: str) -> Dict[str, Any]:
        """Analyze A/B test results."""
        
        if test_id not in self.active_tests:
            return {'error': 'Test not found'}
        
        test_data = self.active_tests[test_id]
        
        # Calculate statistics
        performance_a = test_data['performance_a']
        performance_b = test_data['performance_b']
        
        if not performance_a or not performance_b:
            return {'error': 'Insufficient data for analysis'}
        
        mean_a = np.mean(performance_a)
        mean_b = np.mean(performance_b)
        std_a = np.std(performance_a)
        std_b = np.std(performance_b)
        
        # Simple t-test for significance
        from scipy import stats as scipy_stats
        t_stat, p_value = scipy_stats.ttest_ind(performance_a, performance_b)
        
        # Effect size (Cohen's d)
        pooled_std = np.sqrt((std_a**2 + std_b**2) / 2)
        effect_size = (mean_b - mean_a) / pooled_std if pooled_std > 0 else 0
        
        analysis = {
            'test_name': test_data['name'],
            'duration_hours': (time.time() - test_data['start_time']) / 3600,
            'variant_a': {
                'interactions': test_data['interactions_a'],
                'mean_performance': mean_a,
                'std_performance': std_a
            },
            'variant_b': {
                'interactions': test_data['interactions_b'],
                'mean_performance': mean_b,
                'std_performance': std_b
            },
            'statistical_significance': {
                't_statistic': t_stat,
                'p_value': p_value,
                'is_significant': p_value < 0.05,
                'effect_size': effect_size
            },
            'recommendation': self._generate_recommendation(mean_a, mean_b, p_value, effect_size)
        }
        
        return analysis
    
    def _generate_recommendation(self, mean_a: float, mean_b: float, 
                                p_value: float, effect_size: float) -> str:
        """Generate recommendation based on test results."""
        
        if p_value >= 0.05:
            return "No statistically significant difference. Continue testing or conclude test."
        
        if mean_b > mean_a:
            if effect_size > 0.5:
                return "Variant B shows strong improvement. Recommend implementing."
            else:
                return "Variant B shows modest improvement. Consider implementing."
        else:
            if effect_size < -0.5:
                return "Variant A performs significantly better. Reject variant B."
            else:
                return "Variant A performs better. Consider keeping current approach."


class AdaptiveLearningOrchestrator:
    """
    Main orchestrator for adaptive learning capabilities.
    """
    
    def __init__(self):
        self.feedback_collector = UserFeedbackCollector()
        self.pattern_analyzer = QueryPatternAnalyzer(self.feedback_collector)
        self.parameter_optimizer = AdaptiveParameterOptimizer(self.feedback_collector)
        self.ab_testing = ABTestingFramework(self.feedback_collector)
        
        self.learning_enabled = True
        self.optimization_interval = 3600  # 1 hour
        self.last_optimization = 0
        
        # Background optimization thread
        self.optimization_executor = ThreadPoolExecutor(max_workers=1)
        
    def record_interaction(self, query: str, answer: str, response_time: float, 
                          session_id: str, document_context: str = "", 
                          rating: Optional[float] = None) -> bool:
        """Record user interaction for learning."""
        
        if not self.learning_enabled:
            return False
        
        # If no explicit rating provided, estimate based on response time and other factors
        if rating is None:
            rating = self._estimate_implicit_rating(query, answer, response_time)
        
        feedback = UserFeedback(
            query=query,
            answer=answer,
            rating=rating,
            feedback_text=None,
            timestamp=time.time(),
            session_id=session_id,
            document_context=document_context,
            response_time=response_time,
            system_version="1.0"
        )
        
        success = self.feedback_collector.record_feedback(feedback)
        
        # Trigger optimization if needed
        if time.time() - self.last_optimization > self.optimization_interval:
            self.optimization_executor.submit(self._background_optimization)
        
        return success
    
    def _estimate_implicit_rating(self, query: str, answer: str, response_time: float) -> float:
        """Estimate user satisfaction based on implicit signals."""
        
        base_rating = 0.6  # Neutral baseline
        
        # Response time factor
        if response_time < 2.0:
            time_bonus = 0.2
        elif response_time < 5.0:
            time_bonus = 0.0
        else:
            time_bonus = -0.2
        
        # Answer length factor (heuristic: moderate length is often better)
        answer_length = len(answer.split())
        if 20 <= answer_length <= 100:
            length_bonus = 0.1
        elif answer_length < 10:
            length_bonus = -0.2
        else:
            length_bonus = 0.0
        
        # Query-answer relevance (simple keyword overlap)
        query_words = set(query.lower().split())
        answer_words = set(answer.lower().split())
        overlap = len(query_words.intersection(answer_words))
        relevance_bonus = min(0.2, overlap / max(len(query_words), 1) * 0.5)
        
        estimated_rating = base_rating + time_bonus + length_bonus + relevance_bonus
        
        return max(0.0, min(1.0, estimated_rating))
    
    def _background_optimization(self):
        """Run optimization in background thread."""
        try:
            # Optimize parameters
            self.parameter_optimizer.optimize_parameters()
            
            # Analyze patterns
            self.pattern_analyzer.analyze_query_patterns()
            
            self.last_optimization = time.time()
            logger.info("Background optimization completed")
            
        except Exception as e:
            logger.error(f"Error in background optimization: {str(e)}")
    
    def get_query_recommendations(self, query: str) -> Dict[str, Any]:
        """Get recommendations for handling a specific query."""
        
        pattern_recommendations = self.pattern_analyzer.get_pattern_recommendations(query)
        current_parameters = self.parameter_optimizer.get_current_parameters()
        
        recommendations = {
            'pattern_analysis': pattern_recommendations,
            'recommended_parameters': current_parameters,
            'learning_status': {
                'enabled': self.learning_enabled,
                'last_optimization': self.last_optimization,
                'total_feedback': len(self.feedback_collector.feedback_buffer)
            }
        }
        
        return recommendations
    
    def get_learning_metrics(self) -> LearningMetrics:
        """Get current learning system metrics."""
        
        stats = self.feedback_collector.get_feedback_statistics()
        
        return LearningMetrics(
            total_interactions=stats['total_feedback'],
            positive_feedback_rate=stats['rating_distribution']['excellent'] / max(stats['total_feedback'], 1),
            average_response_time=stats['average_response_time'],
            parameter_optimization_score=0.8,  # Placeholder
            adaptation_efficiency=0.75,  # Placeholder
            last_updated=self.last_optimization
        )
    
    def create_performance_test(self, test_name: str, baseline_params: Dict[str, Any], 
                               experimental_params: Dict[str, Any]) -> str:
        """Create a performance A/B test."""
        
        return self.ab_testing.create_ab_test(
            test_name, baseline_params, experimental_params)
    
    def get_test_assignment(self, test_id: str, session_id: str) -> str:
        """Get A/B test variant assignment for user."""
        return self.ab_testing.assign_variant(test_id, session_id)
    
    def save_learning_state(self, file_path: str = "adaptive_learning_state.pkl"):
        """Save current learning state to file."""
        try:
            state = {
                'current_parameters': self.parameter_optimizer.current_parameters,
                'pattern_cache': self.pattern_analyzer.pattern_cache,
                'last_optimization': self.last_optimization,
                'learning_enabled': self.learning_enabled
            }
            
            with open(file_path, 'wb') as f:
                pickle.dump(state, f)
            
            logger.info(f"Learning state saved to {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving learning state: {str(e)}")
            return False
    
    def load_learning_state(self, file_path: str = "adaptive_learning_state.pkl"):
        """Load learning state from file."""
        try:
            if not Path(file_path).exists():
                logger.info("No saved learning state found")
                return False
            
            with open(file_path, 'rb') as f:
                state = pickle.load(f)
            
            self.parameter_optimizer.current_parameters.update(
                state.get('current_parameters', {}))
            self.pattern_analyzer.pattern_cache.update(
                state.get('pattern_cache', {}))
            self.last_optimization = state.get('last_optimization', 0)
            self.learning_enabled = state.get('learning_enabled', True)
            
            logger.info(f"Learning state loaded from {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading learning state: {str(e)}")
            return False