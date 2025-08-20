"""
Research Integration Module

Integrates research enhancements into the existing RAG system
with backward compatibility and performance monitoring.
"""

import logging
from typing import Optional, Dict, Any, Tuple
import streamlit as st
import time

from core import RAGPipeline
from research_enhancements import ResearchEnhancedRAG
from experimental_framework import ExperimentManager, create_research_benchmark
from config import config
from utils import setup_logging

logger = setup_logging()


class IntegratedRAGSystem:
    """
    Production system that integrates research enhancements
    with fallback to baseline system for reliability.
    """
    
    def __init__(self):
        self.baseline_system = RAGPipeline()
        self.enhanced_system = None
        self.use_enhanced = True
        self.performance_metrics = []
        
        # Initialize enhanced system with error handling
        try:
            self.enhanced_system = ResearchEnhancedRAG(self.baseline_system)
            logger.info("Enhanced RAG system initialized successfully")
        except Exception as e:
            logger.warning(f"Could not initialize enhanced system, using baseline: {str(e)}")
            self.use_enhanced = False
    
    def process_document(self, file_path: str) -> Tuple[Any, Any, str, Optional[Dict]]:
        """Process document with enhanced or baseline system."""
        
        if self.use_enhanced and self.enhanced_system:
            try:
                logger.info("Using enhanced document processing")
                return self.enhanced_system.enhanced_document_processing(file_path)
            except Exception as e:
                logger.warning(f"Enhanced processing failed, falling back to baseline: {str(e)}")
                self.use_enhanced = False
        
        # Fallback to baseline
        logger.info("Using baseline document processing")
        vector_store, qa_chain, doc_hash = self.baseline_system.process_document(file_path)
        return vector_store, qa_chain, doc_hash, None
    
    def ask_question(self, qa_chain, query: str, enhanced_metadata: Optional[Dict] = None) -> Dict[str, Any]:
        """Ask question with enhanced or baseline system."""
        
        start_time = time.time()
        
        if self.use_enhanced and self.enhanced_system and enhanced_metadata:
            try:
                logger.info("Using enhanced question answering")
                response = self.enhanced_system.enhanced_question_answering(
                    query, enhanced_metadata, qa_chain)
                
                # Track performance
                self._track_performance(query, time.time() - start_time, "enhanced", True)
                return response
                
            except Exception as e:
                logger.warning(f"Enhanced QA failed, falling back to baseline: {str(e)}")
                self.use_enhanced = False
        
        # Fallback to baseline
        logger.info("Using baseline question answering")
        response = self.baseline_system.ask_question(qa_chain, query)
        
        # Track performance
        self._track_performance(query, time.time() - start_time, "baseline", True)
        return response
    
    def _track_performance(self, query: str, response_time: float, 
                          system_type: str, success: bool):
        """Track system performance metrics."""
        
        self.performance_metrics.append({
            'query': query[:50],  # Truncated for privacy
            'response_time': response_time,
            'system_type': system_type,
            'success': success,
            'timestamp': time.time()
        })
        
        # Keep only last 100 metrics to avoid memory issues
        if len(self.performance_metrics) > 100:
            self.performance_metrics = self.performance_metrics[-100:]
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary for monitoring."""
        
        if not self.performance_metrics:
            return {"status": "No metrics available"}
        
        baseline_metrics = [m for m in self.performance_metrics if m['system_type'] == 'baseline']
        enhanced_metrics = [m for m in self.performance_metrics if m['system_type'] == 'enhanced']
        
        summary = {
            'total_queries': len(self.performance_metrics),
            'enhanced_system_usage': len(enhanced_metrics) / len(self.performance_metrics) if self.performance_metrics else 0,
            'baseline_system_usage': len(baseline_metrics) / len(self.performance_metrics) if self.performance_metrics else 0,
            'average_response_time': {
                'overall': sum(m['response_time'] for m in self.performance_metrics) / len(self.performance_metrics),
                'baseline': sum(m['response_time'] for m in baseline_metrics) / len(baseline_metrics) if baseline_metrics else 0,
                'enhanced': sum(m['response_time'] for m in enhanced_metrics) / len(enhanced_metrics) if enhanced_metrics else 0
            },
            'success_rate': sum(1 for m in self.performance_metrics if m['success']) / len(self.performance_metrics)
        }
        
        return summary


def add_research_monitoring_sidebar():
    """Add research monitoring to Streamlit sidebar."""
    
    if 'integrated_system' in st.session_state:
        integrated_system = st.session_state.integrated_system
        
        with st.sidebar:
            st.header("🔬 Research Features")
            
            # System status
            if integrated_system.use_enhanced:
                st.success("✅ Enhanced RAG Active")
            else:
                st.warning("⚠️ Baseline RAG Active")
            
            # Performance metrics
            performance = integrated_system.get_performance_summary()
            
            if performance.get('total_queries', 0) > 0:
                st.markdown("**Performance Metrics:**")
                st.metric("Total Queries", performance['total_queries'])
                st.metric("Enhanced Usage", f"{performance['enhanced_system_usage']:.1%}")
                st.metric("Avg Response Time", f"{performance['average_response_time']['overall']:.2f}s")
                st.metric("Success Rate", f"{performance['success_rate']:.1%}")
                
                if performance['average_response_time']['enhanced'] > 0:
                    baseline_time = performance['average_response_time']['baseline']
                    enhanced_time = performance['average_response_time']['enhanced']
                    
                    if baseline_time > 0:
                        speedup = baseline_time / enhanced_time
                        if speedup > 1:
                            st.success(f"🚀 {speedup:.1f}x Faster with Enhanced")
                        elif speedup < 1:
                            st.info(f"📊 Enhanced takes {1/speedup:.1f}x longer (more thorough)")
            
            # Research mode toggle
            if st.checkbox("Enable Experimental Features", key="research_mode"):
                st.info("🧪 Experimental mode activated")
                
                # Show research insights
                if integrated_system.enhanced_system:
                    research_summary = integrated_system.enhanced_system.get_research_summary()
                    
                    if research_summary.get('total_queries', 0) > 0:
                        st.markdown("**Research Insights:**")
                        metrics = research_summary['average_metrics']
                        
                        st.metric("Context Coherence", f"{metrics.get('context_coherence', 0):.3f}")
                        st.metric("Answer Relevance", f"{metrics.get('answer_relevance', 0):.3f}")
                        st.metric("Token Efficiency", f"{metrics.get('token_efficiency', 0):.3f}")


def initialize_integrated_system():
    """Initialize the integrated RAG system in Streamlit session."""
    
    if 'integrated_system' not in st.session_state:
        with st.spinner("🔬 Initializing research-enhanced RAG system..."):
            st.session_state.integrated_system = IntegratedRAGSystem()
        
        if st.session_state.integrated_system.use_enhanced:
            st.success("🎉 Research enhancements loaded successfully!")
        else:
            st.info("ℹ️ Running in baseline mode for maximum reliability")


def run_research_benchmark():
    """Run research benchmark if requested."""
    
    if st.sidebar.button("🏃‍♂️ Run Research Benchmark"):
        if 'integrated_system' not in st.session_state:
            st.error("Please initialize the system first")
            return
        
        with st.spinner("Running research benchmark..."):
            try:
                # Create experiment manager
                manager = ExperimentManager("benchmark_results")
                
                # Use current document as test document
                if st.session_state.get('current_document_path'):
                    test_documents = [st.session_state.current_document_path]
                    benchmark_queries = create_research_benchmark()
                    
                    experiment = manager.create_experiment(
                        name="Live_RAG_Benchmark",
                        description="Real-time comparison of baseline vs enhanced RAG",
                        test_queries=benchmark_queries[:5],  # Use subset for speed
                        test_documents=test_documents,
                        repetitions=1
                    )
                    
                    experiment_id = manager.run_experiment(experiment)
                    
                    st.success(f"✅ Benchmark completed! Experiment ID: {experiment_id}")
                    
                    # Show results summary
                    with open(f"benchmark_results/{experiment_id}_analysis.json") as f:
                        import json
                        analysis = json.load(f)
                        
                        st.markdown("### 📊 Benchmark Results")
                        
                        for metric, data in analysis.get('comparative_analysis', {}).items():
                            if 'improvement_percent' in metric:
                                metric_name = metric.replace('_improvement_percent', '').replace('_', ' ').title()
                                improvement = data
                                
                                if improvement > 0:
                                    st.success(f"**{metric_name}**: +{improvement:.1f}% improvement")
                                else:
                                    st.warning(f"**{metric_name}**: {improvement:.1f}% change")
                
                else:
                    st.warning("Please upload a document first to run benchmark")
                    
            except Exception as e:
                st.error(f"Benchmark failed: {str(e)}")
                logger.error(f"Benchmark error: {str(e)}")


# Backward compatibility functions for existing app.py
def enhanced_process_document(file_path: str):
    """Enhanced document processing with fallback."""
    if 'integrated_system' not in st.session_state:
        initialize_integrated_system()
    
    return st.session_state.integrated_system.process_document(file_path)


def enhanced_ask_question(qa_chain, query: str, enhanced_metadata: Optional[Dict] = None):
    """Enhanced question asking with fallback."""
    if 'integrated_system' not in st.session_state:
        initialize_integrated_system()
    
    return st.session_state.integrated_system.ask_question(qa_chain, query, enhanced_metadata)


# Research dashboard for detailed analysis
def show_research_dashboard():
    """Show detailed research dashboard."""
    
    st.header("🔬 Research Dashboard")
    
    if 'integrated_system' not in st.session_state:
        st.warning("Please initialize the integrated system first")
        return
    
    integrated_system = st.session_state.integrated_system
    
    # System overview
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("System Status")
        if integrated_system.use_enhanced:
            st.success("Enhanced RAG System Active")
            st.info("Research features enabled")
        else:
            st.warning("Baseline RAG System Active")
            st.info("Research features disabled (fallback mode)")
    
    with col2:
        st.subheader("Performance Overview")
        performance = integrated_system.get_performance_summary()
        
        if performance.get('total_queries', 0) > 0:
            st.metric("Total Queries Processed", performance['total_queries'])
            st.metric("Success Rate", f"{performance['success_rate']:.1%}")
            st.metric("Average Response Time", f"{performance['average_response_time']['overall']:.2f}s")
        else:
            st.info("No performance data available yet")
    
    # Research insights
    if integrated_system.use_enhanced and integrated_system.enhanced_system:
        st.subheader("Research Insights")
        
        research_summary = integrated_system.enhanced_system.get_research_summary()
        
        if research_summary.get('total_queries', 0) > 0:
            metrics = research_summary['average_metrics']
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Context Coherence", f"{metrics.get('context_coherence', 0):.3f}")
                st.metric("Answer Relevance", f"{metrics.get('answer_relevance', 0):.3f}")
            
            with col2:
                st.metric("Retrieval Precision", f"{metrics.get('retrieval_precision', 0):.3f}")
                st.metric("Token Efficiency", f"{metrics.get('token_efficiency', 0):.3f}")
            
            with col3:
                st.metric("Response Time", f"{metrics.get('response_time', 0):.2f}s")
                
                # Show performance trend
                trends = research_summary.get('performance_trends', {})
                coherence_trend = trends.get('improving_coherence', 0)
                
                if coherence_trend > 0:
                    st.success("📈 Improving Over Time")
                elif coherence_trend < 0:
                    st.warning("📉 Declining Performance")
                else:
                    st.info("📊 Stable Performance")
        else:
            st.info("No research metrics available yet. Process some documents and ask questions!")
    
    # Experimental controls
    st.subheader("Experimental Controls")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🔄 Reset Performance Metrics"):
            if 'integrated_system' in st.session_state:
                st.session_state.integrated_system.performance_metrics = []
                st.success("Performance metrics reset")
    
    with col2:
        if st.button("🧪 Force Enhanced Mode"):
            if 'integrated_system' in st.session_state:
                st.session_state.integrated_system.use_enhanced = True
                st.success("Enhanced mode enabled")