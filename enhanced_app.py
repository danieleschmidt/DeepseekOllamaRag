"""
Enhanced Streamlit Application with Full Research Integration

This enhanced version of the main application integrates all research enhancements:
1. Multi-modal document processing
2. Adaptive learning capabilities  
3. Research benchmarking tools
4. Advanced analytics and monitoring
5. A/B testing framework
"""

import streamlit as st
import logging
import time
import json
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Optional, Dict, Any, List

# Core imports
from config import config
from core import RAGPipeline
from utils import (
    init_session_state, check_session_timeout, reset_session,
    save_uploaded_file, validate_file, cleanup_temp_files,
    format_file_size, check_ollama_connection
)
from exceptions import (
    DeepSeekRAGException, DocumentProcessingError,
    LLMError, ValidationError, OllamaConnectionError
)

# Research enhancement imports
from research_integration import (
    IntegratedRAGSystem, initialize_integrated_system, 
    add_research_monitoring_sidebar, show_research_dashboard
)
from adaptive_learning import AdaptiveLearningOrchestrator, UserFeedback
from research_benchmarks import BenchmarkRunner
from experimental_framework import ExperimentManager
from multimodal_rag import MultiModalRAGEnhancer

# Configure Streamlit page
st.set_page_config(
    page_title="🔬 " + config.ui.page_title + " - Research Edition",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state and logging
init_session_state()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Enhanced session state for research features
if 'research_mode' not in st.session_state:
    st.session_state.research_mode = False
if 'adaptive_learning' not in st.session_state:
    st.session_state.adaptive_learning = None
if 'benchmark_runner' not in st.session_state:
    st.session_state.benchmark_runner = None
if 'experiment_manager' not in st.session_state:
    st.session_state.experiment_manager = None

# Apply enhanced styling
st.markdown(f"""
    <style>
    /* Main Background */
    .stApp {{
        background: linear-gradient(135deg, {config.ui.background_color} 0%, #E8F4FD 100%);
        color: #212529;
    }}
    
    /* Enhanced Sidebar Styling */
    [data-testid="stSidebar"] {{
        background: linear-gradient(180deg, {config.ui.sidebar_background} 0%, #1a1d21 100%) !important;
        color: #FFFFFF !important;
        border-right: 2px solid #007BFF;
    }}
    [data-testid="stSidebar"] * {{
        color: #FFFFFF !important;
        font-size: 16px !important;
    }}
    
    /* Research Mode Indicator */
    .research-badge {{
        background: linear-gradient(45deg, #FF6B6B, #4ECDC4);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: bold;
        text-align: center;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }}
    
    /* Enhanced Headings */
    h1, h2, h3, h4, h5, h6 {{
        color: #1a1a1a !important;
        font-weight: bold;
        text-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }}
    
    /* Enhanced File Uploader */
    .stFileUploader>div>div>div>button {{
        background: linear-gradient(45deg, {config.ui.secondary_color}, #FFD93D);
        color: #000000;
        font-weight: bold;
        border-radius: 12px;
        border: none;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        transition: transform 0.2s;
    }}
    .stFileUploader>div>div>div>button:hover {{
        transform: translateY(-2px);
    }}
    
    /* Enhanced Buttons */
    .stButton>button {{
        background: linear-gradient(45deg, {config.ui.primary_color}, #0056b3);
        color: white;
        border-radius: 12px;
        border: none;
        font-weight: bold;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        transition: all 0.3s;
    }}
    .stButton>button:hover {{
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0,0,0,0.3);
    }}
    
    /* Enhanced Metrics */
    .stMetric {{
        background: rgba(255,255,255,0.9);
        border-radius: 12px;
        padding: 1rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        border-left: 4px solid {config.ui.primary_color};
    }}
    
    /* Enhanced Alerts */
    .stAlert {{
        border-radius: 12px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        border: none;
    }}
    
    /* Progress Bar Enhancement */
    .stProgress > div > div {{
        background: linear-gradient(45deg, {config.ui.primary_color}, #00d4ff);
        border-radius: 10px;
    }}
    
    /* Research Dashboard Cards */
    .research-card {{
        background: rgba(255,255,255,0.95);
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 8px 25px rgba(0,0,0,0.1);
        border-left: 5px solid #4ECDC4;
    }}
    </style>
""", unsafe_allow_html=True)

# Initialize research systems
def initialize_research_systems():
    """Initialize all research enhancement systems."""
    
    # Initialize integrated RAG system
    initialize_integrated_system()
    
    # Initialize adaptive learning
    if st.session_state.adaptive_learning is None:
        with st.spinner("🧠 Initializing adaptive learning system..."):
            st.session_state.adaptive_learning = AdaptiveLearningOrchestrator()
            
            # Load previous learning state
            if st.session_state.adaptive_learning.load_learning_state():
                st.success("📚 Previous learning state loaded!")
    
    # Initialize benchmark runner
    if st.session_state.benchmark_runner is None:
        with st.spinner("📊 Initializing benchmark system..."):
            st.session_state.benchmark_runner = BenchmarkRunner()
    
    # Initialize experiment manager
    if st.session_state.experiment_manager is None:
        with st.spinner("🔬 Initializing experiment framework..."):
            st.session_state.experiment_manager = ExperimentManager()

# Check session timeout
if check_session_timeout():
    st.warning("🕐 Your session has expired. Please refresh the page.")
    if st.button("🔄 Reset Session"):
        reset_session()
        st.rerun()

# Check Ollama connection
if not check_ollama_connection():
    st.error("⚠️ Cannot connect to Ollama. Please ensure Ollama is running.")
    st.info(f"Expected Ollama URL: {config.model.ollama_base_url}")
    st.stop()

# Initialize research systems
initialize_research_systems()

# App title with research badge
col1, col2 = st.columns([3, 1])
with col1:
    st.title("🔬 " + config.ui.app_title + " - Research Edition")

with col2:
    if st.session_state.get('research_mode', False):
        st.markdown('<div class="research-badge">🧪 RESEARCH MODE</div>', unsafe_allow_html=True)

# Clean up old temp files
cleanup_temp_files()

# Enhanced sidebar with research features
with st.sidebar:
    st.header("🎯 Navigation")
    
    # Main navigation
    page = st.radio("Select View", [
        "📋 Document Q&A", 
        "🔬 Research Dashboard", 
        "📊 Benchmarks", 
        "🧪 Experiments",
        "📈 Analytics"
    ])
    
    st.markdown("---")
    
    # Research mode toggle
    research_mode = st.checkbox("🧪 Enable Research Mode", 
                               value=st.session_state.get('research_mode', False))
    st.session_state.research_mode = research_mode
    
    if research_mode:
        st.success("Research features activated!")
        
        # Advanced settings
        st.header("⚙️ Research Settings")
        
        # Adaptive learning settings
        learning_enabled = st.checkbox("🧠 Adaptive Learning", value=True)
        if learning_enabled and st.session_state.adaptive_learning:
            st.session_state.adaptive_learning.learning_enabled = True
            
            # Show learning metrics
            try:
                metrics = st.session_state.adaptive_learning.get_learning_metrics()
                st.metric("Total Interactions", metrics.total_interactions)
                st.metric("Positive Feedback Rate", f"{metrics.positive_feedback_rate:.1%}")
                st.metric("Avg Response Time", f"{metrics.average_response_time:.2f}s")
            except:
                pass
        
        # System monitoring
        add_research_monitoring_sidebar()
    
    st.markdown("---")
    
    # Standard sidebar content
    st.header("📋 Instructions")
    st.markdown("""
    1. **Upload** a PDF file using the uploader
    2. **Wait** for processing to complete
    3. **Ask** questions related to the document content
    4. **Review** answers with source references
    5. **Rate** responses to improve the system (Research Mode)
    """)
    
    st.header("⚙️ System Status")
    
    # Enhanced system status
    if check_ollama_connection():
        st.success("✅ Ollama Connected")
    else:
        st.error("❌ Ollama Disconnected")
    
    # Research system status
    if research_mode:
        systems_status = {
            "Integrated RAG": "integrated_system" in st.session_state,
            "Adaptive Learning": st.session_state.adaptive_learning is not None,
            "Benchmark Runner": st.session_state.benchmark_runner is not None,
            "Experiment Manager": st.session_state.experiment_manager is not None
        }
        
        for system, status in systems_status.items():
            if status:
                st.success(f"✅ {system}")
            else:
                st.warning(f"⚠️ {system}")
    
    # Configuration display
    st.markdown(f"""
    **Current Configuration:**
    - **Model**: `{config.model.llm_model}`
    - **Embeddings**: `{config.model.embedding_model.split('/')[-1]}`
    - **Search Results**: {config.vector_store.similarity_search_k}
    - **Max File Size**: {config.app.max_file_size_mb}MB
    """)
    
    # Session info
    if "session_start" in st.session_state:
        session_duration = datetime.now() - st.session_state.session_start
        st.markdown(f"**Session Duration**: {str(session_duration).split('.')[0]}")
    
    # Upload history
    if st.session_state.upload_history:
        st.header("📁 Recent Uploads")
        for i, upload in enumerate(st.session_state.upload_history[-3:]):
            st.markdown(f"**{i+1}.** {upload['name']} ({upload['size']})")
    
    # Reset button
    if st.button("🔄 Reset Session", key="reset_sidebar"):
        reset_session()
        st.rerun()

# Main content based on selected page
if page == "📋 Document Q&A":
    # Enhanced document Q&A interface
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("📁 Document Upload & Processing")
        
        uploaded_file = st.file_uploader(
            "Choose a PDF file",
            type=config.app.allowed_file_types,
            help=f"Maximum file size: {config.app.max_file_size_mb}MB"
        )
        
        if uploaded_file is not None:
            try:
                # Validate file
                validate_file(uploaded_file)
                
                # Display file info
                file_size = len(uploaded_file.getvalue())
                st.info(f"📄 **{uploaded_file.name}** ({format_file_size(file_size)})")
                
                # Check if this file was already processed
                file_hash = str(hash(uploaded_file.getvalue()))
                
                if file_hash in st.session_state.processed_documents:
                    st.success("✅ Document already processed! You can ask questions below.")
                    doc_data = st.session_state.processed_documents[file_hash]
                    
                    st.session_state.vector_store = doc_data['vector_store']
                    st.session_state.qa_chain = doc_data['qa_chain']
                    
                    # For research mode, also load enhanced metadata
                    if research_mode and 'enhanced_metadata' in doc_data:
                        st.session_state.enhanced_metadata = doc_data['enhanced_metadata']
                else:
                    # Process new document
                    with st.spinner("🔄 Processing document with research enhancements..."):
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        try:
                            # Save uploaded file
                            status_text.text("💾 Saving file...")
                            progress_bar.progress(20)
                            file_path = save_uploaded_file(uploaded_file)
                            st.session_state.current_document_path = file_path
                            
                            # Process document with research enhancements
                            if research_mode and 'integrated_system' in st.session_state:
                                status_text.text("🔬 Processing with research enhancements...")
                                progress_bar.progress(60)
                                
                                integrated_system = st.session_state.integrated_system
                                result = integrated_system.process_document(file_path)
                                
                                if len(result) == 4:  # Enhanced system returns 4 items
                                    vector_store, qa_chain, doc_hash, enhanced_metadata = result
                                    st.session_state.enhanced_metadata = enhanced_metadata
                                else:  # Fallback to baseline
                                    vector_store, qa_chain, doc_hash = result
                                    st.session_state.enhanced_metadata = None
                            else:
                                status_text.text("📖 Processing document...")
                                progress_bar.progress(60)
                                
                                # Use baseline system
                                if "rag_pipeline" not in st.session_state:
                                    st.session_state.rag_pipeline = RAGPipeline()
                                
                                vector_store, qa_chain, doc_hash = st.session_state.rag_pipeline.process_document(file_path)
                                st.session_state.enhanced_metadata = None
                            
                            # Store in session
                            status_text.text("💾 Storing results...")
                            progress_bar.progress(90)
                            
                            st.session_state.vector_store = vector_store
                            st.session_state.qa_chain = qa_chain
                            
                            doc_data = {
                                'vector_store': vector_store,
                                'qa_chain': qa_chain,
                                'file_name': uploaded_file.name,
                                'processed_at': datetime.now()
                            }
                            
                            if st.session_state.enhanced_metadata:
                                doc_data['enhanced_metadata'] = st.session_state.enhanced_metadata
                            
                            st.session_state.processed_documents[file_hash] = doc_data
                            
                            # Add to upload history
                            st.session_state.upload_history.append({
                                'name': uploaded_file.name,
                                'size': format_file_size(file_size),
                                'timestamp': datetime.now()
                            })
                            
                            progress_bar.progress(100)
                            status_text.text("✅ Processing complete!")
                            
                            # Clean up temp file
                            try:
                                import os
                                os.remove(file_path)
                            except:
                                pass
                            
                            st.success("🎉 Document processed successfully! Enhanced features active." if research_mode else "🎉 Document processed successfully!")
                            
                        except Exception as e:
                            progress_bar.empty()
                            status_text.empty()
                            if isinstance(e, DeepSeekRAGException):
                                st.error(f"❌ {str(e)}")
                            else:
                                st.error(f"❌ Unexpected error: {str(e)}")
                                logger.error(f"Document processing error: {str(e)}", exc_info=True)
                            
            except ValidationError as e:
                st.error(f"❌ Validation Error: {str(e)}")
            except Exception as e:
                st.error(f"❌ Unexpected error: {str(e)}")
                logger.error(f"File upload error: {str(e)}", exc_info=True)
    
    with col2:
        if uploaded_file is not None and st.session_state.qa_chain is not None:
            st.header("📊 Document Statistics")
            
            # Display document statistics
            try:
                # Get vector store stats
                if hasattr(st.session_state.vector_store, 'index'):
                    vector_count = st.session_state.vector_store.index.ntotal
                    st.metric("Document Chunks", vector_count)
                
                st.metric("Model", config.model.llm_model.split(':')[0])
                st.metric("Search Results", config.vector_store.similarity_search_k)
                
                # Enhanced stats for research mode
                if research_mode and st.session_state.enhanced_metadata:
                    metadata = st.session_state.enhanced_metadata
                    
                    if 'multimodal_features' in metadata:
                        features = metadata['multimodal_features']
                        st.metric("Images Found", features.get('num_images', 0))
                        st.metric("Entities Detected", features.get('num_entities', 0))
                        
                        if features.get('vision_available'):
                            st.success("👁️ Vision Analysis Active")
                        else:
                            st.info("📝 Text-Only Analysis")
                
            except Exception as e:
                st.warning("Could not load document statistics")
                if research_mode:
                    st.error(f"Debug: {str(e)}")
    
    # Enhanced Question and Answer Section
    if st.session_state.qa_chain is not None:
        st.header("❓ Intelligent Question Answering")
        
        # Enhanced question input with suggestions
        if research_mode:
            st.markdown("💡 **Smart Question Suggestions:**")
            
            suggestion_cols = st.columns(3)
            suggested_questions = [
                "What are the main findings?",
                "Summarize the key points",
                "Who are the authors?",
                "What methodology was used?",
                "What are the conclusions?",
                "List the key contributions"
            ]
            
            for i, suggestion in enumerate(suggested_questions):
                col_idx = i % 3
                with suggestion_cols[col_idx]:
                    if st.button(f"💭 {suggestion}", key=f"suggest_{i}"):
                        st.session_state.suggested_question = suggestion
        
        # Question input
        default_question = st.session_state.get('suggested_question', '')
        user_question = st.text_input(
            "Enter your question about the document:",
            value=default_question,
            placeholder="What is the main topic of this document?"
        )
        
        # Clear suggestion after use
        if 'suggested_question' in st.session_state:
            del st.session_state.suggested_question
        
        col1, col2, col3 = st.columns([1, 1, 2])
        
        with col1:
            ask_button = st.button("🔍 Ask Question", type="primary")
        
        with col2:
            if st.button("🗑️ Clear History"):
                if "qa_history" in st.session_state:
                    st.session_state.qa_history = []
                    st.rerun()
        
        # Process question with enhanced features
        if ask_button and user_question.strip():
            with st.spinner("🤔 Thinking with enhanced AI..."):
                try:
                    start_time = time.time()
                    
                    # Get response using appropriate system
                    if research_mode and 'integrated_system' in st.session_state and st.session_state.enhanced_metadata:
                        integrated_system = st.session_state.integrated_system
                        response = integrated_system.ask_question(
                            st.session_state.qa_chain, 
                            user_question, 
                            st.session_state.enhanced_metadata
                        )
                    else:
                        # Use baseline system
                        if "rag_pipeline" not in st.session_state:
                            st.session_state.rag_pipeline = RAGPipeline()
                        
                        response = st.session_state.rag_pipeline.ask_question(
                            st.session_state.qa_chain,
                            user_question
                        )
                    
                    response_time = time.time() - start_time
                    
                    # Store in history
                    if "qa_history" not in st.session_state:
                        st.session_state.qa_history = []
                    
                    qa_entry = {
                        'question': user_question,
                        'answer': response['answer'],
                        'sources': response['source_documents'],
                        'response_time': response_time,
                        'timestamp': datetime.now(),
                        'enhanced_features_used': research_mode
                    }
                    
                    # Add research-specific data
                    if research_mode and 'multimodal_context' in response:
                        qa_entry['multimodal_context'] = response['multimodal_context']
                        qa_entry['enhancement_metrics'] = response.get('enhancement_metrics', {})
                    
                    st.session_state.qa_history.append(qa_entry)
                    
                    # Record interaction for adaptive learning
                    if research_mode and st.session_state.adaptive_learning:
                        session_id = st.session_state.get('session_id', 'default')
                        document_context = uploaded_file.name if uploaded_file else ""
                        
                        st.session_state.adaptive_learning.record_interaction(
                            query=user_question,
                            answer=response['answer'],
                            response_time=response_time,
                            session_id=session_id,
                            document_context=document_context
                        )
                    
                    st.success(f"✅ Question processed successfully in {response_time:.2f}s!")
                    
                except Exception as e:
                    if isinstance(e, DeepSeekRAGException):
                        st.error(f"❌ {str(e)}")
                    else:
                        st.error(f"❌ Unexpected error: {str(e)}")
                        logger.error(f"Question processing error: {str(e)}", exc_info=True)
        
        # Enhanced Q&A History Display
        if "qa_history" in st.session_state and st.session_state.qa_history:
            st.header("💬 Q&A History")
            
            for i, qa in enumerate(reversed(st.session_state.qa_history[-5:])):
                with st.expander(f"Q{len(st.session_state.qa_history)-i}: {qa['question'][:60]}...", 
                               expanded=(i==0)):
                    
                    st.markdown(f"**Question:** {qa['question']}")
                    st.markdown(f"**Answer:** {qa['answer']}")
                    
                    # Enhanced metrics display
                    metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
                    
                    with metrics_col1:
                        st.metric("Response Time", f"{qa.get('response_time', 0):.2f}s")
                    
                    with metrics_col2:
                        enhancement_used = qa.get('enhanced_features_used', False)
                        st.metric("Enhancement", "✅ Active" if enhancement_used else "❌ Baseline")
                    
                    with metrics_col3:
                        if research_mode and 'enhancement_metrics' in qa:
                            metrics = qa['enhancement_metrics']
                            if metrics.get('multimodal_retrieval_used'):
                                st.success("🔍 Multi-modal")
                            if metrics.get('knowledge_graph_used'):
                                st.success("🕸️ Knowledge Graph")
                    
                    # Research mode: User feedback collection
                    if research_mode:
                        st.markdown("**Rate this response:**")
                        rating_cols = st.columns(5)
                        
                        for rating in range(1, 6):
                            with rating_cols[rating-1]:
                                if st.button(f"{'⭐' * rating}", key=f"rate_{i}_{rating}"):
                                    # Record feedback
                                    if st.session_state.adaptive_learning:
                                        feedback = UserFeedback(
                                            query=qa['question'],
                                            answer=qa['answer'],
                                            rating=rating / 5.0,  # Convert to 0-1 scale
                                            feedback_text=None,
                                            timestamp=time.time(),
                                            session_id=st.session_state.get('session_id', 'default'),
                                            document_context=uploaded_file.name if uploaded_file else "",
                                            response_time=qa.get('response_time', 0),
                                            system_version="research_1.0"
                                        )
                                        
                                        if st.session_state.adaptive_learning.feedback_collector.record_feedback(feedback):
                                            st.success(f"Thank you for rating: {rating}/5 stars!")
                                            
                                            # Save learning state
                                            st.session_state.adaptive_learning.save_learning_state()
                    
                    # Source documents
                    if qa['sources']:
                        st.markdown("**Sources:**")
                        for j, source in enumerate(qa['sources'][:2]):
                            content_preview = source.page_content[:200] + "..." if len(source.page_content) > 200 else source.page_content
                            st.markdown(f"*Source {j+1}:* {content_preview}")
                    
                    # Multi-modal context (research mode)
                    if research_mode and 'multimodal_context' in qa:
                        multimodal = qa['multimodal_context']
                        
                        if multimodal.get('visual_elements'):
                            st.markdown("**Visual Elements:**")
                            for elem in multimodal['visual_elements'][:2]:
                                st.info(f"📷 Image: {elem.get('image_data', {}).get('caption', 'No caption')}")
                        
                        if multimodal.get('knowledge_graph_info'):
                            st.markdown("**Knowledge Graph:**")
                            kg_info = multimodal['knowledge_graph_info'][0]  # Show first result
                            central_entity = kg_info.get('central_entity', {})
                            st.info(f"🕸️ Key Entity: {central_entity.get('text', 'Unknown')} ({central_entity.get('type', 'entity')})")
                    
                    st.markdown(f"*Asked at: {qa['timestamp'].strftime('%H:%M:%S')}*")
    
    else:
        st.info("👆 Please upload a PDF document to get started with enhanced AI capabilities.")

elif page == "🔬 Research Dashboard":
    show_research_dashboard()

elif page == "📊 Benchmarks":
    st.header("📊 Research Benchmarking Suite")
    
    if not research_mode:
        st.warning("⚠️ Please enable Research Mode in the sidebar to access benchmarking features.")
    else:
        # Benchmark interface
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("Available Benchmarks")
            
            if st.session_state.benchmark_runner:
                benchmarks = st.session_state.benchmark_runner.benchmark_loader.list_benchmarks()
                
                selected_benchmark = st.selectbox(
                    "Select Benchmark to Run:",
                    benchmarks,
                    format_func=lambda x: x.replace('_', ' ').title()
                )
                
                # Show benchmark info
                if selected_benchmark:
                    info = st.session_state.benchmark_runner.benchmark_loader.get_benchmark_info(selected_benchmark)
                    if info:
                        with st.expander("📋 Benchmark Details", expanded=True):
                            st.write(f"**Description:** {info['description']}")
                            st.write(f"**Domain:** {info['domain']}")
                            st.write(f"**Difficulty:** {info['difficulty']}")
                            st.write(f"**Questions:** {info['num_questions']}")
                
                # Run benchmark
                if st.button("🏃‍♂️ Run Benchmark"):
                    with st.spinner(f"Running benchmark: {selected_benchmark}..."):
                        try:
                            # Create systems for comparison
                            systems = {}
                            
                            # Baseline system
                            baseline = RAGPipeline()
                            systems['Baseline RAG'] = baseline
                            
                            # Enhanced system if available
                            if 'integrated_system' in st.session_state:
                                systems['Enhanced RAG'] = st.session_state.integrated_system
                            
                            # Run benchmark
                            results = {}
                            for system_name, system in systems.items():
                                st.info(f"Testing {system_name}...")
                                result = st.session_state.benchmark_runner.run_benchmark(
                                    system, system_name, selected_benchmark)
                                results[system_name] = [result]
                            
                            # Display results
                            st.success("✅ Benchmark completed!")
                            
                            # Generate comparison report
                            if len(results) > 1:
                                report = st.session_state.benchmark_runner.generate_comparison_report(results)
                                
                                st.subheader("📈 Results Summary")
                                
                                # Summary table
                                summary_data = []
                                for system, stats in report['summary'].items():
                                    summary_data.append({
                                        'System': system,
                                        'Overall Score': f"{stats['average_score']:.3f}",
                                        'Response Time': f"{stats['average_time']:.2f}s",
                                        'Score Std Dev': f"{stats['score_std']:.3f}"
                                    })
                                
                                summary_df = pd.DataFrame(summary_data)
                                st.dataframe(summary_df, use_container_width=True)
                                
                                # Recommendations
                                if report['recommendations']:
                                    st.subheader("🎯 Recommendations")
                                    for rec in report['recommendations']:
                                        st.info(f"💡 {rec}")
                                
                                # Statistical significance
                                if report['statistical_analysis']:
                                    st.subheader("📊 Statistical Analysis")
                                    for comparison, stats in report['statistical_analysis'].items():
                                        if stats['significant']:
                                            significance = "✅ Significant" if stats['p_value'] < 0.05 else "❌ Not Significant"
                                            effect_size = abs(stats['effect_size'])
                                            
                                            if effect_size > 0.8:
                                                effect_desc = "Large Effect"
                                            elif effect_size > 0.5:
                                                effect_desc = "Medium Effect"
                                            elif effect_size > 0.2:
                                                effect_desc = "Small Effect"
                                            else:
                                                effect_desc = "Minimal Effect"
                                            
                                            st.metric(
                                                comparison.replace('_vs_', ' vs '),
                                                f"p = {stats['p_value']:.4f}",
                                                f"{effect_desc} ({significance})"
                                            )
                            
                        except Exception as e:
                            st.error(f"❌ Benchmark failed: {str(e)}")
                            logger.error(f"Benchmark error: {str(e)}")
            else:
                st.error("Benchmark runner not initialized")
        
        with col2:
            st.subheader("🏆 Recent Results")
            
            # Show recent benchmark results
            results_dir = Path("benchmark_results")
            if results_dir.exists():
                result_files = sorted(results_dir.glob("*.json"), key=lambda x: x.stat().st_mtime, reverse=True)
                
                for result_file in result_files[:5]:  # Show last 5 results
                    try:
                        with open(result_file) as f:
                            result_data = json.load(f)
                        
                        timestamp = datetime.fromisoformat(result_data.get('timestamp', '2024-01-01T00:00:00'))
                        
                        with st.expander(f"📊 {result_data.get('benchmark_name', 'Unknown')} - {timestamp.strftime('%H:%M:%S')}"):
                            st.metric("Score", f"{result_data.get('overall_score', 0):.3f}")
                            st.metric("System", result_data.get('system_name', 'Unknown'))
                            st.metric("Time", f"{result_data.get('execution_time', 0):.2f}s")
                    except:
                        continue

elif page == "🧪 Experiments":
    st.header("🧪 Experimental Framework")
    
    if not research_mode:
        st.warning("⚠️ Please enable Research Mode in the sidebar to access experimental features.")
    else:
        st.subheader("🔬 A/B Testing and Controlled Experiments")
        
        # Experiment creation
        with st.expander("➕ Create New Experiment", expanded=False):
            exp_name = st.text_input("Experiment Name")
            exp_description = st.text_area("Description")
            
            # Parameter variants
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Variant A (Control)")
                a_similarity_k = st.slider("Similarity Search K", 1, 10, 3, key="a_k")
                a_temperature = st.slider("Temperature", 0.1, 1.0, 0.7, key="a_temp")
                
            with col2:
                st.subheader("Variant B (Treatment)")  
                b_similarity_k = st.slider("Similarity Search K", 1, 10, 5, key="b_k")
                b_temperature = st.slider("Temperature", 0.1, 1.0, 0.5, key="b_temp")
            
            if st.button("🚀 Create Experiment"):
                if exp_name and st.session_state.adaptive_learning:
                    variant_a = {
                        'similarity_search_k': a_similarity_k,
                        'temperature': a_temperature
                    }
                    
                    variant_b = {
                        'similarity_search_k': b_similarity_k,
                        'temperature': b_temperature
                    }
                    
                    test_id = st.session_state.adaptive_learning.create_performance_test(
                        exp_name, variant_a, variant_b)
                    
                    st.success(f"✅ Experiment created! ID: {test_id}")
                    st.session_state.active_experiment = test_id
                else:
                    st.error("Please provide experiment name and ensure adaptive learning is enabled")
        
        # Active experiments
        st.subheader("📊 Experiment Monitoring")
        
        if hasattr(st.session_state, 'active_experiment'):
            st.info(f"🔬 Active Experiment: {st.session_state.active_experiment}")
            
            # Show experiment status
            if st.button("📈 Analyze Experiment"):
                try:
                    analysis = st.session_state.adaptive_learning.ab_testing.analyze_test(
                        st.session_state.active_experiment)
                    
                    if 'error' not in analysis:
                        st.subheader("📊 Results")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.metric("Variant A Performance", 
                                     f"{analysis['variant_a']['mean_performance']:.3f}")
                            st.metric("Variant A Interactions", 
                                     analysis['variant_a']['interactions'])
                        
                        with col2:
                            st.metric("Variant B Performance", 
                                     f"{analysis['variant_b']['mean_performance']:.3f}")
                            st.metric("Variant B Interactions", 
                                     analysis['variant_b']['interactions'])
                        
                        # Statistical significance
                        stats_data = analysis['statistical_significance']
                        significance = "✅ Significant" if stats_data['is_significant'] else "❌ Not Significant"
                        
                        st.metric("Statistical Test", 
                                 f"p = {stats_data['p_value']:.4f}",
                                 significance)
                        
                        # Recommendation
                        st.subheader("🎯 Recommendation")
                        st.info(analysis['recommendation'])
                    else:
                        st.warning(analysis['error'])
                        
                except Exception as e:
                    st.error(f"Analysis failed: {str(e)}")
        else:
            st.info("No active experiments. Create one above to get started!")

elif page == "📈 Analytics":
    st.header("📈 Advanced Analytics")
    
    if not research_mode:
        st.warning("⚠️ Please enable Research Mode in the sidebar to access analytics features.")
    else:
        # System performance analytics
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🎯 Performance Metrics")
            
            if 'integrated_system' in st.session_state:
                performance = st.session_state.integrated_system.get_performance_summary()
                
                if performance.get('total_queries', 0) > 0:
                    st.metric("Total Queries", performance['total_queries'])
                    st.metric("Enhanced Usage", f"{performance['enhanced_system_usage']:.1%}")
                    st.metric("Success Rate", f"{performance['success_rate']:.1%}")
                    
                    # Response time comparison
                    if performance['average_response_time']['enhanced'] > 0:
                        baseline_time = performance['average_response_time']['baseline']
                        enhanced_time = performance['average_response_time']['enhanced']
                        
                        st.subheader("⚡ Response Time Analysis")
                        
                        time_data = pd.DataFrame({
                            'System': ['Baseline', 'Enhanced'],
                            'Avg Response Time (s)': [baseline_time, enhanced_time]
                        })
                        
                        st.bar_chart(time_data.set_index('System'))
                        
                        if baseline_time > 0:
                            speedup = baseline_time / enhanced_time
                            if speedup > 1:
                                st.success(f"🚀 Enhanced system is {speedup:.1f}x faster!")
                            else:
                                st.info(f"📊 Enhanced system takes {1/speedup:.1f}x longer for higher quality")
                else:
                    st.info("No performance data available yet")
        
        with col2:
            st.subheader("🧠 Learning Analytics")
            
            if st.session_state.adaptive_learning:
                try:
                    learning_metrics = st.session_state.adaptive_learning.get_learning_metrics()
                    feedback_stats = st.session_state.adaptive_learning.feedback_collector.get_feedback_statistics()
                    
                    st.metric("Total Interactions", learning_metrics.total_interactions)
                    st.metric("Positive Feedback Rate", f"{learning_metrics.positive_feedback_rate:.1%}")
                    st.metric("Adaptation Efficiency", f"{learning_metrics.adaptation_efficiency:.1%}")
                    
                    # Feedback distribution
                    if feedback_stats['total_feedback'] > 0:
                        st.subheader("📊 User Feedback Distribution")
                        
                        rating_dist = feedback_stats['rating_distribution']
                        
                        feedback_data = pd.DataFrame({
                            'Rating': ['Poor', 'Average', 'Good', 'Excellent'],
                            'Count': [rating_dist['poor'], rating_dist['average'], 
                                    rating_dist['good'], rating_dist['excellent']]
                        })
                        
                        st.bar_chart(feedback_data.set_index('Rating'))
                        
                        # Recent trends
                        if feedback_stats['recent_feedback'] > 0:
                            recent_improvement = (feedback_stats['recent_average_rating'] - 
                                                feedback_stats['average_rating']) * 100
                            
                            if recent_improvement > 0:
                                st.success(f"📈 Recent improvement: +{recent_improvement:.1f} rating points")
                            elif recent_improvement < 0:
                                st.warning(f"📉 Recent decline: {recent_improvement:.1f} rating points")
                            else:
                                st.info("📊 Stable recent performance")
                    
                except Exception as e:
                    st.error(f"Error loading learning analytics: {str(e)}")
            else:
                st.info("Adaptive learning not initialized")
        
        # Query pattern analysis
        if st.session_state.adaptive_learning:
            st.subheader("🔍 Query Pattern Analysis")
            
            try:
                patterns = st.session_state.adaptive_learning.pattern_analyzer.analyze_query_patterns()
                
                if patterns:
                    pattern_data = []
                    for pattern in patterns:
                        pattern_data.append({
                            'Pattern Type': pattern.pattern_type.replace('_', ' ').title(),
                            'Frequency': pattern.frequency,
                            'Success Rate': f"{pattern.success_rate:.1%}",
                            'Avg Response Time': f"{pattern.avg_response_time:.2f}s"
                        })
                    
                    patterns_df = pd.DataFrame(pattern_data)
                    st.dataframe(patterns_df, use_container_width=True)
                    
                    # Most common patterns
                    top_patterns = sorted(patterns, key=lambda x: x.frequency, reverse=True)[:3]
                    
                    st.subheader("🏆 Top Query Patterns")
                    for i, pattern in enumerate(top_patterns, 1):
                        with st.expander(f"{i}. {pattern.pattern_type.replace('_', ' ').title()} ({pattern.frequency} queries)"):
                            st.write(f"**Success Rate:** {pattern.success_rate:.1%}")
                            st.write(f"**Avg Response Time:** {pattern.avg_response_time:.2f}s")
                            st.write(f"**Common Keywords:** {', '.join(pattern.keywords[:5])}")
                            
                            if pattern.optimal_parameters:
                                st.write("**Optimal Parameters:**")
                                for param, value in pattern.optimal_parameters.items():
                                    st.write(f"  - {param}: {value}")
                else:
                    st.info("No query patterns identified yet. Ask more questions to build pattern analysis!")
                    
            except Exception as e:
                st.error(f"Error analyzing query patterns: {str(e)}")
        
        # Export analytics data
        st.subheader("📊 Data Export")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("💾 Export Performance Data"):
                if 'integrated_system' in st.session_state:
                    performance_data = st.session_state.integrated_system.get_performance_summary()
                    
                    # Convert to JSON for download
                    performance_json = json.dumps(performance_data, indent=2, default=str)
                    
                    st.download_button(
                        label="📥 Download Performance Data",
                        data=performance_json,
                        file_name=f"performance_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json"
                    )
        
        with col2:
            if st.button("💾 Export Learning Data"):
                if st.session_state.adaptive_learning:
                    # Save current learning state
                    st.session_state.adaptive_learning.save_learning_state()
                    
                    # Get learning metrics
                    learning_data = {
                        'metrics': asdict(st.session_state.adaptive_learning.get_learning_metrics()),
                        'feedback_stats': st.session_state.adaptive_learning.feedback_collector.get_feedback_statistics()
                    }
                    
                    learning_json = json.dumps(learning_data, indent=2, default=str)
                    
                    st.download_button(
                        label="📥 Download Learning Data",
                        data=learning_json,
                        file_name=f"learning_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json"
                    )
        
        with col3:
            if st.button("💾 Export Q&A History"):
                if "qa_history" in st.session_state and st.session_state.qa_history:
                    qa_data = []
                    
                    for qa in st.session_state.qa_history:
                        qa_export = {
                            'question': qa['question'],
                            'answer': qa['answer'],
                            'response_time': qa.get('response_time', 0),
                            'timestamp': qa['timestamp'].isoformat(),
                            'enhanced_features_used': qa.get('enhanced_features_used', False)
                        }
                        qa_data.append(qa_export)
                    
                    qa_json = json.dumps(qa_data, indent=2, default=str)
                    
                    st.download_button(
                        label="📥 Download Q&A History",
                        data=qa_json,
                        file_name=f"qa_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json"
                    )

# Footer with research edition information
st.markdown("---")
st.markdown(
    f"""
    <div style='text-align: center; color: #666; padding: 20px;'>
    🔬 <strong>DeepSeek RAG Research Edition v1.0</strong><br>
    Enhanced with: Hierarchical Document Understanding • Multi-Modal Processing • Adaptive Learning<br>
    Knowledge Graphs • Experimental Framework • Advanced Benchmarking<br>
    <em>Powered by DeepSeek R1 + Ollama • Built with Streamlit</em>
    </div>
    """, 
    unsafe_allow_html=True
)

# Research mode indicator
if research_mode:
    st.markdown(
        """
        <div style='position: fixed; top: 10px; right: 10px; background: linear-gradient(45deg, #FF6B6B, #4ECDC4); 
                    color: white; padding: 0.5rem 1rem; border-radius: 20px; font-weight: bold; 
                    box-shadow: 0 4px 15px rgba(0,0,0,0.3); z-index: 999;'>
        🧪 RESEARCH MODE ACTIVE
        </div>
        """, 
        unsafe_allow_html=True
    )