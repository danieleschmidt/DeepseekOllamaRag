import streamlit as st
import logging
from datetime import datetime
from typing import Optional

from config import config
from core import RAGPipeline
from utils import (
    init_session_state, check_session_timeout, reset_session,
    save_uploaded_file, validate_file, cleanup_temp_files,
    format_file_size, get_file_info, check_ollama_connection
)
from exceptions import (
    DeepSeekRAGException, DocumentProcessingError,
    LLMError, ValidationError, OllamaConnectionError
)

# Configure Streamlit page
st.set_page_config(
    page_title=config.ui.page_title,
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state and logging
init_session_state()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Apply custom styling
st.markdown(f"""
    <style>
    /* Main Background */
    .stApp {{
        background-color: {config.ui.background_color};
        color: #212529;
    }}

    /* Sidebar Styling */
    [data-testid="stSidebar"] {{
        background-color: {config.ui.sidebar_background} !important;
        color: #FFFFFF !important;
    }}
    [data-testid="stSidebar"] * {{
        color: #FFFFFF !important;
        font-size: 16px !important;
    }}

    /* Headings */
    h1, h2, h3, h4, h5, h6 {{
        color: #000000 !important;
        font-weight: bold;
    }}

    /* Fix Text Visibility */
    p, span, div {{
        color: #212529 !important;
    }}

    /* File Uploader */
    .stFileUploader>div>div>div>button {{
        background-color: {config.ui.secondary_color};
        color: #000000;
        font-weight: bold;
        border-radius: 8px;
    }}

    /* Primary Button */
    .stButton>button {{
        background-color: {config.ui.primary_color};
        color: white;
        border-radius: 8px;
    }}

    /* Error/Success Messages */
    .stAlert {{
        border-radius: 8px;
    }}

    /* Progress Bar */
    .stProgress > div > div {{
        background-color: {config.ui.primary_color};
    }}
    </style>
""", unsafe_allow_html=True)

# Check session timeout
if check_session_timeout():
    st.warning("Your session has expired. Please refresh the page.")
    if st.button("Reset Session"):
        reset_session()
        st.rerun()

# Check Ollama connection
if not check_ollama_connection():
    st.error("⚠️ Cannot connect to Ollama. Please ensure Ollama is running.")
    st.info(f"Expected Ollama URL: {config.model.ollama_base_url}")
    st.stop()

# App title
st.title(config.ui.app_title)

# Clean up old temp files
cleanup_temp_files()

# Initialize RAG pipeline
if "rag_pipeline" not in st.session_state:
    try:
        st.session_state.rag_pipeline = RAGPipeline()
    except Exception as e:
        st.error(f"Failed to initialize RAG pipeline: {str(e)}")
        st.stop()

# Sidebar for instructions and settings
with st.sidebar:
    st.header("📋 Instructions")
    st.markdown("""
    1. **Upload** a PDF file using the uploader below
    2. **Wait** for processing to complete
    3. **Ask** questions related to the document content
    4. **Review** answers with source references
    """)

    st.header("⚙️ System Status")
    
    # Ollama connection status
    if check_ollama_connection():
        st.success("✅ Ollama Connected")
    else:
        st.error("❌ Ollama Disconnected")
    
    # Current configuration
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
        st.header("📁 Upload History")
        for i, upload in enumerate(st.session_state.upload_history[-3:]):
            st.markdown(f"**{i+1}.** {upload['name']} ({upload['size']})")
    
    # Reset button
    if st.button("🔄 Reset Session", key="reset_sidebar"):
        reset_session()
        st.rerun()

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    st.header("📁 Upload Document")
    
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
                st.session_state.vector_store = st.session_state.processed_documents[file_hash]['vector_store']
                st.session_state.qa_chain = st.session_state.processed_documents[file_hash]['qa_chain']
            else:
                # Process new document
                with st.spinner("🔄 Processing document..."):
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    try:
                        # Save uploaded file
                        status_text.text("💾 Saving file...")
                        progress_bar.progress(20)
                        file_path = save_uploaded_file(uploaded_file)
                        
                        # Process document
                        status_text.text("📖 Processing document...")
                        progress_bar.progress(60)
                        vector_store, qa_chain = st.session_state.rag_pipeline.process_document(file_path)
                        
                        # Store in session
                        status_text.text("💾 Storing results...")
                        progress_bar.progress(90)
                        
                        st.session_state.vector_store = vector_store
                        st.session_state.qa_chain = qa_chain
                        st.session_state.processed_documents[file_hash] = {
                            'vector_store': vector_store,
                            'qa_chain': qa_chain,
                            'file_name': uploaded_file.name,
                            'processed_at': datetime.now()
                        }
                        
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
                        
                        st.success("🎉 Document processed successfully! You can now ask questions.")
                        
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
        st.header("📊 Document Stats")
        
        # Display document statistics
        try:
            # Get vector store stats
            if hasattr(st.session_state.vector_store, 'index'):
                vector_count = st.session_state.vector_store.index.ntotal
                st.metric("Document Chunks", vector_count)
            
            st.metric("Model", config.model.llm_model.split(':')[0])
            st.metric("Search Results", config.vector_store.similarity_search_k)
            
        except Exception as e:
            st.warning("Could not load document statistics")

# Question and Answer Section
if st.session_state.qa_chain is not None:
    st.header("❓ Ask Questions")
    
    # Question input
    user_question = st.text_input(
        "Enter your question about the document:",
        placeholder="What is the main topic of this document?"
    )
    
    col1, col2, col3 = st.columns([1, 1, 2])
    
    with col1:
        ask_button = st.button("🔍 Ask Question", type="primary")
    
    with col2:
        if st.button("🗑️ Clear History"):
            if "qa_history" in st.session_state:
                st.session_state.qa_history = []
                st.rerun()
    
    # Process question
    if ask_button and user_question.strip():
        with st.spinner("🤔 Thinking..."):
            try:
                response = st.session_state.rag_pipeline.ask_question(
                    st.session_state.qa_chain,
                    user_question
                )
                
                # Store in history
                if "qa_history" not in st.session_state:
                    st.session_state.qa_history = []
                
                st.session_state.qa_history.append({
                    'question': user_question,
                    'answer': response['answer'],
                    'sources': response['source_documents'],
                    'timestamp': datetime.now()
                })
                
                st.success("✅ Question processed successfully!")
                
            except Exception as e:
                if isinstance(e, DeepSeekRAGException):
                    st.error(f"❌ {str(e)}")
                else:
                    st.error(f"❌ Unexpected error: {str(e)}")
                    logger.error(f"Question processing error: {str(e)}", exc_info=True)
    
    # Display Q&A History
    if "qa_history" in st.session_state and st.session_state.qa_history:
        st.header("💬 Q&A History")
        
        for i, qa in enumerate(reversed(st.session_state.qa_history[-5:])):
            with st.expander(f"Q{len(st.session_state.qa_history)-i}: {qa['question'][:50]}...", expanded=(i==0)):
                st.markdown(f"**Question:** {qa['question']}")
                st.markdown(f"**Answer:** {qa['answer']}")
                
                if qa['sources']:
                    st.markdown("**Sources:**")
                    for j, source in enumerate(qa['sources'][:2]):
                        content_preview = source.page_content[:200] + "..." if len(source.page_content) > 200 else source.page_content
                        st.markdown(f"*Source {j+1}:* {content_preview}")
                
                st.markdown(f"*Asked at: {qa['timestamp'].strftime('%H:%M:%S')}*")

else:
    st.info("👆 Please upload a PDF document to get started.")


