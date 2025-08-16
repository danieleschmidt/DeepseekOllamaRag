"""Core functionality for document processing and RAG operations."""

import os
import logging
import hashlib
import time
from typing import Optional, List, Dict, Any, Tuple
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from langchain_community.document_loaders import PDFPlumberLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.chains.llm import LLMChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains import RetrievalQA
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

from config import config
from exceptions import (
    DocumentProcessingError, EmbeddingGenerationError,
    VectorStoreError, LLMError, OllamaConnectionError,
    ModelNotFoundError
)
from utils import setup_logging, safe_execute, check_ollama_connection, get_available_models
from caching import global_cache, cached, cache_embedding_key, cache_qa_key
from resilience import with_resilience, OLLAMA_RETRY_CONFIG, OLLAMA_CIRCUIT_CONFIG
from logging_config import log_execution_time, log_document_processing, log_query_processing

logger = setup_logging()


class DocumentProcessor:
    """Handle document loading and processing with optimization."""
    
    def __init__(self):
        self.supported_formats = [".pdf"]
        self.chunk_cache = {}
        self.executor = ThreadPoolExecutor(max_workers=2)
    
    def load_document(self, file_path: str) -> List[Document]:
        """Load document from file path."""
        try:
            if not os.path.exists(file_path):
                raise DocumentProcessingError(f"File not found: {file_path}")
            
            file_extension = Path(file_path).suffix.lower()
            if file_extension not in self.supported_formats:
                raise DocumentProcessingError(
                    f"Unsupported file format: {file_extension}. "
                    f"Supported formats: {self.supported_formats}"
                )
            
            logger.info(f"Loading document: {file_path}")
            
            if file_extension == ".pdf":
                loader = PDFPlumberLoader(file_path)
                documents = loader.load()
                
                if not documents:
                    raise DocumentProcessingError("No content extracted from PDF")
                
                logger.info(f"Loaded {len(documents)} pages from PDF")
                return documents
            
        except Exception as e:
            logger.error(f"Error loading document: {str(e)}")
            raise DocumentProcessingError(f"Failed to load document: {str(e)}")
    
    @log_execution_time
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents into optimized chunks."""
        try:
            logger.info("Splitting documents into chunks with optimization")
            
            # Generate cache key based on document content
            content_hash = self._generate_content_hash([doc.page_content for doc in documents])
            cache_key = f"chunks:{content_hash}:{config.vector_store.chunk_size}:{config.vector_store.chunk_overlap}"
            
            # Check cache first
            cached_chunks = global_cache.get(cache_key)
            if cached_chunks:
                logger.info(f"Using cached chunks for document (hash: {content_hash[:8]})")
                return cached_chunks
            
            # Use hybrid approach: semantic + recursive for better performance
            chunks = self._hybrid_chunking(documents)
            
            # Cache the result
            global_cache.set(cache_key, chunks, ttl=7200)  # 2 hours
            
            logger.info(f"Created {len(chunks)} optimized chunks from {len(documents)} documents")
            return chunks
            
        except Exception as e:
            logger.error(f"Error splitting documents: {str(e)}")
            raise DocumentProcessingError(f"Failed to split documents: {str(e)}")
    
    def _hybrid_chunking(self, documents: List[Document]) -> List[Document]:
        """Hybrid chunking strategy for better performance."""
        # For large documents, use recursive chunking first, then semantic
        all_chunks = []
        
        for doc in documents:
            if len(doc.page_content) > 10000:  # Large document
                # First pass: recursive chunking
                recursive_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=config.vector_store.chunk_size * 2,  # Larger initial chunks
                    chunk_overlap=config.vector_store.chunk_overlap,
                    separators=["\n\n", "\n", ". ", " ", ""]
                )
                initial_chunks = recursive_splitter.split_documents([doc])
                
                # Second pass: semantic refinement on smaller chunks
                if len(initial_chunks) > 5:  # Only if we have many chunks
                    try:
                        embeddings = HuggingFaceEmbeddings(
                            model_name=config.model.embedding_model,
                            model_kwargs={"device": config.model.embedding_device}
                        )
                        semantic_splitter = SemanticChunker(embeddings)
                        final_chunks = semantic_splitter.split_documents(initial_chunks[:5])  # Limit for performance
                        final_chunks.extend(initial_chunks[5:])  # Add remaining as-is
                        all_chunks.extend(final_chunks)
                    except Exception:
                        # Fallback to recursive chunks if semantic fails
                        all_chunks.extend(initial_chunks)
                else:
                    all_chunks.extend(initial_chunks)
            else:
                # Small document: use recursive chunking only
                recursive_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=config.vector_store.chunk_size,
                    chunk_overlap=config.vector_store.chunk_overlap
                )
                chunks = recursive_splitter.split_documents([doc])
                all_chunks.extend(chunks)
        
        return all_chunks
    
    def _generate_content_hash(self, contents: List[str]) -> str:
        """Generate hash for document contents."""
        combined_content = "\n".join(contents)
        return hashlib.sha256(combined_content.encode()).hexdigest()


class EmbeddingManager:
    """Handle embedding generation and management with caching and optimization."""
    
    def __init__(self):
        self.embeddings = None
        self.embedding_cache = {}
        self.batch_size = 32  # Optimize for batch processing
        self._initialize_embeddings()
    
    def _initialize_embeddings(self):
        """Initialize embedding model."""
        try:
            logger.info(f"Initializing embedding model: {config.model.embedding_model}")
            self.embeddings = HuggingFaceEmbeddings(
                model_name=config.model.embedding_model,
                model_kwargs={"device": config.model.embedding_device}
            )
            logger.info("Embedding model initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing embeddings: {str(e)}")
            raise EmbeddingGenerationError(f"Failed to initialize embeddings: {str(e)}")
    
    @log_execution_time
    def create_vector_store(self, documents: List[Document]) -> FAISS:
        """Create optimized FAISS vector store from documents."""
        try:
            if not documents:
                raise VectorStoreError("No documents provided for vector store creation")
            
            logger.info(f"Creating optimized vector store from {len(documents)} documents")
            
            # Generate cache key for the entire document set
            doc_contents = [doc.page_content for doc in documents]
            content_hash = hashlib.sha256("\n".join(doc_contents).encode()).hexdigest()
            cache_key = f"vector_store:{content_hash}:{config.model.embedding_model}"
            
            # Check if vector store is cached
            cached_store = global_cache.get(cache_key)
            if cached_store:
                logger.info("Using cached vector store")
                return cached_store
            
            # Process documents in batches for better performance
            if len(documents) > self.batch_size:
                vector_store = self._create_vector_store_batched(documents)
            else:
                vector_store = FAISS.from_documents(documents, self.embeddings)
            
            # Cache the vector store (memory only due to size)
            global_cache.set(cache_key, vector_store, ttl=3600, memory_only=True)
            
            logger.info("Vector store created and cached successfully")
            return vector_store
            
        except Exception as e:
            logger.error(f"Error creating vector store: {str(e)}")
            raise VectorStoreError(f"Failed to create vector store: {str(e)}")
    
    def _create_vector_store_batched(self, documents: List[Document]) -> FAISS:
        """Create vector store using batch processing for large document sets."""
        logger.info(f"Creating vector store in batches (batch_size={self.batch_size})")
        
        # Create initial vector store with first batch
        first_batch = documents[:self.batch_size]
        vector_store = FAISS.from_documents(first_batch, self.embeddings)
        
        # Add remaining documents in batches
        for i in range(self.batch_size, len(documents), self.batch_size):
            batch = documents[i:i + self.batch_size]
            batch_vectors = FAISS.from_documents(batch, self.embeddings)
            vector_store.merge_from(batch_vectors)
            
            logger.debug(f"Processed batch {i//self.batch_size + 1}/{(len(documents) + self.batch_size - 1)//self.batch_size}")
        
        return vector_store
    
    def get_retriever(self, vector_store: FAISS):
        """Get retriever from vector store."""
        try:
            retriever = vector_store.as_retriever(
                search_type=config.vector_store.search_type,
                search_kwargs={"k": config.vector_store.similarity_search_k}
            )
            logger.info("Retriever created successfully")
            return retriever
            
        except Exception as e:
            logger.error(f"Error creating retriever: {str(e)}")
            raise VectorStoreError(f"Failed to create retriever: {str(e)}")


class LLMManager:
    """Handle LLM operations and chain creation with resilience."""
    
    def __init__(self):
        self.llm = None
        self.response_cache = {}
        self._initialize_llm()
    
    @with_resilience(
        retry_config=OLLAMA_RETRY_CONFIG,
        circuit_config=OLLAMA_CIRCUIT_CONFIG,
        service_name="ollama"
    )
    def _initialize_llm(self):
        """Initialize LLM connection with resilience."""
        try:
            # Check Ollama connection
            if not check_ollama_connection():
                raise OllamaConnectionError(
                    f"Cannot connect to Ollama at {config.model.ollama_base_url}. "
                    "Please ensure Ollama is running."
                )
            
            # Check if model is available
            available_models = get_available_models()
            if available_models and config.model.llm_model not in available_models:
                raise ModelNotFoundError(
                    f"Model '{config.model.llm_model}' not found. "
                    f"Available models: {available_models}"
                )
            
            logger.info(f"Initializing LLM: {config.model.llm_model}")
            self.llm = Ollama(
                model=config.model.llm_model,
                base_url=config.model.ollama_base_url,
                temperature=config.model.temperature
            )
            
            # Test the connection
            test_response = self.llm.invoke("Hello")
            logger.info("LLM initialized and tested successfully")
            
        except Exception as e:
            logger.error(f"Error initializing LLM: {str(e)}")
            raise LLMError(f"Failed to initialize LLM: {str(e)}")
    
    def create_qa_chain(self, retriever) -> RetrievalQA:
        """Create QA chain with retriever."""
        try:
            logger.info("Creating QA chain")
            
            # Define prompt template
            prompt_template = """
            Use the following pieces of context to answer the question at the end.
            If you don't know the answer, just say "I don't know" but don't make up an answer.
            Keep the answer crisp and limited to 3-4 sentences.
            
            Context: {context}
            Question: {question}
            
            Helpful Answer:"""
            
            QA_CHAIN_PROMPT = PromptTemplate.from_template(prompt_template)
            
            # Create LLM chain
            llm_chain = LLMChain(
                llm=self.llm, 
                prompt=QA_CHAIN_PROMPT, 
                verbose=config.app.debug
            )
            
            # Create document prompt
            document_prompt = PromptTemplate(
                input_variables=["page_content", "source"],
                template="Content: {page_content}\nSource: {source}",
            )
            
            # Create document combination chain
            combine_documents_chain = StuffDocumentsChain(
                llm_chain=llm_chain,
                document_variable_name="context",
                document_prompt=document_prompt,
                verbose=config.app.debug
            )
            
            # Create QA chain
            qa_chain = RetrievalQA(
                combine_documents_chain=combine_documents_chain,
                retriever=retriever,
                verbose=config.app.debug,
                return_source_documents=True
            )
            
            logger.info("QA chain created successfully")
            return qa_chain
            
        except Exception as e:
            logger.error(f"Error creating QA chain: {str(e)}")
            raise LLMError(f"Failed to create QA chain: {str(e)}")
    
    @log_execution_time
    @with_resilience(
        retry_config=OLLAMA_RETRY_CONFIG,
        circuit_config=OLLAMA_CIRCUIT_CONFIG,
        service_name="ollama"
    )
    def ask_question(self, qa_chain: RetrievalQA, question: str, doc_hash: str = "") -> Dict[str, Any]:
        """Ask question using QA chain with caching and resilience."""
        start_time = time.time()
        
        try:
            logger.info(f"Processing question: {question[:50]}...")
            
            # Check cache first
            cache_key = cache_qa_key(question, doc_hash, config.model.llm_model)
            cached_result = global_cache.get(cache_key)
            if cached_result:
                logger.info("Using cached Q&A result")
                log_query_processing(question, time.time() - start_time, True)
                return cached_result
            
            # Process question
            response = qa_chain({"query": question})
            
            result = {
                "answer": response.get("result", ""),
                "source_documents": response.get("source_documents", []),
                "question": question,
                "model": config.model.llm_model,
                "timestamp": time.time()
            }
            
            # Cache the result
            global_cache.set(cache_key, result, ttl=1800)  # 30 minutes
            
            processing_time = time.time() - start_time
            log_query_processing(question, processing_time, True)
            logger.info(f"Question processed successfully in {processing_time:.2f}s")
            return result
            
        except Exception as e:
            processing_time = time.time() - start_time
            log_query_processing(question, processing_time, False)
            logger.error(f"Error processing question: {str(e)}")
            raise LLMError(f"Failed to process question: {str(e)}")


class RAGPipeline:
    """Main RAG pipeline orchestrator."""
    
    def __init__(self):
        self.document_processor = DocumentProcessor()
        self.embedding_manager = EmbeddingManager()
        self.llm_manager = LLMManager()
    
    @log_execution_time
    def process_document(self, file_path: str) -> Tuple[FAISS, RetrievalQA, str]:
        """Process document and return vector store, QA chain, and document hash."""
        start_time = time.time()
        
        try:
            logger.info(f"Starting optimized document processing pipeline for: {file_path}")
            
            # Load and split document
            documents = self.document_processor.load_document(file_path)
            chunks = self.document_processor.split_documents(documents)
            
            # Generate document hash for caching
            doc_contents = [doc.page_content for doc in documents]
            doc_hash = hashlib.sha256("\n".join(doc_contents).encode()).hexdigest()
            
            # Create vector store
            vector_store = self.embedding_manager.create_vector_store(chunks)
            retriever = self.embedding_manager.get_retriever(vector_store)
            
            # Create QA chain
            qa_chain = self.llm_manager.create_qa_chain(retriever)
            
            processing_time = time.time() - start_time
            log_document_processing(file_path, len(chunks), processing_time)
            logger.info(f"Document processing pipeline completed successfully in {processing_time:.2f}s")
            
            return vector_store, qa_chain, doc_hash
            
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"Error in document processing pipeline after {processing_time:.2f}s: {str(e)}")
            raise
    
    def ask_question(self, qa_chain: RetrievalQA, question: str, doc_hash: str = "") -> Dict[str, Any]:
        """Ask question using the QA chain."""
        return self.llm_manager.ask_question(qa_chain, question, doc_hash)