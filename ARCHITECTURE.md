# Architecture Documentation

## System Overview

DeepseekOllamaRag is a Retrieval Augmented Generation (RAG) system that combines document processing, vector search, and large language model capabilities to provide intelligent document-based question answering.

## Architecture Diagram

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Streamlit UI  │    │  Document       │    │   Vector Store  │
│                 │───▶│  Processing     │───▶│   (FAISS)       │
│ - File Upload   │    │                 │    │                 │
│ - Q&A Interface │    │ - PDF Loader    │    │ - Embeddings    │
└─────────────────┘    │ - Text Splitter │    │ - Similarity    │
         │              │ - Chunking      │    │   Search        │
         │              └─────────────────┘    └─────────────────┘
         │                                              │
         ▼                                              │
┌─────────────────┐    ┌─────────────────┐             │
│  Query Engine   │    │   DeepSeek R1   │             │
│                 │───▶│   via Ollama    │◀────────────┘
│ - Context       │    │                 │
│   Retrieval     │    │ - Local LLM     │
│ - Prompt        │    │ - Answer        │
│   Template      │    │   Generation    │
└─────────────────┘    └─────────────────┘
```

## Data Flow

1. **Document Ingestion**: PDF files are uploaded through Streamlit interface
2. **Text Processing**: PDFPlumberLoader extracts text, SemanticChunker splits into meaningful segments
3. **Vectorization**: HuggingFace embeddings convert text chunks to vector representations
4. **Storage**: FAISS vector store indexes embeddings for efficient similarity search
5. **Query Processing**: User questions are embedded and matched against document vectors
6. **Context Retrieval**: Top-k similar chunks are retrieved as context
7. **Answer Generation**: DeepSeek R1 model generates answers using retrieved context

## Component Architecture

### Frontend Layer
- **Streamlit Application** (`app.py`)
  - File upload interface
  - Question input form
  - Response display
  - Custom CSS styling

### Processing Layer
- **Document Loader**: PDFPlumberLoader for robust PDF text extraction
- **Text Splitter**: SemanticChunker for intelligent document segmentation
- **Embedding Engine**: HuggingFace sentence transformers

### Storage Layer
- **Vector Database**: FAISS for efficient similarity search
- **Temporary Storage**: Local file system for uploaded PDFs

### AI Layer
- **LLM Backend**: DeepSeek R1 (1.5B parameters) via Ollama
- **Prompt Engineering**: Structured templates for consistent responses
- **Chain Management**: LangChain for workflow orchestration

## Technology Stack

### Core Dependencies
- **Streamlit**: Web application framework
- **LangChain**: LLM application framework
- **FAISS**: Vector similarity search
- **HuggingFace Transformers**: Embedding models
- **Ollama**: Local LLM inference
- **PDFPlumber**: PDF text extraction

### Model Architecture
- **Embedding Model**: HuggingFace sentence-transformers
- **LLM**: DeepSeek R1:1.5b
- **Vector Dimensions**: 384 (default for sentence-transformers)
- **Retrieval Strategy**: Similarity search with k=3

## Design Decisions

### ADR-001: Technology Selection
- **Decision**: Use DeepSeek R1 via Ollama for local inference
- **Rationale**: Privacy, cost efficiency, offline capability
- **Alternatives**: OpenAI API, Azure OpenAI, Anthropic Claude

### ADR-002: Vector Store Selection
- **Decision**: FAISS for vector similarity search
- **Rationale**: Performance, local deployment, no external dependencies
- **Alternatives**: Pinecone, Chroma, Weaviate

### ADR-003: Chunking Strategy
- **Decision**: SemanticChunker for intelligent text segmentation
- **Rationale**: Context-aware splitting preserves semantic meaning
- **Alternatives**: Fixed-size chunking, sentence-based splitting

## Security Considerations

- Local file processing (no external data transmission)
- Temporary file cleanup after processing
- No persistent user data storage
- Local LLM inference (no API key exposure)

## Performance Characteristics

- **Document Processing**: ~2-5 seconds for typical PDFs
- **Embedding Generation**: ~1-3 seconds for document chunks
- **Query Response**: ~3-8 seconds including retrieval and generation
- **Memory Usage**: ~2-4GB during processing (model + embeddings)

## Scalability Considerations

### Current Limitations
- Single document processing
- In-memory vector storage
- Single-user interface
- No persistent sessions

### Future Enhancements
- Multi-document collections
- Persistent vector database
- User session management
- Batch processing capabilities