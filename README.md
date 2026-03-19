# DeepseekOllamaRag

A minimal, dependency-light **Retrieval-Augmented Generation (RAG)** system that runs entirely on your machine using [Ollama](https://ollama.com) as the LLM backend.

No cloud APIs. No heavyweight ML frameworks needed. Works with DeepSeek, Llama, Phi, Mistral, Gemma — any model Ollama can run.

---

## What is RAG?

RAG grounds an LLM's answers in retrieved documents rather than relying solely on its training data:

```
Documents → chunk → embed → index
Query → retrieve top-k chunks → augment prompt → LLM → answer
```

This reduces hallucinations and makes it easy to add domain-specific knowledge without fine-tuning.

---

## Architecture

```
rag/
├── chunk_splitter.py   ChunkSplitter   — overlapping fixed-size chunks
├── document_store.py   DocumentStore   — TF-IDF index (numpy, stdlib only)
├── retriever.py        Retriever       — query → top-k relevant chunks
├── ollama_client.py    OllamaClient    — thin HTTP wrapper for Ollama API
│                                         (auto-falls back to mock)
└── pipeline.py         RAGPipeline     — orchestrates the full workflow
```

### Dependencies

| Package  | Used for                          |
|----------|-----------------------------------|
| `numpy`  | TF-IDF matrix + cosine similarity |
| `requests` | HTTP calls to Ollama            |

Everything else is Python stdlib.

---

## Quick Start

### 1. Install Ollama and pull a model

```bash
# Install: https://ollama.com/download
ollama pull deepseek-r1   # or llama3.2, phi3, mistral, etc.
```

### 2. Clone and run the demo

```bash
git clone https://github.com/danieleschmidt/DeepseekOllamaRag
cd DeepseekOllamaRag

python demo.py --model deepseek-r1
```

If Ollama isn't running, the demo automatically switches to a mock backend so you can still see the retrieval logic in action.

### 3. Use in your own code

```python
from rag import OllamaClient, RAGPipeline

client = OllamaClient(model="deepseek-r1")  # auto-mocks if Ollama is down
pipeline = RAGPipeline(ollama_client=client, chunk_size=500, overlap=100, top_k=3)

# Index your documents
pipeline.index([
    "Python is a high-level programming language...",
    "RAG combines retrieval with generation...",
    # add as many as you like
])

# Query
result = pipeline.query("What is Python?")
print(result["answer"])
print(result["context"])   # list of (chunk, score) tuples
```

---

## Component API

### `ChunkSplitter`

```python
from rag import ChunkSplitter

cs = ChunkSplitter(chunk_size=500, overlap=100)
chunks = cs.split(long_text)          # List[str]
chunks = cs.split_all([doc1, doc2])   # flat List[str]
```

### `DocumentStore`

```python
from rag import DocumentStore

store = DocumentStore()
store.add_documents(chunks)                        # add List[str]
results = store.search("my query", top_k=3)        # List[(chunk, score)]
```

### `Retriever`

```python
from rag import Retriever

retriever = Retriever(store, top_k=3)
hits   = retriever.retrieve("query")       # List[(chunk, score)]
texts  = retriever.retrieve_text("query")  # List[str]
```

### `OllamaClient`

```python
from rag import OllamaClient

client = OllamaClient(
    base_url="http://localhost:11434",
    model="deepseek-r1",
    timeout=120,
    force_mock=False,   # set True to skip Ollama for testing
)

answer = client.generate("What is Python?", system="Be concise.")
vector = client.embed("some text")
models = client.list_models()
```

### `RAGPipeline`

```python
from rag import RAGPipeline

p = RAGPipeline(chunk_size=500, overlap=100, top_k=3)
p.index(documents, source_labels=["file1.txt", "file2.txt"])

result = p.query("What is RAG?")
# result = {
#   "question": "...",
#   "context":  [(chunk, score), ...],
#   "answer":   "...",
#   "is_mock":  False,
# }
```

---

## Running Tests

```bash
# From repo root
python -m pytest tests/ -v
```

All tests use the mock backend — no Ollama required.

---

## Supported Models (via Ollama)

| Model          | Pull command                  |
|----------------|-------------------------------|
| DeepSeek-R1    | `ollama pull deepseek-r1`     |
| Llama 3.2      | `ollama pull llama3.2`        |
| Phi-3          | `ollama pull phi3`            |
| Mistral        | `ollama pull mistral`         |
| Gemma 2        | `ollama pull gemma2`          |

---

## License

MIT — see [LICENSE](LICENSE).
