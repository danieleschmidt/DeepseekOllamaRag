#!/usr/bin/env python3
"""
demo.py — DeepseekOllamaRag demo

Indexes 5 documents and queries 3 questions.
Uses real Ollama if available; otherwise falls back to mock.

Usage:
    python demo.py [--model MODEL] [--base-url URL]
"""

import argparse
import textwrap

from rag import OllamaClient, RAGPipeline


DOCUMENTS = [
    """
    Python is a high-level, dynamically-typed programming language known for its readable syntax
    and broad standard library. Created by Guido van Rossum and first released in 1991, Python
    emphasises code readability and expressiveness. It supports multiple programming paradigms
    including procedural, object-oriented, and functional styles.
    """,
    """
    Retrieval-Augmented Generation (RAG) is an AI architecture that combines a retrieval component
    (which fetches relevant documents from a corpus) with a generative language model. Rather than
    relying solely on the knowledge baked into model weights, RAG systems ground their answers in
    retrieved evidence, reducing hallucinations and keeping information up to date.
    """,
    """
    Ollama is an open-source tool for running large language models locally on consumer hardware.
    It packages model weights together with a runtime and exposes a simple REST API. Models such as
    Llama 3, Mistral, DeepSeek, Phi-3, and Gemma can be pulled and run with a single command.
    Ollama is available for macOS, Linux, and Windows.
    """,
    """
    DeepSeek is a series of large language models developed by DeepSeek AI, a Chinese AI research
    company. The DeepSeek-R1 model uses a chain-of-thought reasoning approach and achieves
    competitive benchmark scores against much larger models. DeepSeek models are open-weight and
    can be run locally through tools like Ollama.
    """,
    """
    Vector databases store high-dimensional numerical embeddings and support approximate nearest-
    neighbour (ANN) search. They are a core component of RAG pipelines: text chunks are encoded as
    embedding vectors and the most semantically similar chunks to a query are retrieved in
    milliseconds. Popular vector databases include Chroma, Qdrant, Weaviate, and FAISS (an
    in-memory index from Meta).
    """,
]

QUESTIONS = [
    "What is Retrieval-Augmented Generation and why is it useful?",
    "How does Ollama work and what models does it support?",
    "Who created DeepSeek and what makes it interesting?",
]


def _banner(text: str, width: int = 70) -> None:
    print("\n" + "=" * width)
    print(text.center(width))
    print("=" * width)


def main() -> None:
    parser = argparse.ArgumentParser(description="DeepseekOllamaRag demo")
    parser.add_argument("--model", default="deepseek-r1", help="Ollama model name")
    parser.add_argument("--base-url", default="http://localhost:11434", help="Ollama base URL")
    args = parser.parse_args()

    _banner("DeepseekOllamaRag — Local LLM RAG Demo")

    # Build pipeline
    client = OllamaClient(base_url=args.base_url, model=args.model)
    pipeline = RAGPipeline(ollama_client=client, chunk_size=400, overlap=80, top_k=3)

    backend_label = "MOCK" if client.is_mock else f"Ollama ({args.model})"
    print(f"\nBackend : {backend_label}")

    # Index documents
    print("\n[1/3] Indexing documents …")
    labels = [f"doc_{i+1}" for i in range(len(DOCUMENTS))]
    total_chunks = pipeline.index(DOCUMENTS, source_labels=labels)
    print(f"  ✓  {len(DOCUMENTS)} documents indexed → {total_chunks} chunks")

    # Query
    print("\n[2/3] Running queries …")
    for i, question in enumerate(QUESTIONS, 1):
        print(f"\n{'─'*68}")
        print(f"Q{i}: {question}")
        result = pipeline.query(question)

        print("\n  Retrieved context:")
        for j, (chunk, score) in enumerate(result["context"], 1):
            preview = textwrap.shorten(chunk, width=80, placeholder="…")
            print(f"    [{j}] (score={score:.3f}) {preview}")

        print("\n  Answer:")
        wrapped = textwrap.fill(result["answer"], width=66, initial_indent="    ", subsequent_indent="    ")
        print(wrapped)

    _banner("Done")


if __name__ == "__main__":
    main()
