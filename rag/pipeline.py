"""
RAGPipeline: orchestrates retrieve → augment prompt → generate.

Works with real Ollama or the mock backend.
"""

import logging
from typing import Dict, List, Optional

from .chunk_splitter import ChunkSplitter
from .document_store import DocumentStore
from .ollama_client import OllamaClient
from .retriever import Retriever

logger = logging.getLogger(__name__)


_SYSTEM_PROMPT = (
    "You are a helpful assistant. Answer questions using ONLY the provided context. "
    "If the context does not contain enough information, say so honestly. "
    "Keep answers concise and factual."
)

_PROMPT_TEMPLATE = """\
Context:
{context}

Question: {question}

Answer:"""


class RAGPipeline:
    """
    End-to-end Retrieval-Augmented Generation pipeline.

    Workflow
    --------
    1. ``index(documents)`` — chunk, embed, store
    2. ``query(question)``  — retrieve → augment → generate → return answer

    Parameters
    ----------
    ollama_client : OllamaClient
        Client for the LLM backend (real or mock).
    chunk_size : int
        Chunk size used by ChunkSplitter (default 500 chars).
    overlap : int
        Chunk overlap (default 100 chars).
    top_k : int
        Number of chunks to retrieve per query (default 3).
    """

    def __init__(
        self,
        ollama_client: Optional[OllamaClient] = None,
        chunk_size: int = 500,
        overlap: int = 100,
        top_k: int = 3,
    ) -> None:
        self.client = ollama_client or OllamaClient()
        self.splitter = ChunkSplitter(chunk_size=chunk_size, overlap=overlap)
        self.store = DocumentStore()
        self.retriever = Retriever(self.store, top_k=top_k)
        self._doc_count = 0

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    def index(self, documents: List[str], source_labels: Optional[List[str]] = None) -> int:
        """
        Index a list of document strings into the store.

        Parameters
        ----------
        documents : list[str]
            Raw document texts.
        source_labels : list[str], optional
            Human-readable labels (e.g. filenames) for logging.

        Returns
        -------
        int
            Total number of chunks added to the store.
        """
        labels = source_labels or [f"doc_{i}" for i in range(len(documents))]
        all_chunks: List[str] = []
        for label, text in zip(labels, documents):
            chunks = self.splitter.split(text)
            logger.info("Indexed '%s' → %d chunks", label, len(chunks))
            all_chunks.extend(chunks)

        self.store.add_documents(all_chunks)
        self._doc_count += len(documents)
        logger.info(
            "Index now contains %d chunks from %d documents",
            self.store.chunk_count,
            self._doc_count,
        )
        return len(all_chunks)

    # ------------------------------------------------------------------
    # Querying
    # ------------------------------------------------------------------

    def query(self, question: str) -> Dict[str, object]:
        """
        Retrieve relevant chunks then generate an answer.

        Returns
        -------
        dict with keys:
            question  – original question
            context   – list of (chunk, score) tuples used
            answer    – LLM-generated answer
            is_mock   – whether the mock backend was used
        """
        # 1. Retrieve
        hits = self.retriever.retrieve(question)
        context_texts = [chunk for chunk, _ in hits]

        # 2. Build augmented prompt
        context_block = "\n\n---\n\n".join(context_texts) if context_texts else "(no context found)"
        prompt = _PROMPT_TEMPLATE.format(context=context_block, question=question)

        # 3. Generate
        answer = self.client.generate(prompt, system=_SYSTEM_PROMPT)

        return {
            "question": question,
            "context": hits,           # [(chunk, score), ...]
            "answer": answer,
            "is_mock": self.client.is_mock,
        }

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    @property
    def chunk_count(self) -> int:
        return self.store.chunk_count
