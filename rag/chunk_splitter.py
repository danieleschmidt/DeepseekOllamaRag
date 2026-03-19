"""Split documents into overlapping text chunks."""

from typing import List


class ChunkSplitter:
    """
    Split a document (string) into overlapping fixed-size chunks.

    Parameters
    ----------
    chunk_size : int
        Maximum number of characters per chunk (default 500).
    overlap : int
        Number of characters shared between consecutive chunks (default 100).
    """

    def __init__(self, chunk_size: int = 500, overlap: int = 100) -> None:
        if overlap >= chunk_size:
            raise ValueError("overlap must be smaller than chunk_size")
        self.chunk_size = chunk_size
        self.overlap = overlap

    def split(self, text: str) -> List[str]:
        """Return a list of overlapping chunks from *text*."""
        text = text.strip()
        if not text:
            return []

        chunks: List[str] = []
        step = self.chunk_size - self.overlap
        start = 0

        while start < len(text):
            end = start + self.chunk_size
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            if end >= len(text):
                break
            start += step

        return chunks

    def split_all(self, texts: List[str]) -> List[str]:
        """Split every text in a list and return a flat list of chunks."""
        result: List[str] = []
        for text in texts:
            result.extend(self.split(text))
        return result
