"""Tests for ChunkSplitter."""

import pytest
from rag.chunk_splitter import ChunkSplitter


class TestChunkSplitter:
    def test_single_chunk_small_text(self):
        cs = ChunkSplitter(chunk_size=200, overlap=50)
        result = cs.split("hello world")
        assert result == ["hello world"]

    def test_empty_string(self):
        cs = ChunkSplitter()
        assert cs.split("") == []

    def test_whitespace_only(self):
        cs = ChunkSplitter()
        assert cs.split("   \n  ") == []

    def test_overlap_creates_multiple_chunks(self):
        text = "a" * 1000
        cs = ChunkSplitter(chunk_size=300, overlap=100)
        chunks = cs.split(text)
        assert len(chunks) > 1

    def test_chunk_size_respected(self):
        text = "x" * 2000
        cs = ChunkSplitter(chunk_size=500, overlap=100)
        for chunk in cs.split(text):
            assert len(chunk) <= 500

    def test_overlap_content(self):
        # With enough overlap, adjacent chunks should share content
        text = "ABCDEFGHIJ" * 10  # 100 chars
        cs = ChunkSplitter(chunk_size=40, overlap=20)
        chunks = cs.split(text)
        assert len(chunks) >= 2
        # The end of chunk[0] should appear at the start of chunk[1]
        assert chunks[0][-10:] in chunks[1]

    def test_split_all(self):
        cs = ChunkSplitter(chunk_size=200, overlap=50)
        result = cs.split_all(["short text", "another short text"])
        assert len(result) == 2

    def test_invalid_overlap_raises(self):
        with pytest.raises(ValueError):
            ChunkSplitter(chunk_size=100, overlap=100)
