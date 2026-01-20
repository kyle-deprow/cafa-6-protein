"""Protein Language Model embeddings and similarity search.

This module provides:
- UniProt embedding downloader (pre-computed T5/ESM2 embeddings)
- FAISS index for fast similarity search
- Retrieval utilities for k-NN lookup
"""

from cafa_6_protein.embeddings.downloader import EmbeddingDownloader
from cafa_6_protein.embeddings.index import EmbeddingIndex

__all__ = ["EmbeddingDownloader", "EmbeddingIndex"]
