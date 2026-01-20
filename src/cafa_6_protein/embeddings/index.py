"""FAISS index for fast protein similarity search.

Provides efficient k-NN lookup over protein embeddings.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class EmbeddingIndex:
    """FAISS index for protein embedding similarity search.

    Uses inner product similarity (equivalent to cosine on normalized vectors).

    Attributes:
        index: FAISS index.
        protein_ids: List of protein IDs in index order.
    """

    def __init__(self, embedding_dim: int = 1024) -> None:
        """Initialize an empty index.

        Args:
            embedding_dim: Dimension of embeddings.
        """
        self.embedding_dim = embedding_dim
        self._index: Any = None
        self._protein_ids: list[str] = []
        self._protein_to_idx: dict[str, int] = {}
        self._embeddings: np.ndarray | None = None

    def build(
        self,
        embeddings: np.ndarray,
        protein_ids: list[str],
        normalize: bool = True,
    ) -> None:
        """Build the index from embeddings.

        Args:
            embeddings: Array of shape (n_proteins, embedding_dim).
            protein_ids: List of protein IDs corresponding to rows.
            normalize: Whether to L2-normalize embeddings for cosine similarity.
        """
        try:
            import faiss
        except ImportError:
            logger.warning("FAISS not installed, using numpy fallback")
            self._build_numpy(embeddings, protein_ids, normalize)
            return

        if len(embeddings) != len(protein_ids):
            raise ValueError(
                f"Embeddings ({len(embeddings)}) and protein_ids ({len(protein_ids)}) "
                "must have same length"
            )

        self._protein_ids = list(protein_ids)
        self._protein_to_idx = {pid: i for i, pid in enumerate(protein_ids)}

        # Prepare embeddings
        emb = embeddings.astype(np.float32)
        if normalize:
            # L2 normalize for cosine similarity via inner product
            norms = np.linalg.norm(emb, axis=1, keepdims=True)
            norms[norms == 0] = 1  # Avoid division by zero
            emb = emb / norms

        self._embeddings = emb

        # Build FAISS index
        # Using IndexFlatIP for exact inner product search
        # For >100k proteins, consider IVF or HNSW
        self._index = faiss.IndexFlatIP(self.embedding_dim)
        self._index.add(emb)

        logger.info(f"Built FAISS index with {len(protein_ids)} proteins")

    def _build_numpy(
        self,
        embeddings: np.ndarray,
        protein_ids: list[str],
        normalize: bool = True,
    ) -> None:
        """Numpy fallback when FAISS is not available."""
        self._protein_ids = list(protein_ids)
        self._protein_to_idx = {pid: i for i, pid in enumerate(protein_ids)}

        emb = embeddings.astype(np.float32)
        if normalize:
            norms = np.linalg.norm(emb, axis=1, keepdims=True)
            norms[norms == 0] = 1
            emb = emb / norms

        self._embeddings = emb
        self._index = None  # Use numpy for search

        logger.info(f"Built numpy index with {len(protein_ids)} proteins (FAISS fallback)")

    def search(
        self,
        query: np.ndarray,
        k: int = 50,
        normalize: bool = True,
    ) -> tuple[np.ndarray, np.ndarray, list[list[str]]]:
        """Search for k nearest neighbors.

        Args:
            query: Query embedding of shape (embedding_dim,) or (n_queries, embedding_dim).
            k: Number of neighbors to return.
            normalize: Whether to normalize query vectors.

        Returns:
            Tuple of (distances, indices, protein_ids).
            - distances: Shape (n_queries, k) of similarity scores.
            - indices: Shape (n_queries, k) of neighbor indices.
            - protein_ids: Nested list of protein IDs for each query.
        """
        # Handle single query
        if query.ndim == 1:
            query = query.reshape(1, -1)

        query = query.astype(np.float32)

        if normalize:
            norms = np.linalg.norm(query, axis=1, keepdims=True)
            norms[norms == 0] = 1
            query = query / norms

        k = min(k, len(self._protein_ids))

        if self._index is not None:
            # FAISS search
            distances, indices = self._index.search(query, k)
        else:
            # Numpy fallback
            distances, indices = self._search_numpy(query, k)

        # Convert indices to protein IDs
        protein_id_results: list[list[str]] = [
            [self._protein_ids[idx] for idx in row if idx >= 0] for row in indices
        ]

        return distances, indices, protein_id_results

    def _search_numpy(self, query: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        """Numpy-based k-NN search fallback.

        Args:
            query: Normalized query vectors.
            k: Number of neighbors.

        Returns:
            Tuple of (distances, indices).
        """
        if self._embeddings is None:
            raise ValueError("Index not built")

        # Compute all similarities
        similarities = query @ self._embeddings.T  # (n_queries, n_proteins)

        # Get top-k for each query
        indices = np.argpartition(-similarities, k, axis=1)[:, :k]

        # Sort top-k by similarity
        row_indices = np.arange(len(query))[:, None]
        top_k_sims = similarities[row_indices, indices]
        sorted_order = np.argsort(-top_k_sims, axis=1)
        indices = indices[row_indices, sorted_order]
        distances = top_k_sims[row_indices, sorted_order]

        return distances, indices

    def search_by_protein(
        self,
        protein_id: str,
        k: int = 50,
        exclude_self: bool = True,
    ) -> list[tuple[str, float]]:
        """Search for similar proteins given a protein ID.

        Args:
            protein_id: Query protein ID (must be in index).
            k: Number of neighbors to return.
            exclude_self: Whether to exclude the query protein from results.

        Returns:
            List of (protein_id, similarity) tuples.
        """
        if protein_id not in self._protein_to_idx:
            raise ValueError(f"Protein {protein_id} not in index")
        if self._embeddings is None:
            raise ValueError("Index not built")

        idx = self._protein_to_idx[protein_id]
        query = self._embeddings[idx : idx + 1]

        # Search for k+1 if excluding self
        search_k = k + 1 if exclude_self else k
        distances, _, protein_ids = self.search(query, k=search_k, normalize=False)

        results = list(zip(protein_ids[0], distances[0], strict=False))

        if exclude_self:
            results = [(pid, sim) for pid, sim in results if pid != protein_id][:k]

        return results

    def save(self, path: Path | str) -> None:
        """Save index to disk.

        Args:
            path: Directory to save index files.
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save embeddings
        if self._embeddings is not None:
            np.save(path / "index_embeddings.npy", self._embeddings)

        # Save protein IDs
        with (path / "index_protein_ids.txt").open("w") as f:
            f.write("\n".join(self._protein_ids))

        # Save FAISS index if available
        if self._index is not None:
            try:
                import faiss

                faiss.write_index(self._index, str(path / "index.faiss"))
            except ImportError:
                pass

        logger.info(f"Saved index to {path}")

    @classmethod
    def load(cls, path: Path | str) -> EmbeddingIndex:
        """Load index from disk.

        Args:
            path: Directory containing index files.

        Returns:
            Loaded EmbeddingIndex.
        """
        path = Path(path)

        # Load embeddings
        embeddings = np.load(path / "index_embeddings.npy")

        # Load protein IDs
        with (path / "index_protein_ids.txt").open() as f:
            protein_ids = [line.strip() for line in f if line.strip()]

        # Create index
        index = cls(embedding_dim=embeddings.shape[1])
        index._embeddings = embeddings
        index._protein_ids = protein_ids
        index._protein_to_idx = {pid: i for i, pid in enumerate(protein_ids)}

        # Try to load FAISS index
        faiss_path = path / "index.faiss"
        if faiss_path.exists():
            try:
                import faiss

                index._index = faiss.read_index(str(faiss_path))
                logger.info(f"Loaded FAISS index from {faiss_path}")
            except ImportError:
                logger.warning("FAISS not installed, using numpy fallback")

        logger.info(f"Loaded index with {len(protein_ids)} proteins")
        return index

    def __len__(self) -> int:
        """Return number of proteins in index."""
        return len(self._protein_ids)

    def __contains__(self, protein_id: str) -> bool:
        """Check if protein is in index."""
        return protein_id in self._protein_to_idx
