"""Download pre-computed protein embeddings from UniProt.

UniProt provides per-protein T5 embeddings for SwissProt proteins.
We download the bulk H5 file and extract embeddings for our proteins.

FTP: https://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/embeddings/
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import h5py
import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable

logger = logging.getLogger(__name__)

# UniProt Embeddings FTP
UNIPROT_EMBEDDINGS_URL = (
    "https://ftp.uniprot.org/pub/databases/uniprot/current_release/"
    "knowledgebase/embeddings/uniprot_sprot/per-protein.h5"
)

# T5 embedding dimension
T5_EMBEDDING_DIM = 1024


class EmbeddingDownloader:
    """Download and manage protein embeddings from UniProt.

    Downloads the bulk SwissProt embeddings file and extracts vectors
    for specific proteins.

    Attributes:
        cache_dir: Directory to store embeddings.
        embedding_dim: Dimension of output embeddings.
    """

    def __init__(self, cache_dir: Path | str) -> None:
        """Initialize the embedding downloader.

        Args:
            cache_dir: Directory for embedding cache.
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.embedding_dim = T5_EMBEDDING_DIM

        # Cache file paths
        self._embeddings_file = self.cache_dir / "embeddings.npy"
        self._protein_ids_file = self.cache_dir / "embedding_protein_ids.txt"
        self._bulk_h5_file = self.cache_dir / "uniprot_sprot_embeddings.h5"

        # In-memory cache
        self._embeddings: np.ndarray | None = None
        self._protein_ids: list[str] = []
        self._protein_to_idx: dict[str, int] = {}

        self._load_cache()

    def _load_cache(self) -> None:
        """Load cached embeddings from disk."""
        if self._embeddings_file.exists() and self._protein_ids_file.exists():
            self._embeddings = np.load(self._embeddings_file)
            with self._protein_ids_file.open() as f:
                self._protein_ids = [line.strip() for line in f if line.strip()]
            self._protein_to_idx = {pid: i for i, pid in enumerate(self._protein_ids)}
            if self._embeddings is not None:
                logger.info(
                    f"Loaded {len(self._protein_ids)} embeddings from cache "
                    f"(shape: {self._embeddings.shape})"
                )

    def save(self) -> None:
        """Save embeddings to disk."""
        if self._embeddings is not None and len(self._protein_ids) > 0:
            np.save(self._embeddings_file, self._embeddings)
            with self._protein_ids_file.open("w") as f:
                f.write("\n".join(self._protein_ids))
            logger.info(f"Saved {len(self._protein_ids)} embeddings to cache")

    def has_embedding(self, protein_id: str) -> bool:
        """Check if an embedding is cached.

        Args:
            protein_id: UniProt accession.

        Returns:
            True if embedding exists in cache.
        """
        return protein_id in self._protein_to_idx

    def get_embedding(self, protein_id: str) -> np.ndarray | None:
        """Get a cached embedding.

        Args:
            protein_id: UniProt accession.

        Returns:
            Embedding vector or None if not cached.
        """
        if protein_id in self._protein_to_idx and self._embeddings is not None:
            idx = self._protein_to_idx[protein_id]
            result: np.ndarray = self._embeddings[idx]
            return result
        return None

    def download_bulk_embeddings(
        self,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> bool:
        """Download the bulk SwissProt embeddings file.

        Args:
            progress_callback: Optional callback(downloaded_mb, total_mb).

        Returns:
            True if download successful.
        """
        import requests

        if self._bulk_h5_file.exists():
            logger.info(f"Bulk embeddings already downloaded: {self._bulk_h5_file}")
            return True

        logger.info(f"Downloading SwissProt embeddings from {UNIPROT_EMBEDDINGS_URL}")

        try:
            response = requests.get(UNIPROT_EMBEDDINGS_URL, stream=True, timeout=60)
            response.raise_for_status()

            total_size = int(response.headers.get("content-length", 0))
            downloaded = 0
            chunk_size = 8192 * 16  # 128KB chunks

            with self._bulk_h5_file.open("wb") as f:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    f.write(chunk)
                    downloaded += len(chunk)
                    if progress_callback:
                        progress_callback(
                            downloaded // (1024 * 1024),
                            total_size // (1024 * 1024),
                        )

            logger.info(f"Downloaded {downloaded / (1024**3):.2f} GB")
            return True

        except requests.RequestException as e:
            logger.error(f"Failed to download embeddings: {e}")
            if self._bulk_h5_file.exists():
                self._bulk_h5_file.unlink()
            return False

    def extract_embeddings(
        self,
        protein_ids: list[str],
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> dict[str, np.ndarray]:
        """Extract embeddings for specific proteins from the bulk file.

        Args:
            protein_ids: List of UniProt accessions.
            progress_callback: Optional callback(extracted, total).

        Returns:
            Dict mapping protein IDs to embeddings.
        """
        if not self._bulk_h5_file.exists():
            logger.error("Bulk embeddings file not found. Run download first.")
            return {}

        result: dict[str, np.ndarray] = {}

        # Check what's already cached
        missing = []
        for pid in protein_ids:
            embedding = self.get_embedding(pid)
            if embedding is not None:
                result[pid] = embedding
            else:
                missing.append(pid)

        if not missing:
            return result

        # Extract from H5 file
        new_embeddings: list[np.ndarray] = []
        new_protein_ids: list[str] = []

        with h5py.File(self._bulk_h5_file, "r") as f:
            available_ids = set(f.keys())

            for i, pid in enumerate(missing):
                if pid in available_ids:
                    embedding = np.array(f[pid], dtype=np.float32)
                    new_embeddings.append(embedding)
                    new_protein_ids.append(pid)
                    result[pid] = embedding

                if progress_callback:
                    progress_callback(i + 1, len(missing))

        # Add to cache
        if new_embeddings:
            self._append_embeddings(new_embeddings, new_protein_ids)
            self.save()

        logger.info(
            f"Extracted {len(new_embeddings)}/{len(missing)} embeddings "
            f"({len(missing) - len(new_embeddings)} not in SwissProt)"
        )

        return result

    def fetch_embeddings(
        self,
        protein_ids: list[str],
        progress_callback: Callable[[int, int], None] | None = None,
        _save_interval: int = 1000,
    ) -> dict[str, np.ndarray]:
        """Fetch embeddings for proteins (downloads bulk file if needed).

        Args:
            protein_ids: List of UniProt accessions.
            progress_callback: Optional callback(fetched, total).
            _save_interval: Not used (kept for API compatibility).

        Returns:
            Dict mapping protein IDs to embeddings.
        """
        # First, ensure bulk file is downloaded
        if not self._bulk_h5_file.exists():
            self.download_bulk_embeddings()

        # Extract embeddings
        return self.extract_embeddings(protein_ids, progress_callback)

    def _append_embeddings(self, embeddings: list[np.ndarray], protein_ids: list[str]) -> None:
        """Append new embeddings to the cache.

        Args:
            embeddings: List of embedding vectors.
            protein_ids: Corresponding protein IDs.
        """
        if not embeddings:
            return

        new_array = np.stack(embeddings, axis=0)

        if self._embeddings is None:
            self._embeddings = new_array
        else:
            self._embeddings = np.concatenate([self._embeddings, new_array], axis=0)

        for pid in protein_ids:
            self._protein_to_idx[pid] = len(self._protein_ids)
            self._protein_ids.append(pid)

    def get_all_embeddings(self) -> tuple[np.ndarray, list[str]]:
        """Get all cached embeddings.

        Returns:
            Tuple of (embeddings array, protein IDs list).
        """
        if self._embeddings is None:
            return np.array([]).reshape(0, self.embedding_dim), []
        return self._embeddings, self._protein_ids

    def stats(self) -> dict[str, int | float]:
        """Get cache statistics.

        Returns:
            Dict with cache statistics.
        """
        return {
            "total_proteins": len(self._protein_ids),
            "embedding_dim": self.embedding_dim,
            "cache_size_mb": (
                self._embeddings.nbytes / 1024 / 1024 if self._embeddings is not None else 0
            ),
            "bulk_file_exists": self._bulk_h5_file.exists(),
            "bulk_file_size_gb": (
                self._bulk_h5_file.stat().st_size / 1024**3 if self._bulk_h5_file.exists() else 0
            ),
        }
