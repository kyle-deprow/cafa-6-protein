"""kNN-based GO term prediction using protein embeddings.

Implements nearest neighbor prediction with distance-weighted voting
for GO term prediction.
"""

from typing import Any

import numpy as np
import pandas as pd

# Try to import FAISS, fall back to sklearn if not available
try:
    import faiss

    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False


def distance_weighted_vote(
    distances: np.ndarray,
    annotations: list[dict[str, float]],
    normalize: bool = True,
) -> dict[str, float]:
    """Compute distance-weighted votes for GO terms.

    Uses weight = 1 / (1 + distance) so closer neighbors have more influence.

    Args:
        distances: Array of distances to each neighbor.
        annotations: List of dicts mapping GO terms to scores for each neighbor.
        normalize: Whether to normalize scores to [0, 1].

    Returns:
        Dictionary mapping GO terms to aggregated scores.
    """
    if len(distances) == 0 or len(annotations) == 0:
        return {}

    # Compute weights: closer = higher weight
    weights = 1.0 / (1.0 + distances)

    # Aggregate votes
    term_scores: dict[str, float] = {}
    for weight, neighbor_terms in zip(weights, annotations, strict=True):
        for term, score in neighbor_terms.items():
            if term not in term_scores:
                term_scores[term] = 0.0
            term_scores[term] += weight * score

    # Normalize to [0, 1] if requested
    if normalize and term_scores:
        max_score = max(term_scores.values())
        if max_score > 0:
            term_scores = {term: score / max_score for term, score in term_scores.items()}

    return term_scores


def aggregate_neighbor_annotations(
    neighbor_ids: list[str],
    annotations_df: pd.DataFrame,
) -> list[dict[str, float]]:
    """Get annotations for each neighbor protein.

    Args:
        neighbor_ids: List of neighbor protein IDs.
        annotations_df: DataFrame with 'protein_id' and 'go_term' columns.

    Returns:
        List of dicts, one per neighbor, mapping GO terms to 1.0.
    """
    # Group annotations by protein
    protein_terms: dict[str, dict[str, float]] = {}
    for _, row in annotations_df.iterrows():
        pid = row["protein_id"]
        term = row["go_term"]
        if pid not in protein_terms:
            protein_terms[pid] = {}
        protein_terms[pid][term] = 1.0

    # Get annotations for each neighbor
    result = []
    for pid in neighbor_ids:
        result.append(protein_terms.get(pid, {}))

    return result


class KNNPredictor:
    """kNN-based GO term predictor using protein embeddings.

    Uses FAISS for efficient nearest neighbor search if available,
    otherwise falls back to sklearn.

    Attributes:
        k: Number of neighbors to use for prediction.
        use_faiss: Whether to use FAISS (if available).
    """

    def __init__(self, k: int = 50, use_faiss: bool = True) -> None:
        """Initialize the KNN predictor.

        Args:
            k: Number of neighbors to consider.
            use_faiss: Whether to use FAISS for indexing (if available).
        """
        self.k = k
        self.use_faiss = use_faiss and FAISS_AVAILABLE
        self._index: Any = None
        self._protein_ids: list[str] = []
        self._annotations: pd.DataFrame = pd.DataFrame()
        self._embedding_dim: int = 0

    def fit(
        self,
        embeddings: dict[str, np.ndarray],
        annotations: pd.DataFrame,
    ) -> "KNNPredictor":
        """Fit the predictor with training embeddings and annotations.

        Args:
            embeddings: Dict mapping protein IDs to embedding vectors.
            annotations: DataFrame with 'protein_id' and 'go_term' columns.

        Returns:
            Self for method chaining.
        """
        self._protein_ids = list(embeddings.keys())
        self._annotations = annotations.copy()

        # Stack embeddings into matrix
        embedding_matrix = np.vstack([embeddings[pid] for pid in self._protein_ids])
        self._embedding_dim = embedding_matrix.shape[1]

        # Ensure float32 for FAISS compatibility
        embedding_matrix = embedding_matrix.astype(np.float32)

        if self.use_faiss:
            # Build FAISS index
            self._index = faiss.IndexFlatL2(self._embedding_dim)
            self._index.add(embedding_matrix)
        else:
            # Use sklearn NearestNeighbors as fallback
            from sklearn.neighbors import NearestNeighbors

            self._index = NearestNeighbors(
                n_neighbors=min(self.k, len(self._protein_ids)), metric="euclidean"
            )
            self._index.fit(embedding_matrix)

        return self

    def predict_one(
        self,
        protein_id: str,
        embedding: np.ndarray,
    ) -> pd.DataFrame:
        """Predict GO terms for a single protein.

        Args:
            protein_id: ID for the protein being predicted.
            embedding: Embedding vector for the protein.

        Returns:
            DataFrame with columns ['protein_id', 'go_term', 'score'].
        """
        embedding = embedding.astype(np.float32).reshape(1, -1)

        # Adjust k if larger than training set
        actual_k = min(self.k, len(self._protein_ids))

        if self.use_faiss:
            distances, indices = self._index.search(embedding, actual_k)
            distances = distances[0]
            indices = indices[0]
        else:
            # sklearn returns distances and indices
            distances, indices = self._index.kneighbors(embedding, n_neighbors=actual_k)
            distances = distances[0]
            indices = indices[0]

        # Get neighbor protein IDs
        neighbor_ids = [self._protein_ids[i] for i in indices if i < len(self._protein_ids)]
        distances = distances[: len(neighbor_ids)]

        # Get annotations for neighbors
        neighbor_annotations = aggregate_neighbor_annotations(neighbor_ids, self._annotations)

        # Compute distance-weighted votes
        term_scores = distance_weighted_vote(distances, neighbor_annotations, normalize=True)

        # Convert to DataFrame
        if not term_scores:
            return pd.DataFrame(columns=["protein_id", "go_term", "score"])

        records = [
            {"protein_id": protein_id, "go_term": term, "score": score}
            for term, score in term_scores.items()
        ]
        return pd.DataFrame(records)

    def predict(
        self,
        embeddings: dict[str, np.ndarray],
    ) -> pd.DataFrame:
        """Predict GO terms for multiple proteins.

        Args:
            embeddings: Dict mapping protein IDs to embedding vectors.

        Returns:
            DataFrame with columns ['protein_id', 'go_term', 'score'].
        """
        all_predictions = []
        for protein_id, embedding in embeddings.items():
            preds = self.predict_one(protein_id, embedding)
            all_predictions.append(preds)

        if not all_predictions:
            return pd.DataFrame(columns=["protein_id", "go_term", "score"])

        return pd.concat(all_predictions, ignore_index=True)
