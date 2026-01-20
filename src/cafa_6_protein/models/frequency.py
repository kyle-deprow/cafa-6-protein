"""Frequency-based baseline predictor.

This is a simple baseline that predicts GO terms based on their frequency
in the training set. All test proteins receive the same predictions.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from collections.abc import Iterator


class FrequencyBaseline:
    """Frequency-based baseline predictor.

    Predicts GO terms based on their frequency in training annotations.
    Optionally weighted by Information Accretion (IA) to boost rare terms.

    Attributes:
        top_k: Maximum number of terms to predict per protein.
        min_frequency: Minimum term frequency to include (0-1).
        term_frequencies: Dict mapping GO terms to their frequency.
        term_scores: Dict mapping GO terms to prediction scores.
    """

    def __init__(self, top_k: int = 500, min_frequency: float = 0.0) -> None:
        """Initialize the baseline.

        Args:
            top_k: Maximum terms to predict per protein.
            min_frequency: Minimum term frequency threshold (0-1).
        """
        self.top_k = top_k
        self.min_frequency = min_frequency
        self.term_frequencies: dict[str, float] = {}
        self.term_scores: dict[str, float] = {}
        self._fitted = False

    def fit(
        self,
        annotations: pd.DataFrame,
        ia_weights: pd.DataFrame | None = None,
    ) -> FrequencyBaseline:
        """Fit the baseline on training annotations.

        Args:
            annotations: DataFrame with columns ['protein_id', 'go_term'].
            ia_weights: Optional DataFrame with columns ['go_term', 'ia'].

        Returns:
            Self for chaining.
        """
        n_proteins = annotations["protein_id"].nunique()

        # Count term occurrences (proteins per term)
        term_counts = annotations.groupby("go_term")["protein_id"].nunique()

        # Compute frequencies
        self.term_frequencies = (term_counts / n_proteins).to_dict()

        # Compute scores
        if ia_weights is not None:
            ia_map = ia_weights.set_index("go_term")["ia"].to_dict()
            self.term_scores = {}
            for term, freq in self.term_frequencies.items():
                ia = ia_map.get(term, 1.0)
                # Score = frequency * sqrt(IA) to boost informative rare terms
                self.term_scores[term] = freq * np.sqrt(ia)
        else:
            self.term_scores = self.term_frequencies.copy()

        # Filter by minimum frequency
        self.term_scores = {
            term: score
            for term, score in self.term_scores.items()
            if self.term_frequencies[term] >= self.min_frequency
        }

        # Normalize scores to (0, 1]
        if self.term_scores:
            max_score = max(self.term_scores.values())
            if max_score > 0:
                self.term_scores = {
                    term: score / max_score for term, score in self.term_scores.items()
                }

        self._fitted = True
        return self

    def predict(self, protein_ids: list[str]) -> pd.DataFrame:
        """Predict GO terms for proteins.

        Args:
            protein_ids: List of protein IDs to predict for.

        Returns:
            DataFrame with columns ['protein_id', 'go_term', 'score'].

        Raises:
            ValueError: If fit() has not been called.

        Note:
            For large protein lists, use predict_iter() or write_predictions()
            to avoid memory issues.
        """
        if not self._fitted:
            raise ValueError("must call fit() before predict()")

        if not protein_ids:
            return pd.DataFrame(columns=["protein_id", "go_term", "score"])

        # Get top-k terms by score (precompute once)
        sorted_terms = self._get_sorted_terms()

        # Use numpy for efficient array construction
        n_proteins = len(protein_ids)
        n_terms = len(sorted_terms)

        # For small predictions, use list approach
        if n_proteins * n_terms < 100_000:
            rows = []
            for protein_id in protein_ids:
                for term, score in sorted_terms:
                    rows.append(
                        {
                            "protein_id": protein_id,
                            "go_term": term,
                            "score": score,
                        }
                    )
            return pd.DataFrame(rows)

        # For larger predictions, use numpy broadcasting
        terms = [t[0] for t in sorted_terms]
        scores = [t[1] for t in sorted_terms]

        protein_arr = np.repeat(protein_ids, n_terms)
        term_arr = np.tile(terms, n_proteins)
        score_arr = np.tile(scores, n_proteins)

        return pd.DataFrame(
            {
                "protein_id": protein_arr,
                "go_term": term_arr,
                "score": score_arr,
            }
        )

    def _get_sorted_terms(self) -> list[tuple[str, float]]:
        """Get top-k terms sorted by score descending."""
        return sorted(
            self.term_scores.items(),
            key=lambda x: x[1],
            reverse=True,
        )[: self.top_k]

    def predict_iter(
        self,
        protein_ids: list[str],
        batch_size: int = 10_000,
    ) -> Iterator[pd.DataFrame]:
        """Iterate over predictions in batches.

        Memory-efficient generator for large protein sets.

        Args:
            protein_ids: List of protein IDs.
            batch_size: Number of proteins per batch.

        Yields:
            DataFrames with batch predictions.
        """
        if not self._fitted:
            raise ValueError("must call fit() before predict_iter()")

        sorted_terms = self._get_sorted_terms()
        terms = [t[0] for t in sorted_terms]
        scores = [t[1] for t in sorted_terms]
        n_terms = len(terms)

        for i in range(0, len(protein_ids), batch_size):
            batch_proteins = protein_ids[i : i + batch_size]
            n_batch = len(batch_proteins)

            yield pd.DataFrame(
                {
                    "protein_id": np.repeat(batch_proteins, n_terms),
                    "go_term": np.tile(terms, n_batch),
                    "score": np.tile(scores, n_batch),
                }
            )

    def write_predictions(
        self,
        protein_ids: list[str],
        output_path: Path | str,
        batch_size: int = 10_000,
        progress_callback: callable | None = None,
    ) -> int:
        """Write predictions directly to file in batches.

        Memory-efficient method for large protein sets.

        Args:
            protein_ids: List of protein IDs.
            output_path: Output TSV file path.
            batch_size: Proteins to process per batch.
            progress_callback: Optional callback(processed, total) for progress.

        Returns:
            Total number of predictions written.
        """
        if not self._fitted:
            raise ValueError("must call fit() before write_predictions()")

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        total_rows = 0
        n_proteins = len(protein_ids)

        # Write header
        with output_path.open("w") as f:
            f.write("protein_id\tgo_term\tscore\n")

        # Stream batches to file
        for i, batch_df in enumerate(self.predict_iter(protein_ids, batch_size)):
            batch_df.to_csv(
                output_path,
                sep="\t",
                mode="a",
                header=False,
                index=False,
            )
            total_rows += len(batch_df)

            if progress_callback:
                processed = min((i + 1) * batch_size, n_proteins)
                progress_callback(processed, n_proteins)

        return total_rows

    def get_term_scores(self) -> dict[str, float]:
        """Get the term score dictionary.

        Returns:
            Dict mapping GO terms to their prediction scores.
        """
        return self.term_scores.copy()
