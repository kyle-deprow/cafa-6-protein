"""Ensemble methods for combining multiple prediction sources.

Provides utilities for blending and merging predictions from
different models (GOA baseline, kNN, MLP, etc.).
"""

from typing import Literal

import pandas as pd


def blend_predictions(
    pred1: pd.DataFrame,
    pred2: pd.DataFrame,
    weight1: float = 0.5,
    weight2: float = 0.5,
) -> pd.DataFrame:
    """Blend two prediction DataFrames with weighted averaging.

    For overlapping (protein_id, go_term) pairs, computes weighted average.
    For non-overlapping pairs, uses weighted score from the source that has it.

    Args:
        pred1: First predictions DataFrame with ['protein_id', 'go_term', 'score'].
        pred2: Second predictions DataFrame with ['protein_id', 'go_term', 'score'].
        weight1: Weight for first predictions.
        weight2: Weight for second predictions.

    Returns:
        DataFrame with blended predictions.
    """
    if pred1.empty and pred2.empty:
        return pd.DataFrame(columns=["protein_id", "go_term", "score"])

    if pred1.empty:
        result = pred2.copy()
        result["score"] = result["score"] * weight2
        return result

    if pred2.empty:
        result = pred1.copy()
        result["score"] = result["score"] * weight1
        return result

    # Create a combined index
    pred1_indexed = pred1.set_index(["protein_id", "go_term"])["score"]
    pred2_indexed = pred2.set_index(["protein_id", "go_term"])["score"]

    # Get all unique (protein_id, go_term) pairs
    all_indices = pred1_indexed.index.union(pred2_indexed.index)

    # Compute weighted average
    blended_scores = []
    for idx in all_indices:
        score1 = pred1_indexed.get(idx, 0.0)
        score2 = pred2_indexed.get(idx, 0.0)
        blended_score = weight1 * score1 + weight2 * score2
        blended_scores.append(
            {
                "protein_id": idx[0],
                "go_term": idx[1],
                "score": blended_score,
            }
        )

    return pd.DataFrame(blended_scores)


def merge_predictions(
    predictions: list[pd.DataFrame],
    strategy: Literal["max", "mean"] = "max",
) -> pd.DataFrame:
    """Merge multiple prediction DataFrames.

    Args:
        predictions: List of prediction DataFrames.
        strategy: How to combine scores ('max' or 'mean').

    Returns:
        Merged predictions DataFrame.
    """
    if not predictions:
        return pd.DataFrame(columns=["protein_id", "go_term", "score"])

    # Filter out empty DataFrames
    predictions = [p for p in predictions if not p.empty]
    if not predictions:
        return pd.DataFrame(columns=["protein_id", "go_term", "score"])

    # Concatenate all predictions
    combined = pd.concat(predictions, ignore_index=True)

    # Group by (protein_id, go_term) and aggregate
    if strategy == "max":
        merged = combined.groupby(["protein_id", "go_term"])["score"].max().reset_index()
    elif strategy == "mean":
        merged = combined.groupby(["protein_id", "go_term"])["score"].mean().reset_index()
    else:
        msg = f"Unknown strategy: {strategy}"
        raise ValueError(msg)

    return merged


class Tier0Ensemble:
    """Tier 0 ensemble combining GOA baseline and kNN predictions.

    Attributes:
        goa_weight: Weight for GOA predictions.
        knn_weight: Weight for kNN predictions.
    """

    def __init__(self, goa_weight: float = 0.4, knn_weight: float = 0.6) -> None:
        """Initialize the Tier 0 ensemble.

        Args:
            goa_weight: Weight for GOA predictions (normalized).
            knn_weight: Weight for kNN predictions (normalized).
        """
        # Normalize weights to sum to 1
        total = goa_weight + knn_weight
        if total > 0:
            self.goa_weight = goa_weight / total
            self.knn_weight = knn_weight / total
        else:
            self.goa_weight = 0.5
            self.knn_weight = 0.5

    def predict(
        self,
        goa_predictions: pd.DataFrame | None = None,
        knn_predictions: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """Generate ensemble predictions.

        Args:
            goa_predictions: GOA baseline predictions.
            knn_predictions: kNN-based predictions.

        Returns:
            Blended predictions DataFrame.
        """
        # Handle None inputs
        if goa_predictions is None:
            goa_predictions = pd.DataFrame(columns=["protein_id", "go_term", "score"])
        if knn_predictions is None:
            knn_predictions = pd.DataFrame(columns=["protein_id", "go_term", "score"])

        # Blend predictions
        blended = blend_predictions(
            goa_predictions,
            knn_predictions,
            weight1=self.goa_weight,
            weight2=self.knn_weight,
        )

        # Ensure scores are in valid range (0, 1]
        if not blended.empty:
            blended["score"] = blended["score"].clip(lower=1e-10, upper=1.0)

        return blended
