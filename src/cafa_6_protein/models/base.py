"""Base model interface for protein function prediction."""

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import pandas as pd


class BasePredictor(ABC):
    """Abstract base class for protein function predictors."""

    @abstractmethod
    def fit(self, sequences: pd.DataFrame, annotations: pd.DataFrame) -> "BasePredictor":
        """Train the model on protein sequences and their GO annotations.

        Args:
            sequences: DataFrame with columns ['protein_id', 'sequence'].
            annotations: DataFrame with columns ['protein_id', 'go_term', 'score'].

        Returns:
            Self for method chaining.
        """
        ...

    @abstractmethod
    def predict(self, sequences: pd.DataFrame) -> pd.DataFrame:
        """Predict GO terms for protein sequences.

        Args:
            sequences: DataFrame with columns ['protein_id', 'sequence'].

        Returns:
            DataFrame with columns ['protein_id', 'go_term', 'score'].
        """
        ...

    @abstractmethod
    def get_params(self) -> dict[str, Any]:
        """Get model parameters.

        Returns:
            Dictionary of model parameters.
        """
        ...

    def score(
        self,
        sequences: pd.DataFrame,
        annotations: pd.DataFrame,
        weights: dict[str, float] | None = None,
    ) -> float:
        """Compute weighted F1 score on the given data.

        Args:
            sequences: DataFrame with columns ['protein_id', 'sequence'].
            annotations: Ground truth annotations.
            weights: Optional information accretion weights for GO terms.

        Returns:
            Weighted F1 score.
        """
        predictions = self.predict(sequences)
        return compute_weighted_f1(predictions, annotations, weights)


def compute_weighted_f1(
    predictions: pd.DataFrame,
    ground_truth: pd.DataFrame,
    weights: dict[str, float] | None = None,
) -> float:
    """Compute information-accretion weighted F1 score.

    Args:
        predictions: Predicted annotations.
        ground_truth: Ground truth annotations.
        weights: Information accretion weights for each GO term.

    Returns:
        Weighted F1 score.
    """
    # Placeholder implementation - to be implemented based on CAFA evaluation
    _ = predictions, ground_truth, weights
    return np.float64(0.0).item()
