"""Literature-based GO term relevance classification.

This module provides a PubMedBERT-based classifier to predict whether
a GO term extracted from an abstract is a true annotation for a protein.
"""

from cafa_6_protein.literature.classifier import GORelevanceClassifier
from cafa_6_protein.literature.dataset import (
    ClassifierDataset,
    prepare_training_data,
)
from cafa_6_protein.literature.schemas import (
    ClassifierConfig,
    ClassifierExample,
    ClassifierPrediction,
)

__all__ = [
    "ClassifierConfig",
    "ClassifierDataset",
    "ClassifierExample",
    "ClassifierPrediction",
    "GORelevanceClassifier",
    "prepare_training_data",
]
