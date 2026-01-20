"""CAFA evaluation metrics implementation.

Based on the weighted precision/recall/F1 formulas from:
Jiang Y, et al. An expanded evaluation of protein function prediction methods
shows an improvement in accuracy. Genome Biol. (2016) 17(1): 184
"""

import numpy as np
import pandas as pd


def weighted_precision(
    predictions: pd.DataFrame,
    ground_truth: pd.DataFrame,
    weights: dict[str, float],
) -> float:
    """Compute information-accretion weighted precision.

    Args:
        predictions: DataFrame with ['protein_id', 'go_term', 'score'].
        ground_truth: DataFrame with ['protein_id', 'go_term'].
        weights: Information accretion (ia) weights for each GO term.

    Returns:
        Weighted precision score.
    """
    if predictions.empty:
        return 0.0

    # Get predicted terms above threshold
    pred_terms = set(zip(predictions["protein_id"], predictions["go_term"], strict=False))
    true_terms = set(zip(ground_truth["protein_id"], ground_truth["go_term"], strict=False))

    # Compute weighted true positives and total predicted
    weighted_tp = 0.0
    weighted_total = 0.0

    for protein_id, go_term in pred_terms:
        weight = weights.get(go_term, 1.0)
        weighted_total += weight
        if (protein_id, go_term) in true_terms:
            weighted_tp += weight

    if weighted_total == 0:
        return 0.0

    return weighted_tp / weighted_total


def weighted_recall(
    predictions: pd.DataFrame,
    ground_truth: pd.DataFrame,
    weights: dict[str, float],
) -> float:
    """Compute information-accretion weighted recall.

    Args:
        predictions: DataFrame with ['protein_id', 'go_term', 'score'].
        ground_truth: DataFrame with ['protein_id', 'go_term'].
        weights: Information accretion (ia) weights for each GO term.

    Returns:
        Weighted recall score.
    """
    if ground_truth.empty:
        return 0.0

    pred_terms = set(zip(predictions["protein_id"], predictions["go_term"], strict=False))
    true_terms = set(zip(ground_truth["protein_id"], ground_truth["go_term"], strict=False))

    # Compute weighted true positives and total ground truth
    weighted_tp = 0.0
    weighted_total = 0.0

    for protein_id, go_term in true_terms:
        weight = weights.get(go_term, 1.0)
        weighted_total += weight
        if (protein_id, go_term) in pred_terms:
            weighted_tp += weight

    if weighted_total == 0:
        return 0.0

    return weighted_tp / weighted_total


def weighted_f1_score(
    predictions: pd.DataFrame,
    ground_truth: pd.DataFrame,
    weights: dict[str, float],
    threshold: float = 0.5,
) -> float:
    """Compute information-accretion weighted F1 score.

    Args:
        predictions: DataFrame with ['protein_id', 'go_term', 'score'].
        ground_truth: DataFrame with ['protein_id', 'go_term'].
        weights: Information accretion (ia) weights for each GO term.
        threshold: Score threshold for predictions.

    Returns:
        Weighted F1 score.
    """
    # Filter predictions by threshold
    filtered_preds = predictions[predictions["score"] >= threshold].copy()

    precision = weighted_precision(filtered_preds, ground_truth, weights)
    recall = weighted_recall(filtered_preds, ground_truth, weights)

    if precision + recall == 0:
        return 0.0

    return 2 * precision * recall / (precision + recall)


def max_weighted_f1(
    predictions: pd.DataFrame,
    ground_truth: pd.DataFrame,
    weights: dict[str, float],
    thresholds: np.ndarray | None = None,
) -> tuple[float, float]:
    """Find maximum weighted F1 score across thresholds.

    Args:
        predictions: DataFrame with ['protein_id', 'go_term', 'score'].
        ground_truth: DataFrame with ['protein_id', 'go_term'].
        weights: Information accretion (ia) weights for each GO term.
        thresholds: Array of thresholds to try. Defaults to 0.01 to 1.0.

    Returns:
        Tuple of (max_f1, best_threshold).
    """
    if thresholds is None:
        thresholds = np.arange(0.01, 1.01, 0.01)

    best_f1 = 0.0
    best_threshold = 0.5

    for threshold in thresholds:
        f1 = weighted_f1_score(predictions, ground_truth, weights, threshold)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = float(threshold)

    return best_f1, best_threshold
