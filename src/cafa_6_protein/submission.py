"""Utilities for creating CAFA submission files."""

from pathlib import Path

import pandas as pd


def create_submission(
    predictions: pd.DataFrame,
    output_path: Path,
    max_terms_per_protein: int = 1500,
) -> None:
    """Create a CAFA-formatted submission file.

    Args:
        predictions: DataFrame with columns ['protein_id', 'go_term', 'score'].
        output_path: Path to write the submission file.
        max_terms_per_protein: Maximum GO terms per protein (default 1500).
    """
    # Sort by protein_id and score (descending)
    df = predictions.sort_values(
        ["protein_id", "score"],
        ascending=[True, False],
    ).copy()

    # Limit terms per protein
    df = df.groupby("protein_id").head(max_terms_per_protein)

    # Format score to 3 significant figures
    df["score"] = df["score"].apply(lambda x: f"{x:.3g}")

    # Remove predictions with score 0
    df = df[df["score"] != "0"]

    # Write tab-separated file without header
    df[["protein_id", "go_term", "score"]].to_csv(
        output_path,
        sep="\t",
        header=False,
        index=False,
    )


def validate_submission(filepath: Path) -> list[str]:
    """Validate a submission file format.

    Args:
        filepath: Path to the submission file.

    Returns:
        List of validation errors (empty if valid).
    """
    errors: list[str] = []

    try:
        df = pd.read_csv(filepath, sep="\t", header=None, names=["protein_id", "go_term", "score"])
    except Exception as e:
        errors.append(f"Failed to read file: {e}")
        return errors

    # Check for required columns
    if len(df.columns) < 3:
        errors.append("File must have 3 tab-separated columns")

    # Check score range
    invalid_scores = df[(df["score"] <= 0) | (df["score"] > 1)]
    if not invalid_scores.empty:
        errors.append(f"Found {len(invalid_scores)} scores outside (0, 1] range")

    # Check GO term format
    invalid_go = df[~df["go_term"].str.match(r"^GO:\d{7}$", na=False)]
    text_rows = df[df["go_term"] == "Text"]
    invalid_go = invalid_go[~invalid_go.index.isin(text_rows.index)]
    if not invalid_go.empty:
        errors.append(f"Found {len(invalid_go)} invalid GO term formats")

    # Check terms per protein limit
    terms_per_protein = df[df["go_term"] != "Text"].groupby("protein_id").size()
    over_limit = terms_per_protein[terms_per_protein > 1500]
    if not over_limit.empty:
        errors.append(f"Found {len(over_limit)} proteins with >1500 GO terms")

    return errors
