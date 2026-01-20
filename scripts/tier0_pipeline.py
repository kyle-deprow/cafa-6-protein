"""Tier 0 Pipeline: GOA Baseline + kNN Predictions.

This script implements the Tier 0 prediction pipeline:
1. Load GOA annotations for test proteins (electronic evidence codes)
2. Load pre-computed embeddings for training and test proteins
3. Run kNN prediction using embedding similarity
4. Blend GOA and kNN predictions
5. Propagate predictions to parent terms
6. Generate submission file

Usage:
    uv run python scripts/tier0_pipeline.py \
        --embeddings data/embeddings/t5_embeddings.h5 \
        --goa data/goa_uniprot_all.gaf.gz \
        --output submissions/tier0_submission.tsv
"""

import argparse
import logging
from pathlib import Path

import pandas as pd

from cafa_6_protein.data.goa import (
    load_goa_annotations,
    filter_by_proteins,
    filter_by_evidence_codes,
    goa_to_predictions,
    EVIDENCE_CODES,
)
from cafa_6_protein.data.loader import load_fasta
from cafa_6_protein.data.ontology import load_go_ontology, propagate_to_parents
from cafa_6_protein.models.ensemble import Tier0Ensemble
from cafa_6_protein.models.knn import KNNPredictor
from cafa_6_protein.submission import create_submission

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_embeddings_from_h5(filepath: Path, protein_ids: set[str] | None = None) -> dict[str, any]:
    """Load protein embeddings from HDF5 file.

    Args:
        filepath: Path to HDF5 file with embeddings.
        protein_ids: Optional set of protein IDs to load (loads all if None).

    Returns:
        Dictionary mapping protein IDs to embedding arrays.
    """
    import h5py
    import numpy as np

    embeddings = {}
    with h5py.File(filepath, "r") as f:
        for key in f.keys():
            if protein_ids is None or key in protein_ids:
                embeddings[key] = np.array(f[key])

    return embeddings


def run_tier0_pipeline(
    train_sequences_path: Path,
    train_terms_path: Path,
    test_sequences_path: Path,
    go_obo_path: Path,
    embeddings_path: Path | None = None,
    goa_path: Path | None = None,
    output_path: Path = Path("submissions/tier0_submission.tsv"),
    goa_weight: float = 0.4,
    knn_weight: float = 0.6,
    k_neighbors: int = 50,
) -> pd.DataFrame:
    """Run the Tier 0 prediction pipeline.

    Args:
        train_sequences_path: Path to training FASTA file.
        train_terms_path: Path to training terms TSV.
        test_sequences_path: Path to test FASTA file.
        go_obo_path: Path to GO OBO file.
        embeddings_path: Path to embeddings HDF5 file (optional, for kNN).
        goa_path: Path to GOA GAF file (optional, for GOA baseline).
        output_path: Path for output submission file.
        goa_weight: Weight for GOA predictions.
        knn_weight: Weight for kNN predictions.
        k_neighbors: Number of neighbors for kNN.

    Returns:
        DataFrame with final predictions.
    """
    # Load test protein IDs
    logger.info("Loading test proteins...")
    test_df = load_fasta(test_sequences_path)
    test_protein_ids = set(test_df["protein_id"].str.split().str[0])  # Handle "ID taxon" format
    logger.info(f"Loaded {len(test_protein_ids)} test proteins")

    # Load training data
    logger.info("Loading training data...")
    train_terms = pd.read_csv(train_terms_path, sep="\t")
    train_terms = train_terms.rename(columns={"EntryID": "protein_id", "term": "go_term"})
    logger.info(f"Loaded {len(train_terms)} training annotations")

    # Load GO ontology
    logger.info("Loading GO ontology...")
    go_graph = load_go_ontology(go_obo_path)
    logger.info(f"Loaded GO graph with {go_graph.number_of_nodes()} terms")

    # Initialize predictions
    goa_predictions = None
    knn_predictions = None

    # Step 1: GOA Baseline (if GOA file provided)
    if goa_path is not None and goa_path.exists():
        logger.info("Loading GOA annotations...")
        goa_annotations = load_goa_annotations(goa_path)
        logger.info(f"Loaded {len(goa_annotations)} GOA annotations")

        # Filter to test proteins
        goa_test = filter_by_proteins(goa_annotations, test_protein_ids)
        logger.info(f"Found {len(goa_test)} annotations for test proteins")

        # Filter to electronic evidence codes
        electronic_codes = EVIDENCE_CODES["electronic"]
        goa_electronic = filter_by_evidence_codes(goa_test, electronic_codes)
        logger.info(f"Found {len(goa_electronic)} electronic annotations")

        # Convert to predictions format
        goa_predictions = goa_to_predictions(goa_electronic, score=1.0)
        logger.info(f"GOA baseline: {len(goa_predictions)} predictions")

    # Step 2: kNN Predictions (if embeddings provided)
    if embeddings_path is not None and embeddings_path.exists():
        logger.info("Loading embeddings...")
        
        # Load training protein IDs
        train_df = load_fasta(train_sequences_path)
        train_protein_ids = set(train_df["protein_id"].str.split("|").str[1])  # Handle "sp|ID|name" format
        
        # Load embeddings for both train and test
        all_protein_ids = train_protein_ids | test_protein_ids
        embeddings = load_embeddings_from_h5(embeddings_path, all_protein_ids)
        logger.info(f"Loaded embeddings for {len(embeddings)} proteins")

        # Split into train and test embeddings
        train_embeddings = {pid: emb for pid, emb in embeddings.items() if pid in train_protein_ids}
        test_embeddings = {pid: emb for pid, emb in embeddings.items() if pid in test_protein_ids}

        if train_embeddings and test_embeddings:
            # Train kNN predictor
            logger.info(f"Training kNN predictor with k={k_neighbors}...")
            knn = KNNPredictor(k=k_neighbors)
            knn.fit(train_embeddings, train_terms)

            # Predict for test proteins
            logger.info("Running kNN predictions...")
            knn_predictions = knn.predict(test_embeddings)
            logger.info(f"kNN: {len(knn_predictions)} predictions")

    # Step 3: Ensemble predictions
    logger.info("Blending predictions...")
    ensemble = Tier0Ensemble(goa_weight=goa_weight, knn_weight=knn_weight)
    predictions = ensemble.predict(goa_predictions=goa_predictions, knn_predictions=knn_predictions)
    logger.info(f"Ensemble: {len(predictions)} predictions")

    # Step 4: Propagate to parents
    logger.info("Propagating to parent terms...")
    propagated = propagate_to_parents(predictions, go_graph)
    logger.info(f"After propagation: {len(propagated)} predictions")

    # Step 5: Create submission
    logger.info(f"Creating submission file: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    create_submission(propagated, output_path)
    logger.info("Done!")

    return propagated


def main() -> None:
    """Main entry point for Tier 0 pipeline."""
    parser = argparse.ArgumentParser(description="Tier 0 Pipeline: GOA + kNN Predictions")
    parser.add_argument(
        "--train-sequences",
        type=Path,
        default=Path("data/Train/train_sequences.fasta"),
        help="Path to training sequences FASTA",
    )
    parser.add_argument(
        "--train-terms",
        type=Path,
        default=Path("data/Train/train_terms.tsv"),
        help="Path to training terms TSV",
    )
    parser.add_argument(
        "--test-sequences",
        type=Path,
        default=Path("data/Test/testsuperset.fasta"),
        help="Path to test sequences FASTA",
    )
    parser.add_argument(
        "--go-obo",
        type=Path,
        default=Path("data/Train/go-basic.obo"),
        help="Path to GO OBO file",
    )
    parser.add_argument(
        "--embeddings",
        type=Path,
        default=None,
        help="Path to embeddings HDF5 file (optional)",
    )
    parser.add_argument(
        "--goa",
        type=Path,
        default=None,
        help="Path to GOA GAF file (optional)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("submissions/tier0_submission.tsv"),
        help="Output submission file path",
    )
    parser.add_argument(
        "--goa-weight",
        type=float,
        default=0.4,
        help="Weight for GOA predictions (default: 0.4)",
    )
    parser.add_argument(
        "--knn-weight",
        type=float,
        default=0.6,
        help="Weight for kNN predictions (default: 0.6)",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=50,
        help="Number of neighbors for kNN (default: 50)",
    )

    args = parser.parse_args()

    run_tier0_pipeline(
        train_sequences_path=args.train_sequences,
        train_terms_path=args.train_terms,
        test_sequences_path=args.test_sequences,
        go_obo_path=args.go_obo,
        embeddings_path=args.embeddings,
        goa_path=args.goa,
        output_path=args.output,
        goa_weight=args.goa_weight,
        knn_weight=args.knn_weight,
        k_neighbors=args.k,
    )


if __name__ == "__main__":
    main()
