"""Dataset preparation for GO relevance classifier training."""

from __future__ import annotations

import logging
import random
from collections import defaultdict
from typing import TYPE_CHECKING

import pandas as pd
from tqdm import tqdm

from cafa_6_protein.literature.schemas import ClassifierExample

if TYPE_CHECKING:
    from pathlib import Path

    from cafa_6_protein.pubmed import AbstractCache, GOExtractor, PublicationCache

logger = logging.getLogger(__name__)


def prepare_training_data(
    annotations: pd.DataFrame,
    publication_cache: PublicationCache,
    abstract_cache: AbstractCache,
    go_extractor: GOExtractor,
    ic_values: dict[str, float] | None = None,
    neg_sample_ratio: float = 0.3,
    min_ic: float = 0.0,
    seed: int = 42,
    max_proteins: int | None = None,
) -> list[ClassifierExample]:
    """Prepare training data for GO relevance classifier.

    For each protein:
    1. Get true GO annotations
    2. Extract GO terms from associated abstracts
    3. Positive examples: extracted terms that ARE in true annotations
    4. Negative examples: extracted terms that are NOT in true annotations

    Args:
        annotations: DataFrame with 'protein_id' and 'go_term' columns.
        publication_cache: Cache for protein → PMID mappings.
        abstract_cache: Cache for PMID → abstract data.
        go_extractor: GO term extractor from text.
        ic_values: Optional IC values for GO terms.
        neg_sample_ratio: Fraction of negatives to keep (undersampling).
        min_ic: Minimum IC for terms to include.
        seed: Random seed for reproducibility.
        max_proteins: Optional limit on proteins to process.

    Returns:
        List of ClassifierExample instances.
    """
    random.seed(seed)
    ic_values = ic_values or {}

    # Build annotation lookup
    annot_by_protein: dict[str, set[str]] = defaultdict(set)
    for _, row in annotations.iterrows():
        annot_by_protein[row["protein_id"]].add(row["go_term"])

    protein_ids = list(annot_by_protein.keys())
    if max_proteins:
        protein_ids = protein_ids[:max_proteins]

    logger.info(f"Preparing data for {len(protein_ids)} proteins")

    positives: list[ClassifierExample] = []
    negatives: list[ClassifierExample] = []

    for protein_id in tqdm(protein_ids, desc="Extracting examples"):
        pmids = publication_cache.get_pmids(protein_id)
        if not pmids:
            continue

        true_terms = annot_by_protein[protein_id]

        for pmid in pmids:
            abstract_data = abstract_cache.get_abstract(pmid)
            if not abstract_data:
                continue

            title = abstract_data.title or ""
            abstract_text = abstract_data.abstract or ""
            full_text = f"{title} {abstract_text}"

            # Extract GO terms from this abstract
            extracted = go_extractor.extract_from_abstract(title, abstract_text)

            for go_term in extracted:
                ic = ic_values.get(go_term, 0.0)
                if ic < min_ic:
                    continue

                # Get term info
                term_info = go_extractor.dictionary.get_term(go_term)
                if not term_info:
                    continue

                # Count mentions
                term_name = term_info.name.lower()
                mention_count = full_text.lower().count(term_name)
                title_mention = term_name in title.lower()

                example = ClassifierExample(
                    protein_id=protein_id,
                    go_term=go_term,
                    go_term_name=term_info.name,
                    abstract_text=full_text,
                    pmid=pmid,
                    label=1 if go_term in true_terms else 0,
                    ic_score=ic,
                    namespace=term_info.namespace,
                    mention_count=max(1, mention_count),
                    title_mention=title_mention,
                )

                if go_term in true_terms:
                    positives.append(example)
                else:
                    negatives.append(example)

    # Undersample negatives
    n_neg_keep = int(len(negatives) * neg_sample_ratio)
    if n_neg_keep < len(negatives):
        # Prefer high-IC negatives (harder examples)
        negatives_sorted = sorted(negatives, key=lambda x: -x.ic_score)
        # Keep top 50% by IC, sample rest randomly
        n_hard = n_neg_keep // 2
        hard_negatives = negatives_sorted[:n_hard]
        easy_negatives = random.sample(negatives_sorted[n_hard:], n_neg_keep - n_hard)
        negatives = hard_negatives + easy_negatives

    logger.info(
        f"Prepared {len(positives)} positive, {len(negatives)} negative examples "
        f"(ratio: {len(negatives)/max(1, len(positives)):.1f}:1)"
    )

    return positives + negatives


def save_training_data(examples: list[ClassifierExample], output_path: Path) -> None:
    """Save training examples to parquet file.

    Args:
        examples: List of ClassifierExample.
        output_path: Output parquet file path.
    """
    records = [ex.model_dump() for ex in examples]
    df = pd.DataFrame(records)
    df.to_parquet(output_path, index=False)
    logger.info(f"Saved {len(examples)} examples to {output_path}")


def load_training_data(input_path: Path) -> list[ClassifierExample]:
    """Load training examples from parquet file.

    Args:
        input_path: Input parquet file path.

    Returns:
        List of ClassifierExample.
    """
    df = pd.read_parquet(input_path)
    examples = [
        ClassifierExample(**{str(k): v for k, v in row.items()})
        for row in df.to_dict(orient="records")
    ]
    logger.info(f"Loaded {len(examples)} examples from {input_path}")
    return examples


class ClassifierDataset:
    """PyTorch-compatible dataset for GO relevance classifier.

    This is a simple wrapper that can be used with torch DataLoader.
    Actual tokenization happens in the collate function.
    """

    def __init__(self, examples: list[ClassifierExample]) -> None:
        """Initialize dataset.

        Args:
            examples: List of training examples.
        """
        self.examples = examples

    def __len__(self) -> int:
        """Return number of examples."""
        return len(self.examples)

    def __getitem__(self, idx: int) -> ClassifierExample:
        """Get example by index."""
        return self.examples[idx]

    def split(
        self,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        seed: int = 42,
    ) -> tuple[ClassifierDataset, ClassifierDataset, ClassifierDataset]:
        """Split dataset into train/val/test by protein ID.

        Splits by protein to avoid data leakage.

        Args:
            train_ratio: Fraction for training.
            val_ratio: Fraction for validation.
            seed: Random seed.

        Returns:
            Tuple of (train, val, test) datasets.
        """
        random.seed(seed)

        # Group by protein
        by_protein: dict[str, list[ClassifierExample]] = defaultdict(list)
        for ex in self.examples:
            by_protein[ex.protein_id].append(ex)

        proteins = list(by_protein.keys())
        random.shuffle(proteins)

        n_train = int(len(proteins) * train_ratio)
        n_val = int(len(proteins) * val_ratio)

        train_proteins = set(proteins[:n_train])
        val_proteins = set(proteins[n_train : n_train + n_val])
        test_proteins = set(proteins[n_train + n_val :])

        train_examples = [ex for ex in self.examples if ex.protein_id in train_proteins]
        val_examples = [ex for ex in self.examples if ex.protein_id in val_proteins]
        test_examples = [ex for ex in self.examples if ex.protein_id in test_proteins]

        return (
            ClassifierDataset(train_examples),
            ClassifierDataset(val_examples),
            ClassifierDataset(test_examples),
        )

    def get_class_weights(self) -> tuple[int, int, float]:
        """Get class distribution and weight.

        Returns:
            Tuple of (n_positive, n_negative, pos_weight).
        """
        n_pos = sum(1 for ex in self.examples if ex.label == 1)
        n_neg = len(self.examples) - n_pos
        pos_weight = n_neg / max(1, n_pos)
        return n_pos, n_neg, pos_weight
