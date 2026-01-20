"""End-to-end test for the literature classifier.

This test verifies the full pipeline: data prep -> train -> evaluate -> predict.
Uses a small BERT model to run quickly on CPU.

Run with: pytest tests/test_literature_e2e.py -v -s
"""

from __future__ import annotations

from pathlib import Path

import pytest

from cafa_6_protein.literature import (
    ClassifierConfig,
    ClassifierExample,
    GORelevanceClassifier,
)
from cafa_6_protein.literature.dataset import ClassifierDataset


# Skip if torch not available
torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")


def create_synthetic_examples(n_positive: int = 20, n_negative: int = 40) -> list[ClassifierExample]:
    """Create synthetic training examples for testing."""
    examples = []

    # Positive examples - term appears in abstract and is relevant
    positive_templates = [
        ("P{i}", "GO:0005739", "mitochondrion", "CC", 
         "This protein localizes to the mitochondrion and plays a role in mitochondrial function."),
        ("P{i}", "GO:0005634", "nucleus", "CC",
         "Nuclear localization was observed. The protein accumulates in the nucleus."),
        ("P{i}", "GO:0005886", "plasma membrane", "CC",
         "The protein is anchored to the plasma membrane via a transmembrane domain."),
        ("P{i}", "GO:0016020", "membrane", "CC",
         "Membrane-associated protein involved in membrane trafficking."),
        ("P{i}", "GO:0005737", "cytoplasm", "CC",
         "Cytoplasmic protein with widespread cytoplasm distribution."),
    ]

    for i in range(n_positive):
        template = positive_templates[i % len(positive_templates)]
        examples.append(ClassifierExample(
            protein_id=template[0].format(i=i),
            go_term=template[1],
            go_term_name=template[2],
            namespace=template[3],
            abstract_text=template[4],
            pmid=f"POS{i:04d}",
            label=1,
            ic_score=5.0 + (i % 5),
            mention_count=2,
            title_mention=i % 3 == 0,
        ))

    # Negative examples - term appears but is NOT the true annotation
    negative_templates = [
        ("Q{i}", "GO:0005739", "mitochondrion", "CC",
         "While some proteins are mitochondrial, this one is not."),
        ("Q{i}", "GO:0005634", "nucleus", "CC",
         "Despite studies of nuclear proteins, this protein is cytoplasmic."),
        ("Q{i}", "GO:0005886", "plasma membrane", "CC",
         "Plasma membrane markers were used but the target is intracellular."),
        ("Q{i}", "GO:0016020", "membrane", "CC",
         "Membrane fractionation revealed this is a soluble protein."),
        ("Q{i}", "GO:0008150", "biological_process", "BP",
         "The biological process underlying this remains unclear."),
    ]

    for i in range(n_negative):
        template = negative_templates[i % len(negative_templates)]
        examples.append(ClassifierExample(
            protein_id=template[0].format(i=i),
            go_term=template[1],
            go_term_name=template[2],
            namespace=template[3],
            abstract_text=template[4],
            pmid=f"NEG{i:04d}",
            label=0,
            ic_score=3.0 + (i % 4),
            mention_count=1,
            title_mention=False,
        ))

    return examples


@pytest.fixture(scope="module")
def small_config() -> ClassifierConfig:
    """Create a small config for fast CPU testing."""
    return ClassifierConfig(
        # Use a tiny BERT model for fast testing
        base_model="prajjwal1/bert-tiny",  # 4.4M params vs 110M for BERT-base
        max_length=128,
        freeze_layers=0,  # Train all layers (only 2 in tiny)
        batch_size=8,
        learning_rate=5e-5,
        epochs=2,  # Quick training
        pos_weight=2.0,  # 2:1 negative ratio
        use_features=False,  # Simpler for testing
    )


@pytest.fixture(scope="module")
def training_data() -> tuple[list[ClassifierExample], list[ClassifierExample]]:
    """Create train/val split."""
    all_examples = create_synthetic_examples(n_positive=20, n_negative=40)
    dataset = ClassifierDataset(all_examples)
    train, val, _ = dataset.split(train_ratio=0.7, val_ratio=0.3, seed=42)
    return train.examples, val.examples


class TestLiteratureClassifierE2E:
    """End-to-end tests for the literature classifier."""

    @pytest.mark.slow
    def test_train_and_evaluate(
        self,
        small_config: ClassifierConfig,
        training_data: tuple[list[ClassifierExample], list[ClassifierExample]],
    ) -> None:
        """Test full training and evaluation cycle."""
        train_examples, val_examples = training_data

        classifier = GORelevanceClassifier(config=small_config, device="cpu")

        # Train
        history = classifier.train(
            train_examples=train_examples,
            val_examples=val_examples,
        )

        # Check training ran
        assert len(history["train_loss"]) == small_config.epochs
        assert all(loss > 0 for loss in history["train_loss"])

        # Loss should decrease (or at least not explode)
        assert history["train_loss"][-1] < history["train_loss"][0] * 2

        # Validation metrics should be computed
        assert len(history["val_loss"]) == small_config.epochs
        assert len(history["val_f1"]) == small_config.epochs

    @pytest.mark.slow
    def test_predict(
        self,
        small_config: ClassifierConfig,
        training_data: tuple[list[ClassifierExample], list[ClassifierExample]],
    ) -> None:
        """Test prediction on new examples."""
        train_examples, val_examples = training_data

        classifier = GORelevanceClassifier(config=small_config, device="cpu")

        # Quick train (1 epoch)
        quick_config = ClassifierConfig(
            base_model=small_config.base_model,
            max_length=small_config.max_length,
            batch_size=small_config.batch_size,
            epochs=1,
            use_features=False,
        )
        classifier = GORelevanceClassifier(config=quick_config, device="cpu")
        classifier.train(train_examples=train_examples)

        # Predict
        scores = classifier.predict(val_examples)

        assert len(scores) == len(val_examples)
        assert all(0.0 <= s <= 1.0 for s in scores)

    @pytest.mark.slow
    def test_save_and_load(
        self,
        small_config: ClassifierConfig,
        training_data: tuple[list[ClassifierExample], list[ClassifierExample]],
        tmp_path: Path,
    ) -> None:
        """Test saving and loading classifier."""
        train_examples, val_examples = training_data

        # Train
        quick_config = ClassifierConfig(
            base_model=small_config.base_model,
            max_length=small_config.max_length,
            batch_size=small_config.batch_size,
            epochs=1,
            use_features=False,
        )
        classifier = GORelevanceClassifier(config=quick_config, device="cpu")
        classifier.train(train_examples=train_examples)

        # Get predictions before save
        scores_before = classifier.predict(val_examples[:5])

        # Save
        model_dir = tmp_path / "classifier"
        classifier.save(model_dir)

        assert (model_dir / "config.json").exists()
        assert (model_dir / "encoder").exists()
        assert (model_dir / "tokenizer").exists()
        assert (model_dir / "head.pt").exists()

        # Load
        loaded_classifier = GORelevanceClassifier.load(model_dir, device="cpu")

        # Check config preserved
        assert loaded_classifier.config.base_model == quick_config.base_model
        assert loaded_classifier.config.max_length == quick_config.max_length

        # Predictions should match
        scores_after = loaded_classifier.predict(val_examples[:5])

        for before, after in zip(scores_before, scores_after):
            assert abs(before - after) < 1e-5, f"Prediction mismatch: {before} vs {after}"

    @pytest.mark.slow
    def test_evaluate_metrics(
        self,
        small_config: ClassifierConfig,
        training_data: tuple[list[ClassifierExample], list[ClassifierExample]],
    ) -> None:
        """Test that evaluate returns proper metrics."""
        train_examples, val_examples = training_data

        quick_config = ClassifierConfig(
            base_model=small_config.base_model,
            max_length=small_config.max_length,
            batch_size=small_config.batch_size,
            epochs=1,
            use_features=False,
        )
        classifier = GORelevanceClassifier(config=quick_config, device="cpu")
        classifier.train(train_examples=train_examples)

        metrics = classifier.evaluate(val_examples)

        assert "loss" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1" in metrics
        assert "threshold" in metrics

        assert metrics["loss"] > 0
        assert 0.0 <= metrics["precision"] <= 1.0
        assert 0.0 <= metrics["recall"] <= 1.0
        assert 0.0 <= metrics["f1"] <= 1.0


class TestClassifierHeadWithFeatures:
    """Test classifier with additional features."""

    @pytest.mark.slow
    def test_train_with_features(self) -> None:
        """Test training with IC and namespace features."""
        config = ClassifierConfig(
            base_model="prajjwal1/bert-tiny",
            max_length=64,
            batch_size=4,
            epochs=1,
            use_features=True,  # Enable features
        )

        examples = create_synthetic_examples(n_positive=10, n_negative=20)

        classifier = GORelevanceClassifier(config=config, device="cpu")
        history = classifier.train(train_examples=examples)

        assert len(history["train_loss"]) == 1
        assert history["train_loss"][0] > 0


# Quick smoke test that doesn't require pytest.mark.slow
class TestClassifierSmoke:
    """Quick smoke tests that run without training."""

    def test_classifier_loads_encoder(self) -> None:
        """Test that encoder loads without error."""
        config = ClassifierConfig(
            base_model="prajjwal1/bert-tiny",
            max_length=64,
        )
        classifier = GORelevanceClassifier(config=config, device="cpu")
        classifier._load_encoder()

        assert classifier._encoder is not None
        assert classifier._tokenizer is not None
        assert classifier._head is not None
        assert classifier._is_loaded is True

    def test_prepare_input(self) -> None:
        """Test input preparation."""
        config = ClassifierConfig(
            base_model="prajjwal1/bert-tiny",
            max_length=64,
            use_features=False,
        )
        classifier = GORelevanceClassifier(config=config, device="cpu")
        classifier._load_encoder()

        examples = [
            ClassifierExample(
                protein_id="P12345",
                go_term="GO:0005739",
                go_term_name="mitochondrion",
                abstract_text="Test abstract text.",
                pmid="123",
                label=1,
            ),
        ]

        inputs = classifier._prepare_input(examples)

        assert "input_ids" in inputs
        assert "attention_mask" in inputs
        assert "labels" in inputs
        assert inputs["input_ids"].shape[0] == 1  # batch size 1
