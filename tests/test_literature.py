"""Tests for the literature classifier module."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from cafa_6_protein.literature import (
    ClassifierConfig,
    ClassifierExample,
    ClassifierPrediction,
    prepare_training_data,
)


class TestClassifierExample:
    """Tests for ClassifierExample schema."""

    def test_create_valid_example(self) -> None:
        """Test creating a valid classifier example."""
        example = ClassifierExample(
            protein_id="P12345",
            go_term="GO:0005739",
            go_term_name="mitochondrion",
            abstract_text="This protein localizes to the mitochondrion.",
            pmid="12345678",
            label=1,
            ic_score=5.5,
            namespace="CC",
            mention_count=2,
            title_mention=True,
        )

        assert example.protein_id == "P12345"
        assert example.go_term == "GO:0005739"
        assert example.label == 1
        assert example.ic_score == 5.5
        assert example.namespace == "CC"
        assert example.mention_count == 2
        assert example.title_mention is True

    def test_example_without_label(self) -> None:
        """Test creating example for inference (no label)."""
        example = ClassifierExample(
            protein_id="P12345",
            go_term="GO:0005739",
            go_term_name="mitochondrion",
            abstract_text="Some text.",
            pmid="12345678",
        )

        assert example.label is None

    def test_invalid_go_term_format(self) -> None:
        """Test that invalid GO term format raises error."""
        with pytest.raises(ValueError, match="String should match pattern"):
            ClassifierExample(
                protein_id="P12345",
                go_term="invalid",
                go_term_name="test",
                abstract_text="text",
                pmid="123",
            )

    def test_invalid_label_value(self) -> None:
        """Test that label must be 0 or 1."""
        with pytest.raises(ValueError):
            ClassifierExample(
                protein_id="P12345",
                go_term="GO:0005739",
                go_term_name="test",
                abstract_text="text",
                pmid="123",
                label=2,
            )

    def test_example_is_frozen(self) -> None:
        """Test that example is immutable."""
        example = ClassifierExample(
            protein_id="P12345",
            go_term="GO:0005739",
            go_term_name="mitochondrion",
            abstract_text="text",
            pmid="123",
        )

        with pytest.raises(Exception):  # ValidationError for frozen model
            example.protein_id = "Q99999"  # type: ignore[misc]

    def test_default_values(self) -> None:
        """Test default values are set correctly."""
        example = ClassifierExample(
            protein_id="P12345",
            go_term="GO:0005739",
            go_term_name="mitochondrion",
            abstract_text="text",
            pmid="123",
        )

        assert example.ic_score == 0.0
        assert example.namespace == ""
        assert example.mention_count == 1
        assert example.title_mention is False


class TestClassifierConfig:
    """Tests for ClassifierConfig schema."""

    def test_default_config(self) -> None:
        """Test default configuration values."""
        config = ClassifierConfig()

        assert "PubMedBERT" in config.base_model
        assert config.max_length == 384
        assert config.freeze_layers == 10
        assert config.batch_size == 32
        assert config.learning_rate == 2e-5
        assert config.epochs == 3
        assert config.pos_weight == 10.0
        assert config.neg_sample_ratio == 0.3
        assert config.use_features is True

    def test_custom_config(self) -> None:
        """Test creating custom configuration."""
        config = ClassifierConfig(
            max_length=512,
            batch_size=16,
            learning_rate=1e-5,
            epochs=5,
            freeze_layers=8,
        )

        assert config.max_length == 512
        assert config.batch_size == 16
        assert config.learning_rate == 1e-5
        assert config.epochs == 5
        assert config.freeze_layers == 8

    def test_config_validation_max_length(self) -> None:
        """Test max_length validation."""
        with pytest.raises(ValueError):
            ClassifierConfig(max_length=32)  # Too small

    def test_config_validation_learning_rate(self) -> None:
        """Test learning rate must be positive."""
        with pytest.raises(ValueError):
            ClassifierConfig(learning_rate=0)

    def test_config_validation_neg_sample_ratio(self) -> None:
        """Test neg_sample_ratio must be in (0, 1]."""
        with pytest.raises(ValueError):
            ClassifierConfig(neg_sample_ratio=0)

        with pytest.raises(ValueError):
            ClassifierConfig(neg_sample_ratio=1.5)


class TestClassifierPrediction:
    """Tests for ClassifierPrediction schema."""

    def test_create_prediction(self) -> None:
        """Test creating a prediction."""
        pred = ClassifierPrediction(
            protein_id="P12345",
            go_term="GO:0005739",
            pmid="12345678",
            score=0.85,
            relevant=True,
        )

        assert pred.protein_id == "P12345"
        assert pred.go_term == "GO:0005739"
        assert pred.score == 0.85
        assert pred.relevant is True

    def test_score_validation(self) -> None:
        """Test score must be in [0, 1]."""
        with pytest.raises(ValueError):
            ClassifierPrediction(
                protein_id="P12345",
                go_term="GO:0005739",
                pmid="123",
                score=1.5,
                relevant=True,
            )


class TestPrepareTrainingData:
    """Tests for prepare_training_data function."""

    @pytest.fixture
    def mock_caches(self):
        """Create mock caches for testing."""
        publication_cache = MagicMock()
        abstract_cache = MagicMock()
        go_extractor = MagicMock()

        # Configure publication cache
        publication_cache.get_pmids.return_value = ["PMID123"]

        # Configure abstract cache
        abstract_data = MagicMock()
        abstract_data.title = "Mitochondrial protein function"
        abstract_data.abstract = "This protein is found in the mitochondrion."
        abstract_cache.get_abstract.return_value = abstract_data

        # Configure GO extractor
        go_extractor.extract_from_abstract.return_value = [
            "GO:0005739",  # mitochondrion (true positive)
            "GO:0005634",  # nucleus (false positive)
        ]

        # Configure term info
        term_info_mito = MagicMock()
        term_info_mito.name = "mitochondrion"
        term_info_mito.namespace = "CC"

        term_info_nucleus = MagicMock()
        term_info_nucleus.name = "nucleus"
        term_info_nucleus.namespace = "CC"

        def get_term(go_term):
            if go_term == "GO:0005739":
                return term_info_mito
            elif go_term == "GO:0005634":
                return term_info_nucleus
            return None

        go_extractor.dictionary.get_term.side_effect = get_term

        return publication_cache, abstract_cache, go_extractor

    def test_prepare_creates_examples(self, mock_caches) -> None:
        """Test that prepare_training_data creates examples."""
        publication_cache, abstract_cache, go_extractor = mock_caches

        annotations = pd.DataFrame(
            {
                "protein_id": ["P12345", "P12345"],
                "go_term": ["GO:0005739", "GO:0008150"],
            }
        )

        examples = prepare_training_data(
            annotations=annotations,
            publication_cache=publication_cache,
            abstract_cache=abstract_cache,
            go_extractor=go_extractor,
            neg_sample_ratio=1.0,  # Keep all negatives
        )

        assert len(examples) == 2  # One positive, one negative

    def test_positive_negative_labels(self, mock_caches) -> None:
        """Test that labels are assigned correctly."""
        publication_cache, abstract_cache, go_extractor = mock_caches

        annotations = pd.DataFrame(
            {
                "protein_id": ["P12345"],
                "go_term": ["GO:0005739"],  # Only mitochondrion is true
            }
        )

        examples = prepare_training_data(
            annotations=annotations,
            publication_cache=publication_cache,
            abstract_cache=abstract_cache,
            go_extractor=go_extractor,
            neg_sample_ratio=1.0,
        )

        positives = [e for e in examples if e.label == 1]
        negatives = [e for e in examples if e.label == 0]

        assert len(positives) == 1
        assert len(negatives) == 1
        assert positives[0].go_term == "GO:0005739"
        assert negatives[0].go_term == "GO:0005634"

    def test_skips_proteins_without_pmids(self, mock_caches) -> None:
        """Test that proteins without PMIDs are skipped."""
        publication_cache, abstract_cache, go_extractor = mock_caches
        publication_cache.get_pmids.return_value = []  # No PMIDs

        annotations = pd.DataFrame(
            {
                "protein_id": ["P12345"],
                "go_term": ["GO:0005739"],
            }
        )

        examples = prepare_training_data(
            annotations=annotations,
            publication_cache=publication_cache,
            abstract_cache=abstract_cache,
            go_extractor=go_extractor,
        )

        assert len(examples) == 0

    def test_skips_abstracts_not_in_cache(self, mock_caches) -> None:
        """Test that missing abstracts are skipped."""
        publication_cache, abstract_cache, go_extractor = mock_caches
        abstract_cache.get_abstract.return_value = None  # No abstract

        annotations = pd.DataFrame(
            {
                "protein_id": ["P12345"],
                "go_term": ["GO:0005739"],
            }
        )

        examples = prepare_training_data(
            annotations=annotations,
            publication_cache=publication_cache,
            abstract_cache=abstract_cache,
            go_extractor=go_extractor,
        )

        assert len(examples) == 0

    def test_ic_filtering(self, mock_caches) -> None:
        """Test that low IC terms are filtered."""
        publication_cache, abstract_cache, go_extractor = mock_caches

        annotations = pd.DataFrame(
            {
                "protein_id": ["P12345"],
                "go_term": ["GO:0005739"],
            }
        )

        ic_values = {
            "GO:0005739": 8.0,  # High IC - keep
            "GO:0005634": 1.0,  # Low IC - filter
        }

        examples = prepare_training_data(
            annotations=annotations,
            publication_cache=publication_cache,
            abstract_cache=abstract_cache,
            go_extractor=go_extractor,
            ic_values=ic_values,
            min_ic=5.0,
            neg_sample_ratio=1.0,
        )

        # Only high-IC term should be kept
        assert len(examples) == 1
        assert examples[0].go_term == "GO:0005739"

    def test_negative_undersampling(self, mock_caches) -> None:
        """Test that negative examples are undersampled."""
        publication_cache, abstract_cache, go_extractor = mock_caches

        # Return many negative terms
        go_extractor.extract_from_abstract.return_value = [
            "GO:0005634",  # nucleus
            "GO:0005737",  # cytoplasm
            "GO:0005829",  # cytosol
            "GO:0005886",  # plasma membrane
        ]

        # Add term info for all
        term_infos = {
            "GO:0005634": ("nucleus", "CC"),
            "GO:0005737": ("cytoplasm", "CC"),
            "GO:0005829": ("cytosol", "CC"),
            "GO:0005886": ("plasma membrane", "CC"),
        }

        def get_term(go_term):
            if go_term in term_infos:
                info = MagicMock()
                info.name = term_infos[go_term][0]
                info.namespace = term_infos[go_term][1]
                return info
            return None

        go_extractor.dictionary.get_term.side_effect = get_term

        annotations = pd.DataFrame(
            {
                "protein_id": ["P12345"],
                "go_term": ["GO:0000001"],  # Not in extracted terms
            }
        )

        examples = prepare_training_data(
            annotations=annotations,
            publication_cache=publication_cache,
            abstract_cache=abstract_cache,
            go_extractor=go_extractor,
            neg_sample_ratio=0.5,  # Keep 50%
        )

        # Should have 2 negatives (50% of 4)
        assert len(examples) == 2

    def test_max_proteins_limit(self, mock_caches) -> None:
        """Test that max_proteins limits processing."""
        publication_cache, abstract_cache, go_extractor = mock_caches

        annotations = pd.DataFrame(
            {
                "protein_id": ["P12345", "P12346", "P12347"],
                "go_term": ["GO:0005739", "GO:0005739", "GO:0005739"],
            }
        )

        examples = prepare_training_data(
            annotations=annotations,
            publication_cache=publication_cache,
            abstract_cache=abstract_cache,
            go_extractor=go_extractor,
            max_proteins=1,
            neg_sample_ratio=1.0,
        )

        # Should only process first protein
        protein_ids = {e.protein_id for e in examples}
        assert "P12346" not in protein_ids
        assert "P12347" not in protein_ids

    def test_example_fields_populated(self, mock_caches) -> None:
        """Test that all example fields are populated correctly."""
        publication_cache, abstract_cache, go_extractor = mock_caches

        annotations = pd.DataFrame(
            {
                "protein_id": ["P12345"],
                "go_term": ["GO:0005739"],
            }
        )

        ic_values = {"GO:0005739": 7.5, "GO:0005634": 5.2}

        examples = prepare_training_data(
            annotations=annotations,
            publication_cache=publication_cache,
            abstract_cache=abstract_cache,
            go_extractor=go_extractor,
            ic_values=ic_values,
            neg_sample_ratio=1.0,
        )

        # Check positive example
        pos_example = next(e for e in examples if e.label == 1)
        assert pos_example.protein_id == "P12345"
        assert pos_example.go_term == "GO:0005739"
        assert pos_example.go_term_name == "mitochondrion"
        assert pos_example.pmid == "PMID123"
        assert pos_example.ic_score == 7.5
        assert pos_example.namespace == "CC"
        assert "Mitochondrial" in pos_example.abstract_text


class TestGORelevanceClassifier:
    """Tests for GORelevanceClassifier."""

    def test_import_without_torch(self) -> None:
        """Test that classifier can be imported without torch."""
        from cafa_6_protein.literature import GORelevanceClassifier

        assert GORelevanceClassifier is not None

    def test_classifier_init(self) -> None:
        """Test classifier initialization."""
        from cafa_6_protein.literature import ClassifierConfig, GORelevanceClassifier

        config = ClassifierConfig(batch_size=16)
        classifier = GORelevanceClassifier(config=config, device="cpu")

        assert classifier.config.batch_size == 16
        assert classifier._device_override == "cpu"
        assert classifier._is_loaded is False

    def test_classifier_default_config(self) -> None:
        """Test classifier uses default config if none provided."""
        from cafa_6_protein.literature import GORelevanceClassifier

        classifier = GORelevanceClassifier()

        assert classifier.config.base_model == (
            "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract"
        )

    def test_check_torch_available_raises(self) -> None:
        """Test that helpful error is raised when torch unavailable."""
        from cafa_6_protein.literature.classifier import _check_torch_available

        with patch.dict("sys.modules", {"torch": None}):
            with patch("builtins.__import__", side_effect=ImportError("No torch")):
                with pytest.raises(ImportError, match="torch is required"):
                    _check_torch_available()


class TestClassifierDataset:
    """Tests for ClassifierDataset class."""

    def test_dataset_creation(self) -> None:
        """Test creating dataset from examples."""
        from cafa_6_protein.literature.dataset import ClassifierDataset

        examples = [
            ClassifierExample(
                protein_id="P12345",
                go_term="GO:0005739",
                go_term_name="mitochondrion",
                abstract_text="text",
                pmid="123",
                label=1,
            ),
            ClassifierExample(
                protein_id="P12346",
                go_term="GO:0005634",
                go_term_name="nucleus",
                abstract_text="text2",
                pmid="124",
                label=0,
            ),
        ]

        dataset = ClassifierDataset(examples)

        assert len(dataset) == 2
        assert dataset[0] == examples[0]
        assert dataset[1] == examples[1]

    def test_dataset_split(self) -> None:
        """Test splitting dataset into train/val/test."""
        from cafa_6_protein.literature.dataset import ClassifierDataset

        # Create examples with multiple proteins
        examples = []
        for i in range(10):
            for j in range(10):
                examples.append(
                    ClassifierExample(
                        protein_id=f"P{i}",
                        go_term="GO:0005739",
                        go_term_name="mitochondrion",
                        abstract_text="text",
                        pmid=f"{i}_{j}",
                        label=j % 2,
                    )
                )

        dataset = ClassifierDataset(examples)
        train, val, test = dataset.split(train_ratio=0.6, val_ratio=0.2, seed=42)

        # Should split by protein, not by example
        total_examples = len(train) + len(val) + len(test)
        assert total_examples == len(dataset)

        # Check no protein overlap
        train_proteins = {e.protein_id for e in train.examples}
        val_proteins = {e.protein_id for e in val.examples}
        test_proteins = {e.protein_id for e in test.examples}

        assert len(train_proteins & val_proteins) == 0
        assert len(train_proteins & test_proteins) == 0
        assert len(val_proteins & test_proteins) == 0

    def test_dataset_get_class_weights(self) -> None:
        """Test getting class weights from dataset."""
        from cafa_6_protein.literature.dataset import ClassifierDataset

        examples = [
            ClassifierExample(
                protein_id="P12345",
                go_term="GO:0005739",
                go_term_name="test",
                abstract_text="text",
                pmid="123",
                label=1,
            ),
            ClassifierExample(
                protein_id="P12346",
                go_term="GO:0005634",
                go_term_name="test",
                abstract_text="text2",
                pmid="124",
                label=0,
            ),
            ClassifierExample(
                protein_id="P12347",
                go_term="GO:0005829",
                go_term_name="test",
                abstract_text="text3",
                pmid="125",
                label=0,
            ),
        ]

        dataset = ClassifierDataset(examples)
        n_pos, n_neg, pos_weight = dataset.get_class_weights()

        assert n_pos == 1
        assert n_neg == 2
        assert pos_weight == 2.0


class TestSaveLoadTrainingData:
    """Tests for save_training_data and load_training_data functions."""

    def test_save_and_load_roundtrip(self, tmp_path: Path) -> None:
        """Test that save and load preserves examples."""
        from cafa_6_protein.literature.dataset import (
            load_training_data,
            save_training_data,
        )

        examples = [
            ClassifierExample(
                protein_id="P12345",
                go_term="GO:0005739",
                go_term_name="mitochondrion",
                abstract_text="This protein is in mitochondrion.",
                pmid="12345678",
                label=1,
                ic_score=7.5,
                namespace="CC",
                mention_count=2,
                title_mention=True,
            ),
            ClassifierExample(
                protein_id="P12346",
                go_term="GO:0005634",
                go_term_name="nucleus",
                abstract_text="Nuclear localization observed.",
                pmid="87654321",
                label=0,
                ic_score=4.2,
                namespace="CC",
                mention_count=1,
                title_mention=False,
            ),
        ]

        output_path = tmp_path / "test_examples.parquet"
        save_training_data(examples, output_path)

        assert output_path.exists()

        loaded = load_training_data(output_path)

        assert len(loaded) == len(examples)
        for orig, load in zip(examples, loaded):
            assert orig.protein_id == load.protein_id
            assert orig.go_term == load.go_term
            assert orig.go_term_name == load.go_term_name
            assert orig.label == load.label
            assert orig.ic_score == load.ic_score
            assert orig.namespace == load.namespace
            assert orig.mention_count == load.mention_count
            assert orig.title_mention == load.title_mention

    def test_save_creates_directory(self, tmp_path: Path) -> None:
        """Test that save creates parent directories."""
        from cafa_6_protein.literature.dataset import save_training_data

        examples = [
            ClassifierExample(
                protein_id="P12345",
                go_term="GO:0005739",
                go_term_name="test",
                abstract_text="text",
                pmid="123",
                label=1,
            ),
        ]

        output_path = tmp_path / "nested" / "dir" / "examples.parquet"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        save_training_data(examples, output_path)

        assert output_path.exists()

    def test_load_empty_file(self, tmp_path: Path) -> None:
        """Test loading from parquet with no examples."""
        from cafa_6_protein.literature.dataset import (
            load_training_data,
            save_training_data,
        )

        examples: list[ClassifierExample] = []
        output_path = tmp_path / "empty.parquet"
        save_training_data(examples, output_path)

        loaded = load_training_data(output_path)
        assert len(loaded) == 0
