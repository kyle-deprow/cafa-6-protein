"""Pydantic schemas for the literature classifier."""

from __future__ import annotations

from pydantic import BaseModel, Field


class ClassifierExample(BaseModel):
    """A single training/inference example for the GO relevance classifier.

    Attributes:
        protein_id: UniProt accession.
        go_term: GO term ID (e.g., GO:0005739).
        go_term_name: Human-readable GO term name.
        abstract_text: Combined title and abstract text.
        pmid: PubMed ID of the source abstract.
        label: 1 if relevant (true annotation), 0 if not. None for inference.
        ic_score: Information content of the GO term.
        namespace: GO namespace (BP, MF, CC).
        mention_count: Number of times term appears in abstract.
        title_mention: Whether term appears in title.
    """

    protein_id: str = Field(..., description="UniProt accession")
    go_term: str = Field(..., pattern=r"^GO:\d{7}$", description="GO term ID")
    go_term_name: str = Field(..., description="GO term name")
    abstract_text: str = Field(..., description="Title + abstract text")
    pmid: str = Field(..., description="PubMed ID")
    label: int | None = Field(None, ge=0, le=1, description="1=relevant, 0=not")
    ic_score: float = Field(default=0.0, ge=0.0, description="Information content")
    namespace: str = Field(default="", description="BP, MF, or CC")
    mention_count: int = Field(default=1, ge=1, description="Term mention count")
    title_mention: bool = Field(default=False, description="Term in title")

    model_config = {"frozen": True}


class ClassifierConfig(BaseModel):
    """Configuration for training the GO relevance classifier.

    Attributes:
        base_model: HuggingFace model name.
        max_length: Maximum sequence length.
        freeze_layers: Number of transformer layers to freeze.
        batch_size: Training batch size.
        learning_rate: Learning rate.
        weight_decay: Weight decay for AdamW.
        warmup_ratio: Warmup ratio for scheduler.
        epochs: Number of training epochs.
        pos_weight: Weight for positive class in loss.
        neg_sample_ratio: Ratio of negatives to keep (for undersampling).
        use_features: Whether to use additional features.
    """

    base_model: str = Field(
        default="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract",
        description="Base model name",
    )
    max_length: int = Field(default=384, ge=64, le=512, description="Max sequence length")
    freeze_layers: int = Field(default=10, ge=0, le=12, description="Layers to freeze")
    batch_size: int = Field(default=32, ge=1, description="Batch size")
    learning_rate: float = Field(default=2e-5, gt=0, description="Learning rate")
    weight_decay: float = Field(default=0.01, ge=0, description="Weight decay")
    warmup_ratio: float = Field(default=0.1, ge=0, le=1, description="Warmup ratio")
    epochs: int = Field(default=3, ge=1, description="Training epochs")
    pos_weight: float = Field(default=10.0, ge=1.0, description="Positive class weight")
    neg_sample_ratio: float = Field(default=0.3, gt=0, le=1, description="Negative sampling ratio")
    use_features: bool = Field(default=True, description="Use additional features")

    model_config = {"frozen": True}


class ClassifierPrediction(BaseModel):
    """Prediction from the GO relevance classifier.

    Attributes:
        protein_id: UniProt accession.
        go_term: GO term ID.
        pmid: Source PubMed ID.
        score: Predicted relevance score (0-1).
        relevant: Whether term is predicted relevant (score > threshold).
    """

    protein_id: str
    go_term: str
    pmid: str
    score: float = Field(..., ge=0.0, le=1.0)
    relevant: bool

    model_config = {"frozen": True}
