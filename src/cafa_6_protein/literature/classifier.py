"""GO term relevance classifier using PubMedBERT."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from tqdm import tqdm

from cafa_6_protein.literature.schemas import ClassifierConfig, ClassifierExample

if TYPE_CHECKING:
    import torch

logger = logging.getLogger(__name__)


def _check_torch_available() -> None:
    """Check if torch is available, raise helpful error if not."""
    try:
        import torch  # noqa: F401
    except ImportError as e:
        msg = (
            "torch is required for the literature classifier. "
            "Install with: uv pip install 'cafa-6-protein[classifier]'"
        )
        raise ImportError(msg) from e


def _create_classification_head(
    hidden_size: int = 768,
    num_features: int = 0,
    dropout: float = 0.1,
) -> Any:
    """Create classification head module.

    Args:
        hidden_size: Size of BERT hidden states.
        num_features: Number of additional features.
        dropout: Dropout probability.

    Returns:
        Classification head module.
    """
    import torch
    import torch.nn as nn

    input_size = hidden_size + num_features

    class ClassificationHead(nn.Module):
        """Classification head that optionally concatenates features."""

        def __init__(self) -> None:
            super().__init__()
            self.fc1 = nn.Linear(input_size, 256)
            self.relu = nn.ReLU()
            self.dropout = nn.Dropout(dropout)
            self.fc2 = nn.Linear(256, 1)

        def forward(
            self, cls_embedding: torch.Tensor, features: torch.Tensor | None = None
        ) -> torch.Tensor:
            if features is not None:
                x = torch.cat([cls_embedding, features], dim=-1)
            else:
                x = cls_embedding
            x = self.fc1(x)
            x = self.relu(x)
            x = self.dropout(x)
            return self.fc2(x)

    return ClassificationHead()


class GORelevanceClassifier:
    """PubMedBERT-based classifier for GO term relevance.

    Predicts whether a GO term extracted from an abstract is a
    true annotation for the protein.
    """

    def __init__(
        self,
        config: ClassifierConfig | None = None,
        device: str | None = None,
    ) -> None:
        """Initialize classifier.

        Args:
            config: Classifier configuration.
            device: Device to use ('cuda', 'cpu', or None for auto).
        """
        self.config = config or ClassifierConfig()
        self._device_override = device

        self._encoder: Any = None
        self._tokenizer: Any = None
        self._head: Any = None
        self._is_loaded = False
        self._device: str = "cpu"

    @property
    def device(self) -> str:
        """Get the current device."""
        return self._device

    def _load_encoder(self) -> None:
        """Load the PubMedBERT encoder and tokenizer."""
        if self._is_loaded:
            return

        _check_torch_available()
        import torch
        from transformers import AutoModel, AutoTokenizer

        if self._device_override is None:
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self._device = self._device_override

        logger.info(f"Loading encoder: {self.config.base_model}")
        self._tokenizer = AutoTokenizer.from_pretrained(self.config.base_model)
        self._encoder = AutoModel.from_pretrained(self.config.base_model)

        # Freeze layers if configured
        if self.config.freeze_layers > 0:
            # Freeze embeddings
            for param in self._encoder.embeddings.parameters():
                param.requires_grad = False

            # Freeze specified number of layers
            for i, layer in enumerate(self._encoder.encoder.layer):
                if i < self.config.freeze_layers:
                    for param in layer.parameters():
                        param.requires_grad = False

            n_frozen = self.config.freeze_layers
            n_total = len(self._encoder.encoder.layer)
            logger.info(f"Frozen {n_frozen}/{n_total} encoder layers")

        # Create classification head
        num_features = 6 if self.config.use_features else 0  # ic, ns(3), mention, title
        self._head = _create_classification_head(
            hidden_size=self._encoder.config.hidden_size,
            num_features=num_features,
        )

        self._encoder.to(self._device)
        self._head.to(self._device)
        self._is_loaded = True

    def _prepare_input(
        self,
        examples: list[ClassifierExample],
    ) -> dict[str, Any]:
        """Prepare input tensors from examples.

        Args:
            examples: List of examples.

        Returns:
            Dictionary with input_ids, attention_mask, and optional features.
        """
        import torch

        if not self._tokenizer:
            msg = "Tokenizer not loaded"
            raise RuntimeError(msg)

        # Build input text: [CLS] protein_id [SEP] go_term_name [SEP] abstract
        texts = []
        for ex in examples:
            # Truncate abstract to fit
            abstract = ex.abstract_text[: self.config.max_length * 4]
            text = f"{ex.protein_id} {ex.go_term_name} {abstract}"
            texts.append(text)

        # Tokenize
        encoded = self._tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.config.max_length,
            return_tensors="pt",
        )

        result = {
            "input_ids": encoded["input_ids"].to(self._device),
            "attention_mask": encoded["attention_mask"].to(self._device),
        }

        # Add features if configured
        if self.config.use_features:
            features = []
            for ex in examples:
                # Normalize IC to [0, 1]
                ic_norm = min(ex.ic_score / 15.0, 1.0)

                # One-hot namespace
                ns_bp = 1.0 if ex.namespace == "BP" else 0.0
                ns_mf = 1.0 if ex.namespace == "MF" else 0.0
                ns_cc = 1.0 if ex.namespace == "CC" else 0.0

                # Normalize mention count (log scale)
                mention_norm = min(np.log1p(ex.mention_count) / 3.0, 1.0)

                # Title mention
                title = 1.0 if ex.title_mention else 0.0

                features.append([ic_norm, ns_bp, ns_mf, ns_cc, mention_norm, title])

            result["features"] = torch.tensor(features, dtype=torch.float32).to(self._device)

        # Add labels if available
        labels = [ex.label for ex in examples if ex.label is not None]
        if len(labels) == len(examples):
            result["labels"] = torch.tensor(labels, dtype=torch.float32).to(self._device)

        return result

    def train(
        self,
        train_examples: list[ClassifierExample],
        val_examples: list[ClassifierExample] | None = None,
        pos_weight: float | None = None,
    ) -> dict[str, list[float]]:
        """Train the classifier.

        Args:
            train_examples: Training examples.
            val_examples: Optional validation examples.
            pos_weight: Weight for positive class (overrides config).

        Returns:
            Dictionary with training metrics.
        """
        import torch
        import torch.nn as nn
        from torch.optim import AdamW
        from torch.optim.lr_scheduler import OneCycleLR

        self._load_encoder()

        if not self._encoder or not self._head:
            msg = "Model not loaded"
            raise RuntimeError(msg)

        # Set up optimizer
        params = list(self._head.parameters())
        if self.config.freeze_layers < 12:
            params += [p for p in self._encoder.parameters() if p.requires_grad]

        optimizer = AdamW(
            params,
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

        # Calculate steps
        n_batches = (len(train_examples) + self.config.batch_size - 1) // self.config.batch_size
        total_steps = n_batches * self.config.epochs

        scheduler = OneCycleLR(
            optimizer,
            max_lr=self.config.learning_rate,
            total_steps=total_steps,
            pct_start=self.config.warmup_ratio,
        )

        # Loss function with class weighting
        weight = pos_weight or self.config.pos_weight
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([weight]).to(self._device))

        # Training loop
        history: dict[str, list[float]] = {"train_loss": [], "val_loss": [], "val_f1": []}

        self._encoder.train()
        self._head.train()

        for epoch in range(self.config.epochs):
            epoch_loss = 0.0

            # Shuffle examples
            indices = np.random.permutation(len(train_examples))
            batches = [
                indices[i : i + self.config.batch_size]
                for i in range(0, len(indices), self.config.batch_size)
            ]

            pbar = tqdm(batches, desc=f"Epoch {epoch + 1}/{self.config.epochs}")
            for batch_indices in pbar:
                batch_examples = [train_examples[i] for i in batch_indices]
                inputs = self._prepare_input(batch_examples)

                optimizer.zero_grad()

                # Forward pass
                outputs = self._encoder(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                )
                cls_embedding = outputs.last_hidden_state[:, 0, :]

                features = inputs.get("features")
                logits = self._head(cls_embedding, features).squeeze(-1)

                loss = criterion(logits, inputs["labels"])
                loss.backward()

                optimizer.step()
                scheduler.step()

                epoch_loss += loss.item()
                pbar.set_postfix({"loss": f"{loss.item():.4f}"})

            avg_loss = epoch_loss / len(batches)
            history["train_loss"].append(avg_loss)
            logger.info(f"Epoch {epoch + 1}: train_loss={avg_loss:.4f}")

            # Validation
            if val_examples:
                val_metrics = self.evaluate(val_examples)
                history["val_loss"].append(val_metrics["loss"])
                history["val_f1"].append(val_metrics["f1"])
                logger.info(
                    f"Epoch {epoch + 1}: val_loss={val_metrics['loss']:.4f}, "
                    f"val_f1={val_metrics['f1']:.4f}"
                )

        return history

    def evaluate(
        self,
        examples: list[ClassifierExample],
        threshold: float = 0.5,
    ) -> dict[str, float]:
        """Evaluate classifier on examples.

        Args:
            examples: Examples to evaluate.
            threshold: Classification threshold.

        Returns:
            Dictionary with evaluation metrics.
        """
        import torch
        import torch.nn as nn

        self._load_encoder()

        if not self._encoder or not self._head:
            msg = "Model not loaded"
            raise RuntimeError(msg)

        self._encoder.eval()
        self._head.eval()

        all_labels: list[int] = []
        all_preds: list[float] = []
        total_loss = 0.0

        criterion = nn.BCEWithLogitsLoss()

        with torch.no_grad():
            for i in range(0, len(examples), self.config.batch_size):
                batch = examples[i : i + self.config.batch_size]
                inputs = self._prepare_input(batch)

                outputs = self._encoder(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                )
                cls_embedding = outputs.last_hidden_state[:, 0, :]

                features = inputs.get("features")
                logits = self._head(cls_embedding, features).squeeze(-1)

                if "labels" in inputs:
                    loss = criterion(logits, inputs["labels"])
                    total_loss += loss.item() * len(batch)
                    all_labels.extend(inputs["labels"].cpu().tolist())

                probs = torch.sigmoid(logits).cpu().tolist()
                all_preds.extend(probs)

        # Calculate metrics
        avg_loss = total_loss / len(examples) if examples else 0.0

        if all_labels:
            preds_binary = [1 if p > threshold else 0 for p in all_preds]
            tp = sum(
                1
                for label, p in zip(all_labels, preds_binary, strict=False)
                if label == 1 and p == 1
            )
            fp = sum(
                1
                for label, p in zip(all_labels, preds_binary, strict=False)
                if label == 0 and p == 1
            )
            fn = sum(
                1
                for label, p in zip(all_labels, preds_binary, strict=False)
                if label == 1 and p == 0
            )

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        else:
            precision = recall = f1 = 0.0

        return {
            "loss": avg_loss,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "threshold": threshold,
        }

    def predict(
        self,
        examples: list[ClassifierExample],
    ) -> list[float]:
        """Predict relevance scores for examples.

        Args:
            examples: Examples to predict.

        Returns:
            List of relevance scores (0-1).
        """
        import torch

        self._load_encoder()

        if not self._encoder or not self._head:
            msg = "Model not loaded"
            raise RuntimeError(msg)

        self._encoder.eval()
        self._head.eval()

        all_preds: list[float] = []

        with torch.no_grad():
            for i in range(0, len(examples), self.config.batch_size):
                batch = examples[i : i + self.config.batch_size]
                inputs = self._prepare_input(batch)

                outputs = self._encoder(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                )
                cls_embedding = outputs.last_hidden_state[:, 0, :]

                features = inputs.get("features")
                logits = self._head(cls_embedding, features).squeeze(-1)

                probs = torch.sigmoid(logits).cpu().tolist()
                all_preds.extend(probs)

        return all_preds

    def save(self, output_dir: Path) -> None:
        """Save classifier to directory.

        Args:
            output_dir: Output directory.
        """
        import torch

        if not self._encoder or not self._head or not self._tokenizer:
            msg = "Model not loaded"
            raise RuntimeError(msg)

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save config
        with (output_dir / "config.json").open("w") as f:
            json.dump(self.config.model_dump(), f, indent=2)

        # Save encoder
        self._encoder.save_pretrained(output_dir / "encoder")
        self._tokenizer.save_pretrained(output_dir / "tokenizer")

        # Save classification head
        torch.save(self._head.state_dict(), output_dir / "head.pt")

        logger.info(f"Saved classifier to {output_dir}")

    @classmethod
    def load(cls, model_dir: Path, device: str | None = None) -> GORelevanceClassifier:
        """Load classifier from directory.

        Args:
            model_dir: Model directory.
            device: Device to use.

        Returns:
            Loaded classifier.
        """
        import torch
        from transformers import AutoModel, AutoTokenizer

        model_dir = Path(model_dir)

        # Load config
        with (model_dir / "config.json").open() as f:
            config_dict = json.load(f)
        config = ClassifierConfig(**config_dict)

        classifier = cls(config=config, device=device)
        classifier._load_encoder()

        # Load fine-tuned encoder
        if (model_dir / "encoder").exists():
            classifier._encoder = AutoModel.from_pretrained(model_dir / "encoder")
            classifier._encoder.to(classifier.device)

        # Load tokenizer
        if (model_dir / "tokenizer").exists():
            classifier._tokenizer = AutoTokenizer.from_pretrained(model_dir / "tokenizer")

        # Load classification head
        if classifier._head and (model_dir / "head.pt").exists():
            state_dict = torch.load(model_dir / "head.pt", map_location=classifier.device)
            classifier._head.load_state_dict(state_dict)
            classifier._head.to(classifier.device)

        logger.info(f"Loaded classifier from {model_dir}")
        return classifier
