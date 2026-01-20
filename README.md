# CAFA 6 Protein Function Prediction

A machine learning project for the [CAFA 6 Kaggle Competition](https://www.kaggle.com/competitions/cafa-6-protein-function-prediction) - predicting Gene Ontology (GO) terms for proteins based on their amino acid sequences.

## Overview

Proteins are large molecules responsible for many activities in our cells, tissues, organs, and bodies. This project aims to predict what a protein does based on its amino acid sequence by assigning Gene Ontology (GO) terms across three subontologies:

- **Molecular Function (MF)**: What the protein does at the molecular level
- **Biological Process (BP)**: The biological processes the protein is involved in
- **Cellular Component (CC)**: Where in the cell the protein operates

## Project Structure

```
cafa_6_protein/
├── src/
│   └── cafa_6_protein/
│       ├── __init__.py
│       ├── data/
│       │   ├── __init__.py
│       │   └── loader.py          # Data loading utilities
│       ├── models/
│       │   ├── __init__.py
│       │   └── base.py            # Base predictor interface
│       ├── evaluation/
│       │   ├── __init__.py
│       │   └── metrics.py         # CAFA evaluation metrics
│       └── submission.py          # Submission file utilities
├── tests/
│   ├── __init__.py
│   ├── conftest.py
│   ├── test_data_loader.py
│   ├── test_metrics.py
│   └── test_submission.py
├── pyproject.toml
├── README.md
└── PROJECT_DESCRIPTION.md
```

## Setup

### Prerequisites

- Python 3.11+
- [uv](https://github.com/astral-sh/uv) package manager

### Installation

```bash
# Clone the repository
cd cafa_6_protein

# Create virtual environment and install dependencies
uv sync

# Install development dependencies
uv sync --extra dev
```

## Development

### Running Tests

```bash
uv run pytest
```

With coverage:

```bash
uv run pytest --cov=cafa_6_protein --cov-report=html
```

### Linting and Type Checking

```bash
# Run ruff linter
uv run ruff check .

# Run ruff formatter
uv run ruff format .

# Run mypy type checker
uv run mypy src/
```

## Usage

### Loading Data

```python
from pathlib import Path
from cafa_6_protein.data.loader import load_fasta, load_annotations

# Load protein sequences
sequences = load_fasta(Path("data/train_sequences.fasta"))

# Load GO annotations
annotations = load_annotations(Path("data/train_annotations.tsv"))
```

### Creating Submissions

```python
from pathlib import Path
import pandas as pd
from cafa_6_protein.submission import create_submission, validate_submission

# Create predictions DataFrame
predictions = pd.DataFrame({
    "protein_id": ["P12345", "P12345"],
    "go_term": ["GO:0005524", "GO:0016887"],
    "score": [0.95, 0.87],
})

# Create submission file
create_submission(predictions, Path("submission.tsv"))

# Validate submission
errors = validate_submission(Path("submission.tsv"))
if errors:
    print("Validation errors:", errors)
```

## Evaluation

The competition uses information-accretion weighted F1-measure across three GO subontologies. The final score is the arithmetic mean of the maximum F1 scores for MF, BP, and CC.

See `src/cafa_6_protein/evaluation/metrics.py` for the implementation.

## License

MIT

## Acknowledgments

- [CAFA Challenge](https://www.biofunctionprediction.org/cafa/)
- [Gene Ontology Consortium](http://geneontology.org/)
- Competition organizers at Iowa State University and Northeastern University
