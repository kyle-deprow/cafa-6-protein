"""Tests for data loading utilities."""

from pathlib import Path
from tempfile import NamedTemporaryFile

import pytest

from cafa_6_protein.data.loader import load_annotations, load_fasta


class TestLoadFasta:
    """Tests for FASTA file loading."""

    def test_load_fasta_basic(self) -> None:
        """Test loading a basic FASTA file."""
        fasta_content = """>P12345
MKTAYIAKQRQ
>P67890
MVLSPADKTNV
"""
        with NamedTemporaryFile(mode="w", suffix=".fasta", delete=False) as f:
            f.write(fasta_content)
            f.flush()

            df = load_fasta(Path(f.name))

            assert len(df) == 2
            assert list(df.columns) == ["protein_id", "sequence"]
            assert df.iloc[0]["protein_id"] == "P12345"
            assert df.iloc[0]["sequence"] == "MKTAYIAKQRQ"

    def test_load_fasta_multiline_sequence(self) -> None:
        """Test loading FASTA with multiline sequences."""
        fasta_content = """>P12345
MKTAYIAKQRQ
ISFVKSHFSRQ
"""
        with NamedTemporaryFile(mode="w", suffix=".fasta", delete=False) as f:
            f.write(fasta_content)
            f.flush()

            df = load_fasta(Path(f.name))

            assert len(df) == 1
            assert df.iloc[0]["sequence"] == "MKTAYIAKQRQISFVKSHFSRQ"


class TestLoadAnnotations:
    """Tests for annotation file loading."""

    def test_load_annotations_basic(self) -> None:
        """Test loading a basic annotations file."""
        tsv_content = "P12345\tGO:0005524\t0.95\nP12345\tGO:0016887\t0.87\n"

        with NamedTemporaryFile(mode="w", suffix=".tsv", delete=False) as f:
            f.write(tsv_content)
            f.flush()

            df = load_annotations(Path(f.name))

            assert len(df) == 2
            assert list(df.columns) == ["protein_id", "go_term", "score"]
            assert df.iloc[0]["protein_id"] == "P12345"
            assert df.iloc[0]["go_term"] == "GO:0005524"
            assert df.iloc[0]["score"] == pytest.approx(0.95)
