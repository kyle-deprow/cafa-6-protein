"""Tests for GOA (Gene Ontology Annotation) file loading utilities.

TDD tests for loading and filtering GOA annotations from GAF format files.
"""

from pathlib import Path
from tempfile import NamedTemporaryFile

import pandas as pd
import pytest

from cafa_6_protein.data.goa import load_goa_annotations, filter_by_proteins, EVIDENCE_CODES


# Sample GAF 2.2 format content (tab-separated, 17 columns)
# Columns: DB, DB_Object_ID, DB_Object_Symbol, Qualifier, GO_ID, DB:Reference,
#          Evidence_Code, With/From, Aspect, DB_Object_Name, DB_Object_Synonym,
#          DB_Object_Type, Taxon, Date, Assigned_By, Annotation_Extension, Gene_Product_Form_ID
SAMPLE_GAF_CONTENT = """!gaf-version: 2.2
!date: 2025-06-01
UniProtKB	P12345	ACTB	enables	GO:0005524	PMID:12345	IEA	InterPro:IPR000001	F	Actin	ACTB_HUMAN	protein	taxon:9606	20250601	UniProt		
UniProtKB	P12345	ACTB	located_in	GO:0005737	PMID:12345	IDA	 	C	Actin	ACTB_HUMAN	protein	taxon:9606	20250601	UniProt		
UniProtKB	P67890	HBA1	enables	GO:0005833	PMID:67890	IEA	InterPro:IPR000002	F	Hemoglobin	HBA1_HUMAN	protein	taxon:9606	20250601	UniProt		
UniProtKB	Q11111	TP53	enables	GO:0003677	PMID:11111	EXP		F	p53	TP53_HUMAN	protein	taxon:9606	20250601	UniProt		
UniProtKB	Q11111	TP53	NOT|enables	GO:0005524	PMID:11111	IEA	InterPro:IPR000003	F	p53	TP53_HUMAN	protein	taxon:9606	20250601	UniProt		
UniProtKB	A0A0C5B5G6	MOTSC	involved_in	GO:0008150	PMID:99999	ISS	UniProtKB:P12345	P	MOTS-c	MOTSC_HUMAN	protein	taxon:9606	20250601	UniProt		
"""


class TestLoadGoaAnnotations:
    """Tests for loading GOA annotations from GAF files."""

    def test_load_goa_basic(self, tmp_path: Path) -> None:
        """Test loading a basic GAF file."""
        gaf_file = tmp_path / "test.gaf"
        gaf_file.write_text(SAMPLE_GAF_CONTENT)

        df = load_goa_annotations(gaf_file)

        # Should skip header comments and NOT-qualified annotations
        assert len(df) >= 4
        assert "protein_id" in df.columns
        assert "go_term" in df.columns
        assert "evidence_code" in df.columns
        assert "aspect" in df.columns

    def test_load_goa_excludes_not_annotations(self, tmp_path: Path) -> None:
        """Test that NOT-qualified annotations are excluded."""
        gaf_file = tmp_path / "test.gaf"
        gaf_file.write_text(SAMPLE_GAF_CONTENT)

        df = load_goa_annotations(gaf_file)

        # Q11111 with GO:0005524 has NOT qualifier - should be excluded
        q11111_atp = df[(df["protein_id"] == "Q11111") & (df["go_term"] == "GO:0005524")]
        assert len(q11111_atp) == 0

    def test_load_goa_parses_protein_id(self, tmp_path: Path) -> None:
        """Test that protein IDs are correctly parsed from DB_Object_ID column."""
        gaf_file = tmp_path / "test.gaf"
        gaf_file.write_text(SAMPLE_GAF_CONTENT)

        df = load_goa_annotations(gaf_file)

        protein_ids = set(df["protein_id"].unique())
        assert "P12345" in protein_ids
        assert "P67890" in protein_ids
        assert "Q11111" in protein_ids
        assert "A0A0C5B5G6" in protein_ids

    def test_load_goa_parses_go_terms(self, tmp_path: Path) -> None:
        """Test that GO terms are correctly parsed."""
        gaf_file = tmp_path / "test.gaf"
        gaf_file.write_text(SAMPLE_GAF_CONTENT)

        df = load_goa_annotations(gaf_file)

        go_terms = set(df["go_term"].unique())
        assert "GO:0005524" in go_terms
        assert "GO:0005737" in go_terms
        assert "GO:0005833" in go_terms

    def test_load_goa_parses_evidence_codes(self, tmp_path: Path) -> None:
        """Test that evidence codes are correctly parsed."""
        gaf_file = tmp_path / "test.gaf"
        gaf_file.write_text(SAMPLE_GAF_CONTENT)

        df = load_goa_annotations(gaf_file)

        evidence_codes = set(df["evidence_code"].unique())
        assert "IEA" in evidence_codes
        assert "IDA" in evidence_codes
        assert "EXP" in evidence_codes
        assert "ISS" in evidence_codes

    def test_load_goa_parses_aspect(self, tmp_path: Path) -> None:
        """Test that ontology aspect (F/P/C) is correctly parsed."""
        gaf_file = tmp_path / "test.gaf"
        gaf_file.write_text(SAMPLE_GAF_CONTENT)

        df = load_goa_annotations(gaf_file)

        aspects = set(df["aspect"].unique())
        assert "F" in aspects  # Molecular Function
        assert "C" in aspects  # Cellular Component
        assert "P" in aspects  # Biological Process


class TestFilterByProteins:
    """Tests for filtering annotations by protein list."""

    def test_filter_by_proteins_basic(self, tmp_path: Path) -> None:
        """Test filtering annotations to specific proteins."""
        gaf_file = tmp_path / "test.gaf"
        gaf_file.write_text(SAMPLE_GAF_CONTENT)

        df = load_goa_annotations(gaf_file)
        target_proteins = {"P12345", "P67890"}
        
        filtered = filter_by_proteins(df, target_proteins)

        assert set(filtered["protein_id"].unique()) == target_proteins

    def test_filter_by_proteins_empty_result(self, tmp_path: Path) -> None:
        """Test filtering with non-existent proteins returns empty."""
        gaf_file = tmp_path / "test.gaf"
        gaf_file.write_text(SAMPLE_GAF_CONTENT)

        df = load_goa_annotations(gaf_file)
        target_proteins = {"NONEXISTENT1", "NONEXISTENT2"}
        
        filtered = filter_by_proteins(df, target_proteins)

        assert len(filtered) == 0


class TestFilterByEvidenceCodes:
    """Tests for filtering annotations by evidence codes."""

    def test_filter_electronic_annotations(self, tmp_path: Path) -> None:
        """Test filtering for electronic annotations (IEA, ISS, ISO, etc.)."""
        gaf_file = tmp_path / "test.gaf"
        gaf_file.write_text(SAMPLE_GAF_CONTENT)

        df = load_goa_annotations(gaf_file)
        
        # Filter for computational evidence codes useful for baseline
        electronic_codes = {"IEA", "ISS", "ISO", "ISA", "ISM", "IGC", "IBA", "IBD", "IKR", "IRD"}
        electronic_df = df[df["evidence_code"].isin(electronic_codes)]

        assert len(electronic_df) > 0
        assert all(code in electronic_codes for code in electronic_df["evidence_code"].unique())

    def test_evidence_codes_constant_defined(self) -> None:
        """Test that EVIDENCE_CODES constant is properly defined."""
        assert "IEA" in EVIDENCE_CODES["electronic"]
        assert "EXP" in EVIDENCE_CODES["experimental"]
        assert "TAS" in EVIDENCE_CODES["curated"]


class TestGoaToSubmissionFormat:
    """Tests for converting GOA annotations to submission format."""

    def test_goa_to_predictions_format(self, tmp_path: Path) -> None:
        """Test converting GOA annotations to prediction format with scores."""
        gaf_file = tmp_path / "test.gaf"
        gaf_file.write_text(SAMPLE_GAF_CONTENT)

        df = load_goa_annotations(gaf_file)
        
        # Convert to prediction format
        predictions = df[["protein_id", "go_term"]].copy()
        predictions["score"] = 1.0  # GOA baseline uses confidence 1.0

        assert "protein_id" in predictions.columns
        assert "go_term" in predictions.columns
        assert "score" in predictions.columns
        assert all(predictions["score"] == 1.0)


class TestLoadGoaFromGzipped:
    """Tests for loading gzipped GAF files."""

    def test_load_goa_gzipped(self, tmp_path: Path) -> None:
        """Test loading a gzipped GAF file."""
        import gzip

        gaf_file = tmp_path / "test.gaf.gz"
        with gzip.open(gaf_file, "wt") as f:
            f.write(SAMPLE_GAF_CONTENT)

        df = load_goa_annotations(gaf_file)

        assert len(df) >= 4
        assert "protein_id" in df.columns
