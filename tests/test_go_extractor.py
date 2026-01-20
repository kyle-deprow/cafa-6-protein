"""Tests for GO term dictionary and text extraction."""

from pathlib import Path

import pytest

from cafa_6_protein.pubmed.extractor import GODictionary, extract_go_terms


class TestGODictionary:
    """Tests for GO term dictionary."""

    def test_build_from_obo(self, tmp_path: Path) -> None:
        """Build dictionary from OBO file."""
        # Create minimal OBO file
        obo_content = """format-version: 1.2

[Term]
id: GO:0008150
name: biological_process
namespace: biological_process
def: "A biological process." [GO:curator]

[Term]
id: GO:0003674
name: molecular_function
namespace: molecular_function
def: "A molecular function." [GO:curator]

[Term]
id: GO:0005575
name: cellular_component
namespace: cellular_component
def: "A cellular component." [GO:curator]

[Term]
id: GO:0006915
name: apoptotic process
namespace: biological_process
def: "A programmed cell death process." [GO:curator]
synonym: "apoptosis" EXACT []
synonym: "programmed cell death" RELATED []
"""
        obo_path = tmp_path / "test.obo"
        obo_path.write_text(obo_content)
        
        dictionary = GODictionary.from_obo(obo_path)
        
        assert len(dictionary) >= 4
        assert "GO:0006915" in dictionary.terms

    def test_lookup_by_name(self, tmp_path: Path) -> None:
        """Lookup GO terms by name."""
        dictionary = GODictionary()
        dictionary.add_term("GO:0006915", "apoptotic process", "BP")
        
        result = dictionary.lookup("apoptotic process")
        assert result == {"GO:0006915"}

    def test_lookup_by_synonym(self, tmp_path: Path) -> None:
        """Lookup GO terms by synonym."""
        dictionary = GODictionary()
        dictionary.add_term("GO:0006915", "apoptotic process", "BP", 
                          synonyms=["apoptosis", "programmed cell death"])
        
        result = dictionary.lookup("apoptosis")
        assert result == {"GO:0006915"}
        
        result = dictionary.lookup("programmed cell death")
        assert result == {"GO:0006915"}

    def test_lookup_case_insensitive(self) -> None:
        """Lookup is case-insensitive."""
        dictionary = GODictionary()
        dictionary.add_term("GO:0006915", "apoptotic process", "BP")
        
        result = dictionary.lookup("Apoptotic Process")
        assert result == {"GO:0006915"}

    def test_lookup_not_found(self) -> None:
        """Lookup returns empty set for unknown terms."""
        dictionary = GODictionary()
        dictionary.add_term("GO:0006915", "apoptotic process", "BP")
        
        result = dictionary.lookup("unknown term")
        assert result == set()

    def test_get_term_info(self) -> None:
        """Get term info by GO ID."""
        dictionary = GODictionary()
        dictionary.add_term("GO:0006915", "apoptotic process", "BP")
        
        info = dictionary.get_term("GO:0006915")
        assert info is not None
        assert info["name"] == "apoptotic process"
        assert info["namespace"] == "BP"

    def test_filter_by_namespace(self) -> None:
        """Filter terms by namespace."""
        dictionary = GODictionary()
        dictionary.add_term("GO:0006915", "apoptotic process", "BP")
        dictionary.add_term("GO:0003674", "molecular_function", "MF")
        dictionary.add_term("GO:0005575", "cellular_component", "CC")
        
        bp_terms = dictionary.get_terms_by_namespace("BP")
        assert "GO:0006915" in bp_terms
        assert "GO:0003674" not in bp_terms


class TestExtractGOTerms:
    """Tests for extracting GO terms from text."""

    def test_extract_explicit_go_id(self) -> None:
        """Extract explicit GO:XXXXXXX patterns."""
        text = "This protein is involved in apoptosis (GO:0006915)."
        dictionary = GODictionary()
        dictionary.add_term("GO:0006915", "apoptotic process", "BP")
        
        result = extract_go_terms(text, dictionary)
        assert "GO:0006915" in result

    def test_extract_by_term_name(self) -> None:
        """Extract GO terms by name mention."""
        text = "The protein induces apoptotic process in cells."
        dictionary = GODictionary()
        dictionary.add_term("GO:0006915", "apoptotic process", "BP")
        
        result = extract_go_terms(text, dictionary)
        assert "GO:0006915" in result

    def test_extract_by_synonym(self) -> None:
        """Extract GO terms by synonym mention."""
        text = "Treatment triggered apoptosis."
        dictionary = GODictionary()
        dictionary.add_term("GO:0006915", "apoptotic process", "BP",
                          synonyms=["apoptosis"])
        
        result = extract_go_terms(text, dictionary)
        assert "GO:0006915" in result

    def test_extract_multiple_terms(self) -> None:
        """Extract multiple GO terms from text."""
        text = "The protein regulates cell cycle (GO:0007049) and apoptosis."
        dictionary = GODictionary()
        dictionary.add_term("GO:0007049", "cell cycle", "BP")
        dictionary.add_term("GO:0006915", "apoptotic process", "BP",
                          synonyms=["apoptosis"])
        
        result = extract_go_terms(text, dictionary)
        assert "GO:0007049" in result
        assert "GO:0006915" in result

    def test_extract_from_empty_text(self) -> None:
        """Extract from empty text returns empty set."""
        dictionary = GODictionary()
        dictionary.add_term("GO:0006915", "apoptotic process", "BP")
        
        result = extract_go_terms("", dictionary)
        assert result == set()

    def test_extract_case_insensitive(self) -> None:
        """Extraction is case-insensitive."""
        text = "APOPTOSIS was observed in treated cells."
        dictionary = GODictionary()
        dictionary.add_term("GO:0006915", "apoptotic process", "BP",
                          synonyms=["apoptosis"])
        
        result = extract_go_terms(text, dictionary)
        assert "GO:0006915" in result

    def test_extract_with_word_boundaries(self) -> None:
        """Only extract complete term matches."""
        text = "The protein shows kinase activity."
        dictionary = GODictionary()
        dictionary.add_term("GO:0016301", "kinase activity", "MF")
        dictionary.add_term("GO:0004672", "protein kinase activity", "MF")
        
        result = extract_go_terms(text, dictionary)
        # Should match "kinase activity" but not partial matches
        assert "GO:0016301" in result
