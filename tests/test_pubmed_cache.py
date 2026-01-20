"""Tests for PubMed cache layer."""

import tempfile
from datetime import datetime
from pathlib import Path

import pytest

from cafa_6_protein.pubmed.cache import AbstractCache, PublicationCache


class TestPublicationCache:
    """Tests for PublicationCache (protein → PMID mappings)."""

    def test_init_creates_cache_dir(self, tmp_path: Path) -> None:
        """Cache creates directory if it doesn't exist."""
        cache_dir = tmp_path / "pubmed" / "cache"
        cache = PublicationCache(cache_dir)
        assert cache_dir.exists()

    def test_add_and_get_publications(self, tmp_path: Path) -> None:
        """Can add and retrieve publication mappings."""
        cache = PublicationCache(tmp_path)
        
        # Add mappings
        cache.add_publications("P12345", ["11111111", "22222222"])
        cache.add_publications("P67890", ["33333333"])
        
        # Retrieve
        assert cache.get_pmids("P12345") == {"11111111", "22222222"}
        assert cache.get_pmids("P67890") == {"33333333"}
        assert cache.get_pmids("UNKNOWN") == set()

    def test_get_all_pmids(self, tmp_path: Path) -> None:
        """Can get all unique PMIDs across all proteins."""
        cache = PublicationCache(tmp_path)
        cache.add_publications("P12345", ["11111111", "22222222"])
        cache.add_publications("P67890", ["22222222", "33333333"])  # overlap
        
        all_pmids = cache.get_all_pmids()
        assert all_pmids == {"11111111", "22222222", "33333333"}

    def test_has_protein(self, tmp_path: Path) -> None:
        """Can check if protein is in cache."""
        cache = PublicationCache(tmp_path)
        cache.add_publications("P12345", ["11111111"])
        
        assert cache.has_protein("P12345")
        assert not cache.has_protein("P99999")

    def test_get_missing_proteins(self, tmp_path: Path) -> None:
        """Can identify which proteins are not yet cached."""
        cache = PublicationCache(tmp_path)
        cache.add_publications("P12345", ["11111111"])
        cache.add_publications("P67890", ["22222222"])
        
        proteins = ["P12345", "P67890", "P11111", "P22222"]
        missing = cache.get_missing_proteins(proteins)
        assert missing == ["P11111", "P22222"]

    def test_save_and_load(self, tmp_path: Path) -> None:
        """Cache persists to disk and reloads."""
        cache1 = PublicationCache(tmp_path)
        cache1.add_publications("P12345", ["11111111", "22222222"])
        cache1.add_publications("P67890", ["33333333"])
        cache1.save()
        
        # Load in new instance
        cache2 = PublicationCache(tmp_path)
        assert cache2.get_pmids("P12345") == {"11111111", "22222222"}
        assert cache2.get_pmids("P67890") == {"33333333"}

    def test_stats(self, tmp_path: Path) -> None:
        """Can get cache statistics."""
        cache = PublicationCache(tmp_path)
        cache.add_publications("P12345", ["11111111", "22222222"])
        cache.add_publications("P67890", ["22222222", "33333333"])
        
        stats = cache.stats()
        assert stats.total_proteins == 2
        assert stats.total_pmids == 3  # unique
        assert stats.proteins_with_pubs == 2

    def test_add_protein_with_no_publications(self, tmp_path: Path) -> None:
        """Proteins with no publications are still tracked."""
        cache = PublicationCache(tmp_path)
        cache.add_publications("P12345", [])  # No pubs found
        
        assert cache.has_protein("P12345")
        assert cache.get_pmids("P12345") == set()
        
        stats = cache.stats()
        assert stats.total_proteins == 1
        assert stats.proteins_with_pubs == 0


class TestAbstractCache:
    """Tests for AbstractCache (PMID → abstract text)."""

    def test_init_creates_database(self, tmp_path: Path) -> None:
        """Cache creates SQLite database."""
        cache = AbstractCache(tmp_path)
        db_path = tmp_path / "abstracts.db"
        assert db_path.exists()

    def test_add_and_get_abstract(self, tmp_path: Path) -> None:
        """Can add and retrieve abstracts."""
        cache = AbstractCache(tmp_path)
        
        cache.add_abstract(
            pmid="12345678",
            title="A study of protein function",
            abstract="We investigated the role of X in Y...",
            pub_year=2024,
            journal="Nature",
        )
        
        result = cache.get_abstract("12345678")
        assert result is not None
        assert result.pmid == "12345678"
        assert result.title == "A study of protein function"
        assert result.abstract == "We investigated the role of X in Y..."
        assert result.pub_year == 2024
        assert result.journal == "Nature"

    def test_get_missing_returns_none(self, tmp_path: Path) -> None:
        """Returns None for missing PMIDs."""
        cache = AbstractCache(tmp_path)
        assert cache.get_abstract("99999999") is None

    def test_has_pmid(self, tmp_path: Path) -> None:
        """Can check if PMID is in cache."""
        cache = AbstractCache(tmp_path)
        cache.add_abstract("12345678", "Title", "Abstract", 2024, "Journal")
        
        assert cache.has_pmid("12345678")
        assert not cache.has_pmid("99999999")

    def test_get_missing_pmids(self, tmp_path: Path) -> None:
        """Can identify which PMIDs are not yet cached."""
        cache = AbstractCache(tmp_path)
        cache.add_abstract("11111111", "T1", "A1", 2024, "J1")
        cache.add_abstract("22222222", "T2", "A2", 2024, "J2")
        
        pmids = ["11111111", "22222222", "33333333", "44444444"]
        missing = cache.get_missing_pmids(pmids)
        assert set(missing) == {"33333333", "44444444"}

    def test_batch_get_abstracts(self, tmp_path: Path) -> None:
        """Can retrieve multiple abstracts at once."""
        cache = AbstractCache(tmp_path)
        cache.add_abstract("11111111", "T1", "Abstract one", 2024, "J1")
        cache.add_abstract("22222222", "T2", "Abstract two", 2024, "J2")
        cache.add_abstract("33333333", "T3", "Abstract three", 2024, "J3")
        
        results = cache.get_abstracts(["11111111", "33333333", "99999999"])
        assert len(results) == 2
        assert results["11111111"].abstract == "Abstract one"
        assert results["33333333"].abstract == "Abstract three"
        assert "99999999" not in results

    def test_mark_not_found(self, tmp_path: Path) -> None:
        """Can mark PMIDs as not found (to avoid re-fetching)."""
        cache = AbstractCache(tmp_path)
        cache.mark_not_found("99999999")
        
        # Should be tracked as processed (not missing)
        assert cache.is_processed("99999999")
        # But no abstract data
        assert cache.get_abstract("99999999") is None

    def test_stats(self, tmp_path: Path) -> None:
        """Can get cache statistics."""
        cache = AbstractCache(tmp_path)
        cache.add_abstract("11111111", "T1", "A1", 2024, "J1")
        cache.add_abstract("22222222", "T2", "A2", 2024, "J2")
        cache.mark_not_found("33333333")
        
        stats = cache.stats()
        assert stats.total_abstracts == 2
        assert stats.total_processed == 3
        assert stats.not_found == 1

    def test_persistence(self, tmp_path: Path) -> None:
        """Data persists across cache instances."""
        cache1 = AbstractCache(tmp_path)
        cache1.add_abstract("12345678", "Title", "Abstract text", 2024, "Journal")
        
        # New instance reads from same DB
        cache2 = AbstractCache(tmp_path)
        result = cache2.get_abstract("12345678")
        assert result is not None
        assert result.title == "Title"

    def test_iterate_abstracts(self, tmp_path: Path) -> None:
        """Can iterate over all abstracts efficiently."""
        cache = AbstractCache(tmp_path)
        cache.add_abstract("11111111", "T1", "A1", 2024, "J1")
        cache.add_abstract("22222222", "T2", "A2", 2024, "J2")
        cache.add_abstract("33333333", "T3", "A3", 2024, "J3")
        
        abstracts = list(cache.iter_abstracts())
        assert len(abstracts) == 3
        pmids = {a.pmid for a in abstracts}
        assert pmids == {"11111111", "22222222", "33333333"}
