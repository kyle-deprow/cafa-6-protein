"""Tests for UniProt API client."""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from cafa_6_protein.pubmed.cache import PublicationCache
from cafa_6_protein.pubmed.uniprot import UniProtClient, parse_uniprot_response


class TestParseUniProtResponse:
    """Tests for parsing UniProt API responses."""

    def test_parse_single_protein_with_references(self) -> None:
        """Parse a protein with publication references."""
        response_data = {
            "results": [
                {
                    "primaryAccession": "P12345",
                    "references": [
                        {
                            "citation": {
                                "citationType": "journal article",
                                "citationCrossReferences": [
                                    {"database": "PubMed", "id": "11111111"},
                                    {"database": "DOI", "id": "10.1234/abc"},
                                ],
                            }
                        },
                        {
                            "citation": {
                                "citationType": "journal article",
                                "citationCrossReferences": [
                                    {"database": "PubMed", "id": "22222222"},
                                ],
                            }
                        },
                    ],
                }
            ]
        }
        
        result = parse_uniprot_response(response_data)
        assert result == {"P12345": ["11111111", "22222222"]}

    def test_parse_protein_without_references(self) -> None:
        """Parse a protein with no publication references."""
        response_data = {
            "results": [
                {
                    "primaryAccession": "P12345",
                    "references": [],
                }
            ]
        }
        
        result = parse_uniprot_response(response_data)
        assert result == {"P12345": []}

    def test_parse_protein_missing_references_key(self) -> None:
        """Parse a protein response missing the references key."""
        response_data = {
            "results": [
                {
                    "primaryAccession": "P12345",
                }
            ]
        }
        
        result = parse_uniprot_response(response_data)
        assert result == {"P12345": []}

    def test_parse_multiple_proteins(self) -> None:
        """Parse response with multiple proteins."""
        response_data = {
            "results": [
                {
                    "primaryAccession": "P12345",
                    "references": [
                        {
                            "citation": {
                                "citationCrossReferences": [
                                    {"database": "PubMed", "id": "11111111"},
                                ],
                            }
                        },
                    ],
                },
                {
                    "primaryAccession": "P67890",
                    "references": [
                        {
                            "citation": {
                                "citationCrossReferences": [
                                    {"database": "PubMed", "id": "22222222"},
                                ],
                            }
                        },
                    ],
                },
            ]
        }
        
        result = parse_uniprot_response(response_data)
        assert result == {
            "P12345": ["11111111"],
            "P67890": ["22222222"],
        }

    def test_parse_empty_response(self) -> None:
        """Parse empty response."""
        response_data = {"results": []}
        result = parse_uniprot_response(response_data)
        assert result == {}


class TestUniProtClient:
    """Tests for UniProt API client."""

    def test_init_creates_cache(self, tmp_path: Path) -> None:
        """Client initializes publication cache."""
        client = UniProtClient(cache_dir=tmp_path)
        assert client.cache is not None
        assert (tmp_path / "publications.parquet").parent.exists()

    def test_fetch_uses_cache(self, tmp_path: Path) -> None:
        """Fetch returns cached results without API call."""
        # Pre-populate cache
        cache = PublicationCache(tmp_path)
        cache.add_publications("P12345", ["11111111", "22222222"])
        cache.save()
        
        # Create client with same cache
        client = UniProtClient(cache_dir=tmp_path)
        
        # Should not make API call
        with patch.object(client, "_fetch_batch") as mock_fetch:
            result = client.fetch_publications(["P12345"])
            mock_fetch.assert_not_called()
        
        assert result["P12345"] == {"11111111", "22222222"}

    def test_fetch_only_missing_proteins(self, tmp_path: Path) -> None:
        """Fetch only queries API for proteins not in cache."""
        # Pre-populate cache with one protein
        cache = PublicationCache(tmp_path)
        cache.add_publications("P12345", ["11111111"])
        cache.save()
        
        client = UniProtClient(cache_dir=tmp_path)
        
        # Mock API to return second protein
        mock_response = {"P67890": ["22222222"]}
        with patch.object(client, "_fetch_batch", return_value=mock_response) as mock_fetch:
            result = client.fetch_publications(["P12345", "P67890"])
            
            # Should only fetch P67890
            mock_fetch.assert_called_once()
            call_args = mock_fetch.call_args[0][0]
            assert call_args == ["P67890"]
        
        assert result["P12345"] == {"11111111"}
        assert result["P67890"] == {"22222222"}

    def test_fetch_saves_to_cache(self, tmp_path: Path) -> None:
        """Fetched results are saved to cache."""
        client = UniProtClient(cache_dir=tmp_path)
        
        mock_response = {"P12345": ["11111111", "22222222"]}
        with patch.object(client, "_fetch_batch", return_value=mock_response):
            client.fetch_publications(["P12345"])
        
        # Verify saved to cache
        new_cache = PublicationCache(tmp_path)
        assert new_cache.get_pmids("P12345") == {"11111111", "22222222"}

    def test_fetch_handles_empty_list(self, tmp_path: Path) -> None:
        """Fetch handles empty protein list."""
        client = UniProtClient(cache_dir=tmp_path)
        result = client.fetch_publications([])
        assert result == {}

    def test_get_cached_publications(self, tmp_path: Path) -> None:
        """Can get publications from cache without API call."""
        cache = PublicationCache(tmp_path)
        cache.add_publications("P12345", ["11111111"])
        cache.add_publications("P67890", ["22222222"])
        cache.save()
        
        client = UniProtClient(cache_dir=tmp_path)
        
        # Get only cached, no API
        result = client.get_cached_publications(["P12345", "P67890", "P99999"])
        
        assert result["P12345"] == {"11111111"}
        assert result["P67890"] == {"22222222"}
        assert "P99999" not in result
