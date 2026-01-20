"""Tests for NCBI/PubMed client."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from cafa_6_protein.pubmed.ncbi import NCBIClient, parse_efetch_response


class TestParseEfetchResponse:
    """Tests for parsing NCBI E-fetch XML responses."""

    def test_parse_single_abstract(self) -> None:
        """Parse response with one article."""
        xml_response = """<?xml version="1.0"?>
        <PubmedArticleSet>
            <PubmedArticle>
                <MedlineCitation>
                    <PMID>12345678</PMID>
                    <Article>
                        <ArticleTitle>Test Article Title</ArticleTitle>
                        <Abstract>
                            <AbstractText>This is the abstract text.</AbstractText>
                        </Abstract>
                    </Article>
                </MedlineCitation>
            </PubmedArticle>
        </PubmedArticleSet>
        """
        result = parse_efetch_response(xml_response)
        
        assert len(result) == 1
        assert "12345678" in result
        assert result["12345678"]["title"] == "Test Article Title"
        assert result["12345678"]["abstract"] == "This is the abstract text."

    def test_parse_multiple_abstract_sections(self) -> None:
        """Parse structured abstract with multiple sections."""
        xml_response = """<?xml version="1.0"?>
        <PubmedArticleSet>
            <PubmedArticle>
                <MedlineCitation>
                    <PMID>12345678</PMID>
                    <Article>
                        <ArticleTitle>Structured Abstract Article</ArticleTitle>
                        <Abstract>
                            <AbstractText Label="BACKGROUND">Background info.</AbstractText>
                            <AbstractText Label="METHODS">Methods used.</AbstractText>
                            <AbstractText Label="RESULTS">Results found.</AbstractText>
                            <AbstractText Label="CONCLUSIONS">Final conclusions.</AbstractText>
                        </Abstract>
                    </Article>
                </MedlineCitation>
            </PubmedArticle>
        </PubmedArticleSet>
        """
        result = parse_efetch_response(xml_response)
        
        assert "12345678" in result
        abstract = result["12345678"]["abstract"]
        assert "Background info." in abstract
        assert "Methods used." in abstract
        assert "Results found." in abstract
        assert "Final conclusions." in abstract

    def test_parse_article_without_abstract(self) -> None:
        """Parse article that has no abstract."""
        xml_response = """<?xml version="1.0"?>
        <PubmedArticleSet>
            <PubmedArticle>
                <MedlineCitation>
                    <PMID>12345678</PMID>
                    <Article>
                        <ArticleTitle>No Abstract Article</ArticleTitle>
                    </Article>
                </MedlineCitation>
            </PubmedArticle>
        </PubmedArticleSet>
        """
        result = parse_efetch_response(xml_response)
        
        assert "12345678" in result
        assert result["12345678"]["title"] == "No Abstract Article"
        assert result["12345678"]["abstract"] == ""

    def test_parse_multiple_articles(self) -> None:
        """Parse response with multiple articles."""
        xml_response = """<?xml version="1.0"?>
        <PubmedArticleSet>
            <PubmedArticle>
                <MedlineCitation>
                    <PMID>11111111</PMID>
                    <Article>
                        <ArticleTitle>First Article</ArticleTitle>
                        <Abstract>
                            <AbstractText>First abstract.</AbstractText>
                        </Abstract>
                    </Article>
                </MedlineCitation>
            </PubmedArticle>
            <PubmedArticle>
                <MedlineCitation>
                    <PMID>22222222</PMID>
                    <Article>
                        <ArticleTitle>Second Article</ArticleTitle>
                        <Abstract>
                            <AbstractText>Second abstract.</AbstractText>
                        </Abstract>
                    </Article>
                </MedlineCitation>
            </PubmedArticle>
        </PubmedArticleSet>
        """
        result = parse_efetch_response(xml_response)
        
        assert len(result) == 2
        assert result["11111111"]["abstract"] == "First abstract."
        assert result["22222222"]["abstract"] == "Second abstract."

    def test_parse_empty_response(self) -> None:
        """Parse empty PubmedArticleSet."""
        xml_response = """<?xml version="1.0"?>
        <PubmedArticleSet>
        </PubmedArticleSet>
        """
        result = parse_efetch_response(xml_response)
        assert result == {}


class TestNCBIClient:
    """Tests for NCBI client with caching."""

    def test_init_creates_cache(self, tmp_path: Path) -> None:
        """Client creates abstract cache on init."""
        client = NCBIClient(tmp_path)
        
        assert client.cache is not None
        assert client.batch_size > 0

    def test_fetch_uses_cache(self, tmp_path: Path) -> None:
        """Fetch returns cached abstracts without API call."""
        client = NCBIClient(tmp_path)
        
        # Pre-populate cache
        client.cache.add_abstract("12345", "Cached title", "Cached abstract", 2020, "Test Journal")
        
        result = client.fetch_abstracts(["12345"])
        
        assert "12345" in result
        assert result["12345"]["abstract"] == "Cached abstract"

    def test_fetch_only_missing_pmids(self, tmp_path: Path) -> None:
        """Fetch only queries API for uncached PMIDs."""
        client = NCBIClient(tmp_path)
        
        # Pre-populate one PMID
        client.cache.add_abstract("11111", "Cached", "Cached abstract", 2019, "J1")
        
        # Mock the API call
        mock_response = """<?xml version="1.0"?>
        <PubmedArticleSet>
            <PubmedArticle>
                <MedlineCitation>
                    <PMID>22222</PMID>
                    <Article>
                        <ArticleTitle>Fetched Title</ArticleTitle>
                        <Abstract>
                            <AbstractText>Fetched abstract.</AbstractText>
                        </Abstract>
                    </Article>
                </MedlineCitation>
            </PubmedArticle>
        </PubmedArticleSet>
        """
        
        with patch.object(client, "_fetch_batch") as mock_fetch:
            mock_fetch.return_value = parse_efetch_response(mock_response)
            
            result = client.fetch_abstracts(["11111", "22222"])
        
        # Should only fetch the missing PMID
        mock_fetch.assert_called_once()
        call_args = mock_fetch.call_args[0][0]
        assert "22222" in call_args
        assert "11111" not in call_args

    def test_fetch_saves_to_cache(self, tmp_path: Path) -> None:
        """Fetched abstracts are saved to cache."""
        client = NCBIClient(tmp_path)
        
        mock_response = """<?xml version="1.0"?>
        <PubmedArticleSet>
            <PubmedArticle>
                <MedlineCitation>
                    <PMID>12345</PMID>
                    <Article>
                        <ArticleTitle>Test Title</ArticleTitle>
                        <Abstract>
                            <AbstractText>Test abstract.</AbstractText>
                        </Abstract>
                    </Article>
                </MedlineCitation>
            </PubmedArticle>
        </PubmedArticleSet>
        """
        
        with patch.object(client, "_fetch_batch") as mock_fetch:
            mock_fetch.return_value = parse_efetch_response(mock_response)
            client.fetch_abstracts(["12345"])
        
        # Verify saved to cache
        assert client.cache.has_pmid("12345")
        cached = client.cache.get_abstract("12345")
        assert cached["abstract"] == "Test abstract."

    def test_fetch_marks_not_found(self, tmp_path: Path) -> None:
        """PMIDs not in response are marked as not found."""
        client = NCBIClient(tmp_path)
        
        # Empty response - PMID doesn't exist
        mock_response = """<?xml version="1.0"?>
        <PubmedArticleSet>
        </PubmedArticleSet>
        """
        
        with patch.object(client, "_fetch_batch") as mock_fetch:
            mock_fetch.return_value = parse_efetch_response(mock_response)
            client.fetch_abstracts(["99999"])
        
        # Should be marked as processed (not found)
        assert client.cache.is_processed("99999")

    def test_fetch_handles_empty_list(self, tmp_path: Path) -> None:
        """Fetch with empty list returns empty dict."""
        client = NCBIClient(tmp_path)
        result = client.fetch_abstracts([])
        assert result == {}

    def test_get_cached_abstracts(self, tmp_path: Path) -> None:
        """Get cached abstracts without API call."""
        client = NCBIClient(tmp_path)
        
        # Add some to cache
        client.cache.add_abstract("111", "Title 1", "Abstract 1", 2020, "J1")
        client.cache.add_abstract("222", "Title 2", "Abstract 2", 2021, "J2")
        
        result = client.get_cached_abstracts(["111", "222", "333"])
        
        assert len(result) == 2
        assert "111" in result
        assert "222" in result
        assert "333" not in result
