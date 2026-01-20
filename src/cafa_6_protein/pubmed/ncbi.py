"""NCBI E-utilities client for fetching PubMed abstracts.

Provides batched fetching of abstracts with caching.
"""

from __future__ import annotations

import contextlib
import logging
import time
import xml.etree.ElementTree as ET
from typing import TYPE_CHECKING, Any

import requests

from cafa_6_protein.pubmed.cache import AbstractCache

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)

# NCBI E-utilities endpoints
NCBI_EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"

# Rate limiting (NCBI allows 3 requests/second without API key)
REQUESTS_PER_SECOND = 3
MIN_REQUEST_INTERVAL = 1.0 / REQUESTS_PER_SECOND


def parse_efetch_response(xml_text: str) -> dict[str, dict[str, Any]]:
    """Parse NCBI E-fetch XML response to extract abstracts.

    Args:
        xml_text: XML response from efetch.

    Returns:
        Dict mapping PMIDs to dicts with 'title', 'abstract', 'pub_year', 'journal'.
    """
    result: dict[str, dict[str, Any]] = {}

    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as e:
        logger.warning(f"Failed to parse XML response: {e}")
        return result

    for article in root.findall(".//PubmedArticle"):
        # Get PMID
        pmid_elem = article.find(".//PMID")
        if pmid_elem is None or not pmid_elem.text:
            continue
        pmid = pmid_elem.text

        # Get title
        title_elem = article.find(".//ArticleTitle")
        title = title_elem.text if title_elem is not None and title_elem.text else ""

        # Get abstract (may have multiple sections)
        abstract_parts: list[str] = []
        abstract_elem = article.find(".//Abstract")
        if abstract_elem is not None:
            for text_elem in abstract_elem.findall("AbstractText"):
                if text_elem.text:
                    label = text_elem.get("Label", "")
                    if label:
                        abstract_parts.append(f"{label}: {text_elem.text}")
                    else:
                        abstract_parts.append(text_elem.text)

        abstract = " ".join(abstract_parts)

        # Get publication year
        pub_year: int | None = None
        year_elem = article.find(".//PubDate/Year")
        if year_elem is not None and year_elem.text:
            with contextlib.suppress(ValueError):
                pub_year = int(year_elem.text)

        # Get journal name
        journal: str | None = None
        journal_elem = article.find(".//Journal/Title")
        if journal_elem is not None and journal_elem.text:
            journal = journal_elem.text

        result[pmid] = {
            "title": title,
            "abstract": abstract,
            "pub_year": pub_year,
            "journal": journal,
        }

    return result


class NCBIClient:
    """Client for fetching abstracts from NCBI PubMed.

    Handles batched requests with rate limiting and caching.

    Attributes:
        cache: Abstract cache for storing results.
        batch_size: Number of PMIDs per API request.
    """

    def __init__(
        self,
        cache_dir: Path | str,
        batch_size: int = 200,
        timeout: int = 30,
        api_key: str | None = None,
    ) -> None:
        """Initialize the NCBI client.

        Args:
            cache_dir: Directory for abstract cache.
            batch_size: PMIDs per batch request.
            timeout: Request timeout in seconds.
            api_key: Optional NCBI API key for higher rate limits.
        """
        self.cache = AbstractCache(cache_dir)
        self.batch_size = batch_size
        self.timeout = timeout
        self.api_key = api_key
        self._last_request_time = 0.0
        self._session = requests.Session()

    def _rate_limit(self) -> None:
        """Enforce rate limiting between requests."""
        # With API key, can do 10 requests/second
        interval = MIN_REQUEST_INTERVAL if not self.api_key else 0.1
        elapsed = time.time() - self._last_request_time
        if elapsed < interval:
            time.sleep(interval - elapsed)
        self._last_request_time = time.time()

    def _fetch_batch(self, pmids: list[str]) -> dict[str, dict[str, Any]]:
        """Fetch abstracts for a batch of PMIDs.

        Args:
            pmids: List of PMIDs to query.

        Returns:
            Dict mapping PMIDs to dicts with 'title', 'abstract', etc.

        Raises:
            requests.RequestException: On API errors.
        """
        if not pmids:
            return {}

        self._rate_limit()

        params: dict[str, Any] = {
            "db": "pubmed",
            "id": ",".join(pmids),
            "rettype": "xml",
            "retmode": "xml",
        }

        if self.api_key:
            params["api_key"] = self.api_key

        logger.debug(f"Fetching {len(pmids)} abstracts from NCBI")

        response = self._session.get(
            NCBI_EFETCH_URL,
            params=params,
            timeout=self.timeout,
        )
        response.raise_for_status()

        return parse_efetch_response(response.text)

    def fetch_abstracts(
        self,
        pmids: list[str],
        progress_callback: callable | None = None,
    ) -> dict[str, dict[str, Any]]:
        """Fetch abstracts for PMIDs.

        Uses cache for already-fetched PMIDs, only queries API for missing.

        Args:
            pmids: List of PMIDs.
            progress_callback: Optional callback(fetched, total) for progress.

        Returns:
            Dict mapping PMIDs to dicts with 'title', 'abstract', etc.
        """
        if not pmids:
            return {}

        result: dict[str, dict[str, str]] = {}

        # Get cached abstracts
        for pmid in pmids:
            cached = self.cache.get_abstract(pmid)
            if cached:
                result[pmid] = cached

        # Find missing PMIDs (not cached and not marked as not-found)
        missing = self.cache.get_missing_pmids(pmids)

        if not missing:
            logger.info(f"All {len(pmids)} PMIDs found in cache")
            return result

        logger.info(f"Fetching {len(missing)} abstracts from NCBI")

        # Fetch in batches
        fetched_count = 0
        for i in range(0, len(missing), self.batch_size):
            batch = missing[i : i + self.batch_size]

            try:
                batch_result = self._fetch_batch(batch)

                # Update cache and result
                for pmid, data in batch_result.items():
                    self.cache.add_abstract(
                        pmid,
                        data["title"],
                        data["abstract"],
                        data.get("pub_year"),
                        data.get("journal"),
                    )
                    result[pmid] = data

                # Mark PMIDs not in response as not found
                for pmid in batch:
                    if pmid not in batch_result:
                        self.cache.mark_not_found(pmid)

                fetched_count += len(batch)

                if progress_callback:
                    progress_callback(fetched_count, len(missing))

            except requests.RequestException as e:
                logger.warning(f"Error fetching batch: {e}")
                # Continue with next batch

        return result

    def get_cached_abstracts(
        self,
        pmids: list[str],
    ) -> dict[str, dict[str, Any]]:
        """Get abstracts from cache only (no API calls).

        Args:
            pmids: List of PMIDs.

        Returns:
            Dict mapping cached PMIDs to dicts with 'title', 'abstract', etc.
        """
        result: dict[str, dict[str, Any]] = {}

        for pmid in pmids:
            cached = self.cache.get_abstract(pmid)
            if cached:
                result[pmid] = cached

        return result

    def stats(self) -> dict[str, Any]:
        """Get cache statistics.

        Returns:
            Dict with cache statistics.
        """
        return self.cache.stats()
