"""UniProt API client for fetching protein publication references.

Provides batched fetching of protein → PMID mappings with caching.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any

import requests

from cafa_6_protein.pubmed.cache import PublicationCache

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)

# UniProt API endpoints
UNIPROT_API_BASE = "https://rest.uniprot.org"
UNIPROT_SEARCH_URL = f"{UNIPROT_API_BASE}/uniprotkb/search"

# Rate limiting
REQUESTS_PER_SECOND = 3
MIN_REQUEST_INTERVAL = 1.0 / REQUESTS_PER_SECOND


def parse_uniprot_response(response_data: dict[str, Any]) -> dict[str, list[str]]:
    """Parse UniProt API response to extract protein → PMID mappings.

    Args:
        response_data: JSON response from UniProt search API.

    Returns:
        Dict mapping protein accessions to lists of PMIDs.
    """
    result: dict[str, list[str]] = {}

    for entry in response_data.get("results", []):
        accession = entry.get("primaryAccession", "")
        if not accession:
            continue

        pmids: list[str] = []

        for ref in entry.get("references", []):
            citation = ref.get("citation", {})
            for xref in citation.get("citationCrossReferences", []):
                if xref.get("database") == "PubMed":
                    pmid = xref.get("id")
                    if pmid:
                        pmids.append(pmid)

        result[accession] = pmids

    return result


class UniProtClient:
    """Client for fetching publication references from UniProt.

    Handles batched requests with rate limiting and caching.

    Attributes:
        cache: Publication cache for storing results.
        batch_size: Number of proteins per API request.
    """

    def __init__(
        self,
        cache_dir: Path | str,
        batch_size: int = 500,
        timeout: int = 30,
    ) -> None:
        """Initialize the UniProt client.

        Args:
            cache_dir: Directory for publication cache.
            batch_size: Proteins per batch request.
            timeout: Request timeout in seconds.
        """
        self.cache = PublicationCache(cache_dir)
        self.batch_size = batch_size
        self.timeout = timeout
        self._last_request_time = 0.0
        self._session = requests.Session()

    def _rate_limit(self) -> None:
        """Enforce rate limiting between requests."""
        elapsed = time.time() - self._last_request_time
        if elapsed < MIN_REQUEST_INTERVAL:
            time.sleep(MIN_REQUEST_INTERVAL - elapsed)
        self._last_request_time = time.time()

    def _fetch_batch(self, protein_ids: list[str]) -> dict[str, list[str]]:
        """Fetch publication references for a batch of proteins.

        Args:
            protein_ids: List of UniProt accessions to query.

        Returns:
            Dict mapping protein accessions to lists of PMIDs.

        Raises:
            requests.RequestException: On API errors.
        """
        if not protein_ids:
            return {}

        self._rate_limit()

        # Build query with proper syntax for multiple accessions
        # UniProt uses: (accession:P12345) OR (accession:P67890)
        accession_clauses = [f"accession:{pid}" for pid in protein_ids]
        query = " OR ".join(accession_clauses)

        params = {
            "query": query,
            "fields": "accession,lit_pubmed_id",
            "format": "json",
            "size": len(protein_ids),
        }

        logger.debug(f"Fetching {len(protein_ids)} proteins from UniProt")

        response = self._session.get(
            UNIPROT_SEARCH_URL,
            params=params,
            timeout=self.timeout,
        )
        response.raise_for_status()

        data = response.json()
        return parse_uniprot_response(data)

    def fetch_publications(
        self,
        protein_ids: list[str],
        progress_callback: callable | None = None,
    ) -> dict[str, set[str]]:
        """Fetch publication references for proteins.

        Uses cache for already-fetched proteins, only queries API for missing.

        Args:
            protein_ids: List of UniProt accessions.
            progress_callback: Optional callback(fetched, total) for progress.

        Returns:
            Dict mapping protein accessions to sets of PMIDs.
        """
        if not protein_ids:
            return {}

        result: dict[str, set[str]] = {}

        # Get cached proteins
        for protein_id in protein_ids:
            if self.cache.has_protein(protein_id):
                result[protein_id] = self.cache.get_pmids(protein_id)

        # Find missing proteins
        missing = self.cache.get_missing_proteins(protein_ids)

        if not missing:
            logger.info(f"All {len(protein_ids)} proteins found in cache")
            return result

        logger.info(f"Fetching {len(missing)} proteins from UniProt API")

        # Fetch in batches
        fetched_count = 0
        for i in range(0, len(missing), self.batch_size):
            batch = missing[i : i + self.batch_size]

            try:
                batch_result = self._fetch_batch(batch)

                # Update cache and result
                for protein_id, pmids in batch_result.items():
                    self.cache.add_publications(protein_id, pmids)
                    result[protein_id] = set(pmids)

                # Mark proteins not in response as having no publications
                for protein_id in batch:
                    if protein_id not in batch_result:
                        self.cache.add_publications(protein_id, [])
                        result[protein_id] = set()

                fetched_count += len(batch)

                if progress_callback:
                    progress_callback(fetched_count, len(missing))

            except requests.RequestException as e:
                logger.warning(f"Error fetching batch: {e}")
                # Continue with next batch

        # Save cache
        self.cache.save()

        return result

    def get_cached_publications(
        self,
        protein_ids: list[str],
    ) -> dict[str, set[str]]:
        """Get publications from cache only (no API calls).

        Args:
            protein_ids: List of UniProt accessions.

        Returns:
            Dict mapping cached protein accessions to sets of PMIDs.
        """
        result: dict[str, set[str]] = {}

        for protein_id in protein_ids:
            if self.cache.has_protein(protein_id):
                result[protein_id] = self.cache.get_pmids(protein_id)

        return result

    def stats(self) -> dict[str, Any]:
        """Get cache statistics.

        Returns:
            Dict with cache statistics.
        """
        return self.cache.stats()
