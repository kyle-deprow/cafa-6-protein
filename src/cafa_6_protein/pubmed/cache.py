"""Cache layer for PubMed data.

Provides persistent caching for:
- PublicationCache: Protein → PMID mappings (Parquet)
- AbstractCache: PMID → Abstract text (SQLite)
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from cafa_6_protein.models.schemas import (
    AbstractCacheStats,
    AbstractData,
    PublicationCacheStats,
)

if TYPE_CHECKING:
    from collections.abc import Iterator


class PublicationCache:
    """Cache for protein → publication (PMID) mappings.

    Uses Parquet for efficient storage and fast protein lookups.

    Attributes:
        cache_dir: Directory where cache files are stored.
    """

    CACHE_FILE = "publications.parquet"
    META_FILE = "publications_meta.json"

    def __init__(self, cache_dir: Path | str) -> None:
        """Initialize the publication cache.

        Args:
            cache_dir: Directory for cache files. Created if doesn't exist.
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self._cache_path = self.cache_dir / self.CACHE_FILE
        self._meta_path = self.cache_dir / self.META_FILE

        # In-memory index: protein_id -> set of PMIDs
        self._protein_to_pmids: dict[str, set[str]] = {}

        # Load existing cache
        self._load()

    def _load(self) -> None:
        """Load cache from disk if it exists."""
        if not self._cache_path.exists():
            return

        try:
            df = pd.read_parquet(self._cache_path)
            for protein_id, group in df.groupby("protein_id"):
                self._protein_to_pmids[str(protein_id)] = set(group["pmid"].dropna())
        except Exception:
            # Corrupted cache, start fresh
            self._protein_to_pmids = {}

    def save(self) -> None:
        """Save cache to disk."""
        if not self._protein_to_pmids:
            return

        # Build DataFrame
        rows: list[dict[str, str | None]] = []
        for protein_id, pmids in self._protein_to_pmids.items():
            if pmids:
                for pmid in pmids:
                    rows.append({"protein_id": protein_id, "pmid": pmid})
            else:
                # Track proteins with no publications
                rows.append({"protein_id": protein_id, "pmid": None})

        df = pd.DataFrame(rows)
        df.to_parquet(self._cache_path, index=False)

        # Save metadata
        meta = {
            "total_proteins": len(self._protein_to_pmids),
            "total_pmids": len(self.get_all_pmids()),
            "last_updated": datetime.now().isoformat(),
        }
        self._meta_path.write_text(json.dumps(meta, indent=2))

    def add_publications(self, protein_id: str, pmids: list[str]) -> None:
        """Add publication mappings for a protein.

        Args:
            protein_id: UniProt protein accession.
            pmids: List of PubMed IDs associated with the protein.
        """
        self._protein_to_pmids[protein_id] = set(pmids)

    def get_pmids(self, protein_id: str) -> set[str]:
        """Get PMIDs for a protein.

        Args:
            protein_id: UniProt protein accession.

        Returns:
            Set of PMIDs, empty if protein not in cache.
        """
        return self._protein_to_pmids.get(protein_id, set())

    def get_all_pmids(self) -> set[str]:
        """Get all unique PMIDs across all proteins.

        Returns:
            Set of all cached PMIDs.
        """
        all_pmids: set[str] = set()
        for pmids in self._protein_to_pmids.values():
            all_pmids.update(pmids)
        return all_pmids

    def has_protein(self, protein_id: str) -> bool:
        """Check if protein is in cache.

        Args:
            protein_id: UniProt protein accession.

        Returns:
            True if protein has been queried (even if no pubs found).
        """
        return protein_id in self._protein_to_pmids

    def get_missing_proteins(self, protein_ids: list[str]) -> list[str]:
        """Get proteins not yet in cache.

        Args:
            protein_ids: List of protein IDs to check.

        Returns:
            List of protein IDs not in cache.
        """
        return [p for p in protein_ids if p not in self._protein_to_pmids]

    def stats(self) -> PublicationCacheStats:
        """Get cache statistics.

        Returns:
            PublicationCacheStats with cache statistics.
        """
        proteins_with_pubs = sum(1 for pmids in self._protein_to_pmids.values() if pmids)
        return PublicationCacheStats(
            total_proteins=len(self._protein_to_pmids),
            total_pmids=len(self.get_all_pmids()),
            proteins_with_pubs=proteins_with_pubs,
        )


class AbstractCache:
    """Cache for PMID → abstract text mappings.

    Uses SQLite for efficient storage and querying.

    Attributes:
        cache_dir: Directory where the database is stored.
    """

    DB_FILE = "abstracts.db"

    def __init__(self, cache_dir: Path | str) -> None:
        """Initialize the abstract cache.

        Args:
            cache_dir: Directory for database file. Created if doesn't exist.
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self._db_path = self.cache_dir / self.DB_FILE
        self._conn: sqlite3.Connection | None = None

        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        """Get database connection (lazy initialization)."""
        if self._conn is None:
            self._conn = sqlite3.connect(self._db_path)
            self._conn.row_factory = sqlite3.Row
        return self._conn

    def _init_db(self) -> None:
        """Initialize database schema."""
        conn = self._get_conn()
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS abstracts (
                pmid TEXT PRIMARY KEY,
                title TEXT,
                abstract TEXT,
                pub_year INTEGER,
                journal TEXT,
                fetched_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS fetch_status (
                pmid TEXT PRIMARY KEY,
                status TEXT,
                error_msg TEXT,
                checked_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            CREATE INDEX IF NOT EXISTS idx_abstracts_fetched
                ON abstracts(fetched_at);
        """)
        conn.commit()

    def add_abstract(
        self,
        pmid: str,
        title: str,
        abstract: str,
        pub_year: int | None,
        journal: str | None,
        *,
        commit: bool = True,
    ) -> None:
        """Add an abstract to the cache.

        Args:
            pmid: PubMed ID.
            title: Article title.
            abstract: Abstract text.
            pub_year: Publication year.
            journal: Journal name.
            commit: Whether to commit immediately (set False for batching).
        """
        conn = self._get_conn()
        conn.execute(
            """
            INSERT OR REPLACE INTO abstracts
                (pmid, title, abstract, pub_year, journal, fetched_at)
            VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            """,
            (pmid, title, abstract, pub_year, journal),
        )
        conn.execute(
            """
            INSERT OR REPLACE INTO fetch_status (pmid, status, checked_at)
            VALUES (?, 'success', CURRENT_TIMESTAMP)
            """,
            (pmid,),
        )
        if commit:
            conn.commit()

    def add_abstracts_batch(
        self,
        abstracts: list[AbstractData],
    ) -> None:
        """Add multiple abstracts in a single transaction.

        More efficient than individual add_abstract calls.

        Args:
            abstracts: List of AbstractData to add.
        """
        if not abstracts:
            return

        conn = self._get_conn()
        for data in abstracts:
            conn.execute(
                """
                INSERT OR REPLACE INTO abstracts
                    (pmid, title, abstract, pub_year, journal, fetched_at)
                VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                """,
                (data.pmid, data.title, data.abstract, data.pub_year, data.journal),
            )
            conn.execute(
                """
                INSERT OR REPLACE INTO fetch_status (pmid, status, checked_at)
                VALUES (?, 'success', CURRENT_TIMESTAMP)
                """,
                (data.pmid,),
            )
        conn.commit()

    def mark_not_found_batch(self, pmids: list[str]) -> None:
        """Mark multiple PMIDs as not found in a single transaction.

        Args:
            pmids: List of PMIDs that were not found.
        """
        if not pmids:
            return

        conn = self._get_conn()
        for pmid in pmids:
            conn.execute(
                """
                INSERT OR REPLACE INTO fetch_status (pmid, status, error_msg, checked_at)
                VALUES (?, 'not_found', NULL, CURRENT_TIMESTAMP)
                """,
                (pmid,),
            )
        conn.commit()

    def commit(self) -> None:
        """Manually commit pending changes."""
        conn = self._get_conn()
        conn.commit()

    def get_abstract(self, pmid: str) -> AbstractData | None:
        """Get abstract data for a PMID.

        Args:
            pmid: PubMed ID.

        Returns:
            AbstractData with abstract data, or None if not found.
        """
        conn = self._get_conn()
        row = conn.execute("SELECT * FROM abstracts WHERE pmid = ?", (pmid,)).fetchone()

        if row is None:
            return None

        return AbstractData.from_db_row(dict(row))

    def get_abstracts(self, pmids: list[str]) -> dict[str, AbstractData]:
        """Get multiple abstracts at once.

        Args:
            pmids: List of PubMed IDs.

        Returns:
            Dict mapping PMIDs to AbstractData (missing PMIDs excluded).
        """
        if not pmids:
            return {}

        conn = self._get_conn()
        placeholders = ",".join("?" * len(pmids))
        rows = conn.execute(
            f"SELECT * FROM abstracts WHERE pmid IN ({placeholders})",
            pmids,
        ).fetchall()

        return {row["pmid"]: AbstractData.from_db_row(dict(row)) for row in rows}

    def has_pmid(self, pmid: str) -> bool:
        """Check if PMID has an abstract in cache.

        Args:
            pmid: PubMed ID.

        Returns:
            True if abstract exists in cache.
        """
        conn = self._get_conn()
        row = conn.execute("SELECT 1 FROM abstracts WHERE pmid = ?", (pmid,)).fetchone()
        return row is not None

    def is_processed(self, pmid: str) -> bool:
        """Check if PMID has been processed (success or not found).

        Args:
            pmid: PubMed ID.

        Returns:
            True if PMID has been checked (even if not found).
        """
        conn = self._get_conn()
        row = conn.execute("SELECT 1 FROM fetch_status WHERE pmid = ?", (pmid,)).fetchone()
        return row is not None

    def get_missing_pmids(self, pmids: list[str]) -> list[str]:
        """Get PMIDs not yet processed.

        Args:
            pmids: List of PubMed IDs to check.

        Returns:
            List of PMIDs not in cache.
        """
        if not pmids:
            return []

        conn = self._get_conn()

        # SQLite has a limit on query variables (typically 999)
        # Batch the query to avoid "too many SQL variables" error
        batch_size = 900
        processed: set[str] = set()

        for i in range(0, len(pmids), batch_size):
            batch = pmids[i : i + batch_size]
            placeholders = ",".join("?" * len(batch))
            rows = conn.execute(
                f"SELECT pmid FROM fetch_status WHERE pmid IN ({placeholders})",
                batch,
            ).fetchall()
            processed.update(row["pmid"] for row in rows)

        return [p for p in pmids if p not in processed]

    def mark_not_found(self, pmid: str, error_msg: str | None = None) -> None:
        """Mark a PMID as not found (to avoid re-fetching).

        Args:
            pmid: PubMed ID.
            error_msg: Optional error message.
        """
        conn = self._get_conn()
        conn.execute(
            """
            INSERT OR REPLACE INTO fetch_status (pmid, status, error_msg, checked_at)
            VALUES (?, 'not_found', ?, CURRENT_TIMESTAMP)
            """,
            (pmid, error_msg),
        )
        conn.commit()

    def stats(self) -> AbstractCacheStats:
        """Get cache statistics.

        Returns:
            AbstractCacheStats with cache statistics.
        """
        conn = self._get_conn()

        abstract_count = conn.execute("SELECT COUNT(*) FROM abstracts").fetchone()[0]

        processed_count = conn.execute("SELECT COUNT(*) FROM fetch_status").fetchone()[0]

        not_found_count = conn.execute(
            "SELECT COUNT(*) FROM fetch_status WHERE status = 'not_found'"
        ).fetchone()[0]

        return AbstractCacheStats(
            total_abstracts=abstract_count,
            total_processed=processed_count,
            not_found=not_found_count,
        )

    def iter_abstracts(self) -> Iterator[AbstractData]:
        """Iterate over all abstracts.

        Yields:
            AbstractData for each cached abstract.
        """
        conn = self._get_conn()
        cursor = conn.execute("SELECT * FROM abstracts")

        for row in cursor:
            yield AbstractData.from_db_row(dict(row))

    def close(self) -> None:
        """Close database connection."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None
