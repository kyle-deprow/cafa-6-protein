"""GO term dictionary and text extraction.

Provides dictionary-based extraction of GO terms from text.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

import ahocorasick_rs

logger = logging.getLogger(__name__)

# Pattern for explicit GO IDs
GO_ID_PATTERN = re.compile(r"GO:\d{7}")

# Namespace mappings
NAMESPACE_MAP = {
    "biological_process": "BP",
    "molecular_function": "MF",
    "cellular_component": "CC",
}


class GODictionary:
    """Dictionary for GO term lookup and extraction.

    Supports lookup by term name, synonyms, and GO ID.
    Uses Aho-Corasick algorithm for fast multi-pattern matching.

    Attributes:
        terms: Dict mapping GO IDs to term info.
        name_index: Dict mapping normalized names to GO IDs.
        _automaton: Aho-Corasick automaton for fast matching.
        _pattern_to_go: Mapping from pattern index to GO IDs.
    """

    def __init__(self) -> None:
        """Initialize empty dictionary."""
        self.terms: dict[str, dict[str, Any]] = {}
        self.name_index: dict[str, set[str]] = {}
        self._automaton: ahocorasick_rs.AhoCorasick | None = None
        self._patterns: list[str] = []
        self._pattern_to_go: list[set[str]] = []

    def __len__(self) -> int:
        """Return number of terms in dictionary."""
        return len(self.terms)

    @classmethod
    def from_obo(cls, obo_path: Path | str) -> GODictionary:
        """Build dictionary from OBO file.

        Args:
            obo_path: Path to GO OBO file.

        Returns:
            Populated GODictionary.
        """
        dictionary = cls()
        obo_path = Path(obo_path)

        logger.info(f"Loading GO terms from {obo_path}")

        current_term: dict[str, Any] | None = None

        with obo_path.open() as f:
            for line in f:
                line = line.strip()

                if line == "[Term]":
                    # Save previous term
                    if current_term and "id" in current_term:
                        dictionary.add_term(
                            current_term["id"],
                            current_term.get("name", ""),
                            current_term.get("namespace", ""),
                            current_term.get("synonyms", []),
                        )
                    current_term = {"synonyms": []}

                elif current_term is not None:
                    if line.startswith("id: GO:"):
                        current_term["id"] = line.split(": ", 1)[1]
                    elif line.startswith("name: "):
                        current_term["name"] = line.split(": ", 1)[1]
                    elif line.startswith("namespace: "):
                        ns = line.split(": ", 1)[1]
                        current_term["namespace"] = NAMESPACE_MAP.get(ns, ns)
                    elif line.startswith("synonym: "):
                        # Parse synonym: "synonym text" TYPE []
                        match = re.match(r'synonym: "([^"]+)"', line)
                        if match:
                            current_term["synonyms"].append(match.group(1))
                    elif line.startswith("is_obsolete: true"):
                        # Skip obsolete terms
                        current_term = None

            # Save last term
            if current_term and "id" in current_term:
                dictionary.add_term(
                    current_term["id"],
                    current_term.get("name", ""),
                    current_term.get("namespace", ""),
                    current_term.get("synonyms", []),
                )

        logger.info(f"Loaded {len(dictionary)} GO terms")

        # Build Aho-Corasick automaton for fast matching
        dictionary.build_automaton()

        return dictionary

    def add_term(
        self,
        go_id: str,
        name: str,
        namespace: str,
        synonyms: list[str] | None = None,
    ) -> None:
        """Add a GO term to the dictionary.

        Args:
            go_id: GO ID (e.g., "GO:0006915").
            name: Term name.
            namespace: BP, MF, or CC.
            synonyms: Optional list of synonyms.
        """
        self.terms[go_id] = {
            "name": name,
            "namespace": namespace,
            "synonyms": synonyms or [],
        }

        # Index by name
        normalized = name.lower()
        if normalized not in self.name_index:
            self.name_index[normalized] = set()
        self.name_index[normalized].add(go_id)

        # Index by synonyms
        for syn in synonyms or []:
            normalized = syn.lower()
            if normalized not in self.name_index:
                self.name_index[normalized] = set()
            self.name_index[normalized].add(go_id)

    def lookup(self, text: str) -> set[str]:
        """Lookup GO IDs for a term name or synonym.

        Args:
            text: Term name or synonym to lookup.

        Returns:
            Set of matching GO IDs.
        """
        normalized = text.lower()
        return self.name_index.get(normalized, set())

    def get_term(self, go_id: str) -> dict[str, Any] | None:
        """Get term info by GO ID.

        Args:
            go_id: GO ID to lookup.

        Returns:
            Term info dict or None if not found.
        """
        return self.terms.get(go_id)

    def get_terms_by_namespace(self, namespace: str) -> set[str]:
        """Get all GO IDs for a namespace.

        Args:
            namespace: Namespace (BP, MF, CC).

        Returns:
            Set of GO IDs in that namespace.
        """
        return {go_id for go_id, info in self.terms.items() if info["namespace"] == namespace}

    def get_all_searchable_terms(self) -> dict[str, set[str]]:
        """Get all searchable terms (names + synonyms) mapped to GO IDs.

        Returns:
            Dict mapping searchable text to GO ID sets.
        """
        return dict(self.name_index)

    def build_automaton(self) -> None:
        """Build Aho-Corasick automaton for fast matching.

        Called automatically after loading from OBO.
        """
        self._patterns = list(self.name_index.keys())
        self._pattern_to_go = [self.name_index[p] for p in self._patterns]

        if self._patterns:
            self._automaton = ahocorasick_rs.AhoCorasick(
                self._patterns,
                matchkind=ahocorasick_rs.MATCHKIND_STANDARD,
            )

        logger.info(f"Built automaton with {len(self._patterns)} patterns")

    def find_matches(self, text: str) -> set[str]:
        """Find all GO term matches in text using Aho-Corasick.

        Args:
            text: Text to search (should be lowercased).

        Returns:
            Set of matched GO IDs.
        """
        if not self._automaton or not text:
            return set()

        result: set[str] = set()

        # Find all matches
        for match in self._automaton.find_matches_as_indexes(text):
            pattern_idx, start, end = match
            # Check word boundaries
            is_word_start = start == 0 or not text[start - 1].isalnum()
            is_word_end = end == len(text) or not text[end].isalnum()

            if is_word_start and is_word_end:
                result.update(self._pattern_to_go[pattern_idx])

        return result


def extract_go_terms(text: str, dictionary: GODictionary) -> set[str]:
    """Extract GO terms from text.

    Looks for:
    1. Explicit GO:XXXXXXX patterns
    2. Term names and synonyms from dictionary (using Aho-Corasick if available)

    Args:
        text: Text to search.
        dictionary: GO term dictionary.

    Returns:
        Set of extracted GO IDs.
    """
    if not text:
        return set()

    result: set[str] = set()
    text_lower = text.lower()

    # 1. Extract explicit GO IDs
    for match in GO_ID_PATTERN.finditer(text):
        go_id = match.group()
        if go_id in dictionary.terms:
            result.add(go_id)

    # 2. Use Aho-Corasick for fast multi-pattern matching if available
    if dictionary._automaton is not None:
        result.update(dictionary.find_matches(text_lower))
    else:
        # Fallback to regex (slower, used in tests with small dictionaries)
        for term_text, go_ids in dictionary.name_index.items():
            pattern = r"\b" + re.escape(term_text) + r"\b"
            if re.search(pattern, text_lower):
                result.update(go_ids)

    return result


class GOExtractor:
    """High-level GO term extractor from abstracts.

    Combines dictionary and extraction for batch processing.
    """

    def __init__(self, dictionary: GODictionary) -> None:
        """Initialize extractor with dictionary.

        Args:
            dictionary: GO term dictionary.
        """
        self.dictionary = dictionary

    @classmethod
    def from_obo(cls, obo_path: Path | str) -> GOExtractor:
        """Create extractor from OBO file.

        Args:
            obo_path: Path to GO OBO file.

        Returns:
            Initialized GOExtractor.
        """
        dictionary = GODictionary.from_obo(obo_path)
        return cls(dictionary)

    def extract(self, text: str) -> set[str]:
        """Extract GO terms from text.

        Args:
            text: Text to search.

        Returns:
            Set of extracted GO IDs.
        """
        return extract_go_terms(text, self.dictionary)

    def extract_from_abstract(
        self,
        title: str,
        abstract: str,
    ) -> set[str]:
        """Extract GO terms from article title and abstract.

        Args:
            title: Article title.
            abstract: Abstract text.

        Returns:
            Set of extracted GO IDs.
        """
        combined = f"{title} {abstract}"
        return self.extract(combined)

    def extract_batch(
        self,
        texts: list[str],
    ) -> list[set[str]]:
        """Extract GO terms from multiple texts.

        Args:
            texts: List of texts to search.

        Returns:
            List of GO ID sets (one per text).
        """
        return [self.extract(text) for text in texts]
