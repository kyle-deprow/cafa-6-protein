"""GO term dictionary and text extraction.

Provides dictionary-based extraction of GO terms from text.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

import ahocorasick_rs

from cafa_6_protein.models.schemas import GOTermInfo

logger = logging.getLogger(__name__)

# Pattern for explicit GO IDs
GO_ID_PATTERN = re.compile(r"GO:\d{7}")

# Namespace mappings
NAMESPACE_MAP = {
    "biological_process": "BP",
    "molecular_function": "MF",
    "cellular_component": "CC",
}

# Default minimum pattern length to avoid short abbreviations matching everywhere
DEFAULT_MIN_PATTERN_LENGTH = 5

# Overly generic terms that match too broadly in scientific text
# These are typically very high-level GO terms or common English words
GENERIC_STOPWORDS = frozenset(
    [
        # Very generic single-word terms
        "binding",
        "membrane",
        "activity",
        "process",
        "signaling",
        "growth",
        "response",
        "regulation",
        "transport",
        "cell",
        "protein",
        "development",
        # Common words that happen to be GO terms
        "nucleus",
        "cytoplasm",
        "receptor",
        "complex",
        "localization",
        "assembly",
        "organization",
        "modification",
    ]
)


class GODictionary:
    """Dictionary for GO term lookup and extraction.

    Supports lookup by term name, synonyms, and GO ID.
    Uses Aho-Corasick algorithm for fast multi-pattern matching.

    Attributes:
        terms: Dict mapping GO IDs to term info.
        name_index: Dict mapping normalized names to GO IDs.
        _automaton: Aho-Corasick automaton for fast matching.
        _pattern_to_go: Mapping from pattern index to GO IDs.
        min_pattern_length: Minimum length for patterns to include.
        use_stopwords: Whether to filter out generic stopwords.
    """

    def __init__(
        self,
        min_pattern_length: int = DEFAULT_MIN_PATTERN_LENGTH,
        use_stopwords: bool = True,
    ) -> None:
        """Initialize empty dictionary.

        Args:
            min_pattern_length: Minimum characters for a pattern to be indexed.
            use_stopwords: Whether to filter out generic stopwords.
        """
        self.terms: dict[str, GOTermInfo] = {}
        self.name_index: dict[str, set[str]] = {}
        self._automaton: ahocorasick_rs.AhoCorasick | None = None
        self._patterns: list[str] = []
        self._pattern_to_go: list[set[str]] = []
        self.min_pattern_length = min_pattern_length
        self.use_stopwords = use_stopwords
        self._filtered_count = 0

    def __len__(self) -> int:
        """Return number of terms in dictionary."""
        return len(self.terms)

    @classmethod
    def from_obo(
        cls,
        obo_path: Path | str,
        min_pattern_length: int = DEFAULT_MIN_PATTERN_LENGTH,
        use_stopwords: bool = True,
    ) -> GODictionary:
        """Build dictionary from OBO file.

        Args:
            obo_path: Path to GO OBO file.
            min_pattern_length: Minimum characters for patterns (filters abbreviations).
            use_stopwords: Whether to filter out generic stopwords.

        Returns:
            Populated GODictionary.
        """
        dictionary = cls(
            min_pattern_length=min_pattern_length,
            use_stopwords=use_stopwords,
        )
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

        logger.info(
            f"Loaded {len(dictionary)} GO terms, "
            f"{len(dictionary.name_index)} indexed patterns "
            f"(filtered {dictionary._filtered_count} short/generic patterns)"
        )

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
        self.terms[go_id] = GOTermInfo(
            name=name,
            namespace=namespace,
            synonyms=synonyms or [],
        )

        # Index by name (if not filtered)
        normalized = name.lower()
        if self._should_index_pattern(normalized):
            if normalized not in self.name_index:
                self.name_index[normalized] = set()
            self.name_index[normalized].add(go_id)
        else:
            self._filtered_count += 1

        # Index by synonyms (if not filtered)
        for syn in synonyms or []:
            normalized = syn.lower()
            if self._should_index_pattern(normalized):
                if normalized not in self.name_index:
                    self.name_index[normalized] = set()
                self.name_index[normalized].add(go_id)
            else:
                self._filtered_count += 1

    def _should_index_pattern(self, pattern: str) -> bool:
        """Check if a pattern should be indexed based on filtering rules.

        Args:
            pattern: Lowercased pattern to check.

        Returns:
            True if pattern should be indexed.
        """
        # Filter by minimum length
        if len(pattern) < self.min_pattern_length:
            return False

        # Filter stopwords
        if self.use_stopwords and pattern in GENERIC_STOPWORDS:
            return False

        return True

    def lookup(self, text: str) -> set[str]:
        """Lookup GO IDs for a term name or synonym.

        Args:
            text: Term name or synonym to lookup.

        Returns:
            Set of matching GO IDs.
        """
        normalized = text.lower()
        return self.name_index.get(normalized, set())

    def get_term(self, go_id: str) -> GOTermInfo | None:
        """Get term info by GO ID.

        Args:
            go_id: GO ID to lookup.

        Returns:
            GOTermInfo or None if not found.
        """
        return self.terms.get(go_id)

    def get_terms_by_namespace(self, namespace: str) -> set[str]:
        """Get all GO IDs for a namespace.

        Args:
            namespace: Namespace (BP, MF, CC).

        Returns:
            Set of GO IDs in that namespace.
        """
        return {go_id for go_id, info in self.terms.items() if info.namespace == namespace}

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
    def from_obo(
        cls,
        obo_path: Path | str,
        min_pattern_length: int = DEFAULT_MIN_PATTERN_LENGTH,
        use_stopwords: bool = True,
    ) -> GOExtractor:
        """Create extractor from OBO file.

        Args:
            obo_path: Path to GO OBO file.
            min_pattern_length: Minimum characters for patterns (filters abbreviations).
            use_stopwords: Whether to filter out generic stopwords.

        Returns:
            Initialized GOExtractor.
        """
        dictionary = GODictionary.from_obo(
            obo_path,
            min_pattern_length=min_pattern_length,
            use_stopwords=use_stopwords,
        )
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
