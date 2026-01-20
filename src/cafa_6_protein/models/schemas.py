"""Pydantic models for the retrieval-augmented prediction pipeline.

Provides strongly-typed data structures for:
- Neighbor evidence from k-NN retrieval
- Aggregation configuration
- Prediction results
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from numpy.typing import NDArray


class NeighborEvidence(BaseModel):
    """Evidence from a single neighbor protein in k-NN retrieval.

    Attributes:
        protein_id: UniProt accession of the neighbor.
        similarity: Similarity score (0-1, higher = more similar).
        go_terms: GO terms from training annotations.
        pmids: PubMed IDs associated with this protein.
        literature_terms: GO terms extracted from associated abstracts.
    """

    protein_id: str = Field(..., description="UniProt accession of neighbor")
    similarity: float = Field(..., ge=0.0, le=1.0, description="Similarity score")
    go_terms: set[str] = Field(default_factory=set, description="GO terms from annotations")
    pmids: set[str] = Field(default_factory=set, description="Associated PubMed IDs")
    literature_terms: set[str] = Field(default_factory=set, description="GO terms from literature")

    model_config = {"frozen": False, "arbitrary_types_allowed": True}


class AggregationConfig(BaseModel):
    """Configuration for score aggregation in retrieval-augmented prediction.

    Attributes:
        alpha: Weight for label-based scores (1-alpha for literature).
        literature_discount: Discount factor for literature-extracted terms.
        min_score_threshold: Minimum score to include in predictions.
        normalize_scores: Whether to normalize final scores to [0, 1].
        propagate_ancestors: Whether to propagate scores to ancestor terms.
        k: Number of neighbors to retrieve.
    """

    alpha: float = Field(
        default=0.7, ge=0.0, le=1.0, description="Weight for annotation vs literature"
    )
    literature_discount: float = Field(
        default=0.5, ge=0.0, le=1.0, description="Discount for literature scores"
    )
    min_score_threshold: float = Field(
        default=0.01, ge=0.0, le=1.0, description="Minimum score threshold"
    )
    normalize_scores: bool = Field(default=True, description="Whether to normalize to [0,1]")
    propagate_ancestors: bool = Field(
        default=True, description="Whether to propagate to GO ancestors"
    )
    k: int = Field(default=50, ge=1, le=1000, description="Number of neighbors")
    min_literature_ic: float = Field(
        default=2.0, ge=0.0, description="Minimum IC for literature-extracted terms"
    )
    weight_by_ic: bool = Field(default=True, description="Whether to weight literature terms by IC")
    literature_namespaces: frozenset[str] = Field(
        default=frozenset(["CC"]),
        description="GO namespaces to use for literature extraction (CC, MF, BP)",
    )

    model_config = {"frozen": True}


class GOPrediction(BaseModel):
    """A single GO term prediction for a protein.

    Attributes:
        protein_id: UniProt accession of the protein.
        go_term: GO term ID (e.g., GO:0008150).
        score: Prediction confidence score (0-1).
    """

    protein_id: str = Field(..., description="UniProt accession")
    go_term: str = Field(..., pattern=r"^GO:\d{7}$", description="GO term ID")
    score: float = Field(..., ge=0.0, le=1.0, description="Confidence score")

    model_config = {"frozen": True}


class ProteinPredictions(BaseModel):
    """All GO term predictions for a single protein.

    Attributes:
        protein_id: UniProt accession.
        predictions: List of (GO term, score) predictions.
        neighbors_used: Number of neighbors used for prediction.
    """

    protein_id: str = Field(..., description="UniProt accession")
    predictions: dict[str, float] = Field(
        default_factory=dict, description="GO term -> score mapping"
    )
    neighbors_used: int = Field(default=0, ge=0, description="Neighbors used")

    model_config = {"frozen": False}

    def to_dataframe_rows(self) -> list[dict[str, str | float]]:
        """Convert to list of dicts for DataFrame creation."""
        return [
            {"protein_id": self.protein_id, "go_term": term, "score": score}
            for term, score in self.predictions.items()
        ]


class EmbeddingVector(BaseModel):
    """A protein embedding vector.

    Attributes:
        protein_id: UniProt accession.
        embedding: The embedding vector (1024-dim for T5).
    """

    protein_id: str = Field(..., description="UniProt accession")
    embedding: list[float] = Field(..., min_length=1, description="Embedding vector")

    model_config = {"frozen": True, "arbitrary_types_allowed": True}

    @property
    def as_numpy(self) -> NDArray[np.float32]:
        """Get embedding as numpy array."""
        return np.array(self.embedding, dtype=np.float32)


class RetrievalResult(BaseModel):
    """Result of k-NN retrieval for a query protein.

    Attributes:
        query_protein_id: The protein being queried.
        neighbors: List of neighbor evidence.
        config: Configuration used for retrieval.
    """

    query_protein_id: str = Field(..., description="Query protein ID")
    neighbors: list[NeighborEvidence] = Field(
        default_factory=list, description="Retrieved neighbors"
    )
    config: AggregationConfig = Field(
        default_factory=AggregationConfig, description="Aggregation config"
    )

    model_config = {"frozen": False}

    def get_aggregated_scores(self) -> dict[str, float]:
        """Aggregate GO term scores from all neighbors.

        Returns:
            Dict mapping GO terms to aggregated scores.
        """
        from collections import defaultdict

        scores: dict[str, float] = defaultdict(float)

        for neighbor in self.neighbors:
            sim = neighbor.similarity

            # Label-based scores
            for go_term in neighbor.go_terms:
                scores[go_term] += self.config.alpha * sim

            # Literature-based scores
            for go_term in neighbor.literature_terms:
                scores[go_term] += (1 - self.config.alpha) * sim * self.config.literature_discount

        # Normalize
        if self.config.normalize_scores and scores:
            max_score = max(scores.values())
            if max_score > 0:
                scores = {t: s / max_score for t, s in scores.items()}

        # Threshold
        scores = {t: s for t, s in scores.items() if s >= self.config.min_score_threshold}

        return dict(scores)


class PubMedAbstract(BaseModel):
    """A PubMed abstract with metadata.

    Attributes:
        pmid: PubMed ID.
        title: Article title.
        abstract: Abstract text.
        authors: List of author names.
        journal: Journal name.
        year: Publication year.
    """

    pmid: str = Field(..., description="PubMed ID")
    title: str = Field(default="", description="Article title")
    abstract: str = Field(default="", description="Abstract text")
    authors: list[str] = Field(default_factory=list, description="Author names")
    journal: str = Field(default="", description="Journal name")
    year: int | None = Field(default=None, ge=1900, le=2100, description="Publication year")

    model_config = {"frozen": True}


class PublicationMapping(BaseModel):
    """Mapping of a protein to its publications.

    Attributes:
        protein_id: UniProt accession.
        pmids: Set of PubMed IDs.
    """

    protein_id: str = Field(..., description="UniProt accession")
    pmids: set[str] = Field(default_factory=set, description="PubMed IDs")

    model_config = {"frozen": False}


class PublicationCacheStats(BaseModel):
    """Statistics for the PublicationCache.

    Attributes:
        total_proteins: Total number of proteins in cache.
        total_pmids: Total number of unique PMIDs.
        proteins_with_pubs: Number of proteins that have publications.
    """

    total_proteins: int = Field(..., ge=0, description="Total proteins in cache")
    total_pmids: int = Field(..., ge=0, description="Total unique PMIDs")
    proteins_with_pubs: int = Field(..., ge=0, description="Proteins with publications")

    model_config = {"frozen": True}


class AbstractCacheStats(BaseModel):
    """Statistics for the AbstractCache.

    Attributes:
        total_abstracts: Total number of abstracts in cache.
        total_processed: Total PMIDs that have been processed.
        not_found: Number of PMIDs marked as not found.
    """

    total_abstracts: int = Field(..., ge=0, description="Total abstracts in cache")
    total_processed: int = Field(..., ge=0, description="Total PMIDs processed")
    not_found: int = Field(..., ge=0, description="PMIDs not found")

    model_config = {"frozen": True}


class AbstractData(BaseModel):
    """Abstract data retrieved from PubMed cache.

    Attributes:
        pmid: PubMed ID.
        title: Article title.
        abstract: Abstract text.
        pub_year: Publication year.
        journal: Journal name.
        fetched_at: Timestamp when fetched.
    """

    pmid: str = Field(..., description="PubMed ID")
    title: str = Field(default="", description="Article title")
    abstract: str = Field(default="", description="Abstract text")
    pub_year: int | None = Field(default=None, ge=1900, le=2100, description="Publication year")
    journal: str | None = Field(default=None, description="Journal name")
    fetched_at: str | None = Field(default=None, description="When the abstract was fetched")

    model_config = {"frozen": True}

    @classmethod
    def from_db_row(cls, row: dict[str, Any]) -> AbstractData:
        """Create from a database row dict."""
        return cls(
            pmid=row["pmid"],
            title=row.get("title", ""),
            abstract=row.get("abstract", ""),
            pub_year=row.get("pub_year"),
            journal=row.get("journal"),
            fetched_at=str(row["fetched_at"]) if row.get("fetched_at") else None,
        )


class GOTermInfo(BaseModel):
    """Information about a Gene Ontology term.

    Attributes:
        name: Term name.
        namespace: Namespace (BP, MF, or CC).
        synonyms: List of synonyms.
    """

    name: str = Field(..., description="Term name")
    namespace: str = Field(..., description="Namespace (BP, MF, CC)")
    synonyms: list[str] = Field(default_factory=list, description="Synonyms")

    model_config = {"frozen": True}


class KNNPredictorParams(BaseModel):
    """Parameters for KNN predictor.

    Attributes:
        k: Number of neighbors.
        use_faiss: Whether FAISS is used.
        embedding_dim: Dimension of embeddings.
        n_proteins: Number of training proteins.
    """

    k: int = Field(..., ge=1, description="Number of neighbors")
    use_faiss: bool = Field(..., description="Whether FAISS is used")
    embedding_dim: int = Field(..., ge=0, description="Embedding dimension")
    n_proteins: int = Field(..., ge=0, description="Number of training proteins")

    model_config = {"frozen": True}


class RetrievalPredictorParams(BaseModel):
    """Parameters for Retrieval-Augmented predictor.

    Attributes:
        k: Number of neighbors.
        alpha: Weight for annotation vs literature.
        literature_discount: Discount for literature scores.
        min_score_threshold: Minimum score threshold.
        normalize_scores: Whether to normalize scores.
        propagate_ancestors: Whether to propagate to ancestors.
        has_literature: Whether literature data is available.
        has_ontology: Whether ontology is available.
    """

    k: int = Field(..., ge=1, description="Number of neighbors")
    alpha: float = Field(..., ge=0.0, le=1.0, description="Annotation vs literature weight")
    literature_discount: float = Field(..., ge=0.0, le=1.0, description="Literature discount")
    min_score_threshold: float = Field(..., ge=0.0, le=1.0, description="Min score threshold")
    normalize_scores: bool = Field(..., description="Whether to normalize scores")
    propagate_ancestors: bool = Field(..., description="Whether to propagate to ancestors")
    has_literature: bool = Field(..., description="Whether literature data is available")
    has_ontology: bool = Field(..., description="Whether ontology is available")

    model_config = {"frozen": True}


class NCBIClientStats(BaseModel):
    """Statistics for the NCBI client (wraps AbstractCacheStats).

    Attributes:
        total_abstracts: Total number of abstracts in cache.
        total_processed: Total PMIDs that have been processed.
        not_found: Number of PMIDs marked as not found.
    """

    total_abstracts: int = Field(..., ge=0, description="Total abstracts in cache")
    total_processed: int = Field(..., ge=0, description="Total PMIDs processed")
    not_found: int = Field(..., ge=0, description="PMIDs not found")

    model_config = {"frozen": True}


class UniProtClientStats(BaseModel):
    """Statistics for the UniProt client (wraps PublicationCacheStats).

    Attributes:
        total_proteins: Total number of proteins in cache.
        total_pmids: Total number of unique PMIDs.
        proteins_with_pubs: Number of proteins that have publications.
    """

    total_proteins: int = Field(..., ge=0, description="Total proteins in cache")
    total_pmids: int = Field(..., ge=0, description="Total unique PMIDs")
    proteins_with_pubs: int = Field(..., ge=0, description="Proteins with publications")

    model_config = {"frozen": True}


class NeighborExplanation(BaseModel):
    """Explanation of a neighbor protein for interpretability.

    Attributes:
        protein_id: UniProt accession of the neighbor.
        similarity: Similarity score (0-1).
        num_go_terms: Number of GO terms from annotations.
        num_pmids: Number of associated PubMed IDs.
        num_literature_terms: Number of GO terms from literature.
        top_go_terms: Top GO terms from annotations.
    """

    protein_id: str = Field(..., description="UniProt accession of neighbor")
    similarity: float = Field(..., ge=0.0, le=1.0, description="Similarity score")
    num_go_terms: int = Field(..., ge=0, description="Number of GO terms from annotations")
    num_pmids: int = Field(..., ge=0, description="Number of associated PubMed IDs")
    num_literature_terms: int = Field(..., ge=0, description="Number of GO terms from literature")
    top_go_terms: list[str] = Field(default_factory=list, description="Top GO terms")

    model_config = {"frozen": True}
