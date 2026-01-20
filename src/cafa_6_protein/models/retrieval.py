"""Retrieval-Augmented Protein Function Prediction.

Combines kNN-based sequence similarity with literature evidence
for interpretable, scalable predictions.

Stage 4: Score Aggregation
- Label-based scores from training annotations (via kNN)
- Literature-based scores from PubMed abstract extraction
- Configurable alpha weighting between sources
"""

from __future__ import annotations

import logging
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from cafa_6_protein.models.schemas import (
    AggregationConfig,
    NeighborEvidence,
    NeighborExplanation,
    RetrievalPredictorParams,
)

if TYPE_CHECKING:
    import networkx as nx

    from cafa_6_protein.pubmed.cache import AbstractCache, PublicationCache
    from cafa_6_protein.pubmed.extractor import GOExtractor

logger = logging.getLogger(__name__)


def convert_distance_to_similarity(
    distances: np.ndarray,
    method: str = "inverse",
) -> np.ndarray:
    """Convert distance scores to similarity scores.

    Args:
        distances: Array of distance values (L2 distances from FAISS).
        method: Conversion method ('inverse' or 'exp').

    Returns:
        Array of similarity values in [0, 1].
    """
    if method == "inverse":
        # weight = 1 / (1 + distance)
        similarities = 1.0 / (1.0 + distances)
    elif method == "exp":
        # weight = exp(-distance)
        similarities = np.exp(-distances)
    else:
        msg = f"Unknown conversion method: {method}"
        raise ValueError(msg)

    return similarities


def aggregate_scores(
    evidence_list: list[NeighborEvidence],
    config: AggregationConfig | None = None,
    ic_values: dict[str, float] | None = None,
) -> dict[str, float]:
    """Aggregate scores from multiple neighbor evidence sources.

    Combines:
    1. Label-based scores: GO terms from training annotations
    2. Literature-based scores: GO terms from abstract extraction

    Score formula:
        score(term) = sum over neighbors of:
            alpha * similarity * (1 if term in go_terms else 0) +
            (1-alpha) * similarity * discount * ic_weight * (1 if term in literature_terms else 0)

    Where ic_weight = normalized_ic(term) if weight_by_ic else 1.0

    Args:
        evidence_list: List of NeighborEvidence from kNN retrieval.
        config: Aggregation configuration.
        ic_values: Optional IC values for weighting literature terms.

    Returns:
        Dictionary mapping GO terms to aggregated scores.
    """
    if config is None:
        config = AggregationConfig()

    scores: dict[str, float] = defaultdict(float)

    # Compute IC normalization factor for weighting (max IC ~ 10-15)
    max_ic = 10.0  # Reasonable upper bound for normalization

    for evidence in evidence_list:
        sim = evidence.similarity

        # Label-based scores (from training annotations)
        for go_term in evidence.go_terms:
            scores[go_term] += config.alpha * sim

        # Literature-based scores (from abstract extraction)
        for go_term in evidence.literature_terms:
            ic_weight = 1.0
            if config.weight_by_ic and ic_values:
                # Normalize IC to [0, 1] range and use as weight
                ic = ic_values.get(go_term, 0.0)
                ic_weight = min(ic / max_ic, 1.0)

            scores[go_term] += (1 - config.alpha) * sim * config.literature_discount * ic_weight

    # Normalize to [0, 1] if requested
    if config.normalize_scores and scores:
        max_score = max(scores.values())
        if max_score > 0:
            scores = {term: score / max_score for term, score in scores.items()}

    # Apply threshold
    scores = {t: s for t, s in scores.items() if s >= config.min_score_threshold}

    return dict(scores)


def propagate_to_ancestors(
    term_scores: dict[str, float],
    ontology: nx.DiGraph,
) -> dict[str, float]:
    """Propagate term scores to ancestor terms.

    Parent term gets max score of itself and all descendants.

    Args:
        term_scores: Dictionary mapping GO terms to scores.
        ontology: GO ontology graph.

    Returns:
        Dictionary with propagated scores.
    """
    from cafa_6_protein.data.ontology import get_ancestors

    result: dict[str, float] = {}

    for term, score in term_scores.items():
        # Add the original term
        if term not in result:
            result[term] = score
        else:
            result[term] = max(result[term], score)

        # Add all ancestors with the same score
        ancestors = get_ancestors(ontology, term)
        for ancestor in ancestors:
            if ancestor not in result:
                result[ancestor] = score
            else:
                result[ancestor] = max(result[ancestor], score)

    return result


class RetrievalAugmentedPredictor:
    """Retrieval-augmented predictor combining kNN and literature evidence.

    This is the main predictor implementing the full pipeline:
    1. Retrieve k nearest neighbors using embeddings
    2. Collect GO terms from neighbor annotations
    3. Enrich with GO terms from associated literature
    4. Aggregate scores with configurable weighting
    5. Propagate to ancestors and threshold

    Attributes:
        k: Number of neighbors to retrieve.
        config: Aggregation configuration.
        knn: Underlying kNN predictor (fitted with embeddings).
    """

    def __init__(
        self,
        k: int = 50,
        config: AggregationConfig | None = None,
        use_faiss: bool = True,
    ) -> None:
        """Initialize the retrieval-augmented predictor.

        Args:
            k: Number of neighbors to retrieve.
            config: Aggregation configuration (uses defaults if None).
            use_faiss: Whether to use FAISS for kNN (if available).
        """
        self.k = k
        self.config = config or AggregationConfig()
        self.use_faiss = use_faiss

        # Components (set during fit)
        self._index: Any = None
        self._protein_ids: list[str] = []
        self._annotations_by_protein: dict[str, set[str]] = {}
        self._embedding_dim: int = 0

        # Optional literature enrichment
        self._publication_cache: PublicationCache | None = None
        self._abstract_cache: AbstractCache | None = None
        self._go_extractor: GOExtractor | None = None

        # Information content for filtering/weighting literature terms
        self._ic_values: dict[str, float] = {}

        # GO terms by namespace for filtering (populated from extractor dictionary)
        self._terms_by_namespace: dict[str, set[str]] = {}

        # Optional ontology for propagation
        self._ontology: nx.DiGraph | None = None

    def set_literature_enrichment(
        self,
        publication_cache: PublicationCache,
        abstract_cache: AbstractCache,
        go_extractor: GOExtractor,
    ) -> RetrievalAugmentedPredictor:
        """Configure literature enrichment.

        Args:
            publication_cache: Cache for protein → PMID mappings.
            abstract_cache: Cache for PMID → abstract text.
            go_extractor: GO term extractor from text.

        Returns:
            Self for method chaining.
        """
        self._publication_cache = publication_cache
        self._abstract_cache = abstract_cache
        self._go_extractor = go_extractor

        # Build namespace index from extractor dictionary
        self._terms_by_namespace = {
            ns: go_extractor.dictionary.get_terms_by_namespace(ns) for ns in ["BP", "MF", "CC"]
        }
        logger.info(
            f"Literature namespace filter: {self.config.literature_namespaces}, "
            f"terms: BP={len(self._terms_by_namespace.get('BP', set()))}, "
            f"MF={len(self._terms_by_namespace.get('MF', set()))}, "
            f"CC={len(self._terms_by_namespace.get('CC', set()))}"
        )

        return self

    def set_ic_values(self, ic_values: dict[str, float]) -> RetrievalAugmentedPredictor:
        """Set Information Content values for filtering literature terms.

        Terms with IC below config.min_literature_ic will be filtered out.
        If config.weight_by_ic is True, literature scores are weighted by IC.

        Args:
            ic_values: Dictionary mapping GO term IDs to IC values.

        Returns:
            Self for method chaining.
        """
        self._ic_values = ic_values
        logger.info(f"Set IC values for {len(ic_values)} GO terms")
        return self

    def set_ontology(self, ontology: nx.DiGraph) -> RetrievalAugmentedPredictor:
        """Set ontology for ancestor propagation.

        Args:
            ontology: GO ontology graph.

        Returns:
            Self for method chaining.
        """
        self._ontology = ontology
        return self

    def fit(
        self,
        embeddings: dict[str, np.ndarray],
        annotations: pd.DataFrame,
    ) -> RetrievalAugmentedPredictor:
        """Fit the predictor with training embeddings and annotations.

        Args:
            embeddings: Dict mapping protein IDs to embedding vectors.
            annotations: DataFrame with 'protein_id' and 'go_term' columns.

        Returns:
            Self for method chaining.
        """
        self._protein_ids = list(embeddings.keys())

        # Build annotations index
        self._annotations_by_protein = {}
        for _, row in annotations.iterrows():
            pid = row["protein_id"]
            term = row["go_term"]
            if pid not in self._annotations_by_protein:
                self._annotations_by_protein[pid] = set()
            self._annotations_by_protein[pid].add(term)

        # Stack embeddings into matrix
        embedding_matrix = np.vstack([embeddings[pid] for pid in self._protein_ids])
        self._embedding_dim = embedding_matrix.shape[1]

        # Ensure float32 for FAISS compatibility
        embedding_matrix = embedding_matrix.astype(np.float32)

        # Build index
        if self.use_faiss:
            try:
                import faiss

                self._index = faiss.IndexFlatL2(self._embedding_dim)
                self._index.add(embedding_matrix)
                logger.info(f"Built FAISS index with {len(self._protein_ids)} proteins")
            except ImportError:
                logger.warning("FAISS not available, falling back to sklearn")
                self._build_sklearn_index(embedding_matrix)
        else:
            self._build_sklearn_index(embedding_matrix)

        return self

    def _build_sklearn_index(self, embedding_matrix: np.ndarray) -> None:
        """Build sklearn NearestNeighbors index as fallback."""
        from sklearn.neighbors import NearestNeighbors

        self._index = NearestNeighbors(
            n_neighbors=min(self.k, len(self._protein_ids)),
            metric="euclidean",
        )
        self._index.fit(embedding_matrix)

    def retrieve_neighbors(
        self,
        embedding: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Retrieve k nearest neighbors for an embedding.

        Args:
            embedding: Query embedding vector.

        Returns:
            Tuple of (distances, indices) arrays.
        """
        embedding = embedding.astype(np.float32).reshape(1, -1)
        actual_k = min(self.k, len(self._protein_ids))

        try:
            import faiss

            if isinstance(self._index, faiss.Index):
                distances, indices = self._index.search(embedding, actual_k)
                return distances[0], indices[0]
        except ImportError:
            pass

        # sklearn fallback
        distances, indices = self._index.kneighbors(embedding, n_neighbors=actual_k)
        return distances[0], indices[0]

    def build_evidence(
        self,
        distances: np.ndarray,
        indices: np.ndarray,
    ) -> list[NeighborEvidence]:
        """Build evidence from retrieved neighbors.

        Args:
            distances: Distance to each neighbor.
            indices: Indices of neighbors in training set.

        Returns:
            List of NeighborEvidence objects.
        """
        similarities = convert_distance_to_similarity(distances)
        evidence_list = []

        for sim, idx in zip(similarities, indices, strict=True):
            if idx >= len(self._protein_ids):
                continue

            protein_id = self._protein_ids[idx]
            go_terms = self._annotations_by_protein.get(protein_id, set())

            evidence = NeighborEvidence(
                protein_id=protein_id,
                similarity=float(sim),
                go_terms=go_terms,
            )

            # Enrich with literature if available
            if self._publication_cache is not None:
                evidence.pmids = self._publication_cache.get_pmids(protein_id)

            if (
                self._abstract_cache is not None
                and self._go_extractor is not None
                and evidence.pmids
            ):
                evidence.literature_terms = self._extract_literature_terms(evidence.pmids)

            evidence_list.append(evidence)

        return evidence_list

    def _extract_literature_terms(self, pmids: set[str]) -> set[str]:
        """Extract GO terms from abstracts for given PMIDs.

        Filters terms by minimum IC threshold if IC values are set.

        Args:
            pmids: Set of PubMed IDs.

        Returns:
            Set of extracted GO terms (filtered by IC if configured).
        """
        if not self._abstract_cache or not self._go_extractor:
            return set()

        terms: set[str] = set()

        for pmid in pmids:
            abstract_data = self._abstract_cache.get_abstract(pmid)
            if abstract_data:
                title = abstract_data.title or ""
                abstract = abstract_data.abstract or ""
                extracted = self._go_extractor.extract_from_abstract(title, abstract)
                terms.update(extracted)

        # Filter by namespace if configured
        if self.config.literature_namespaces and self._terms_by_namespace:
            allowed_terms: set[str] = set()
            for ns in self.config.literature_namespaces:
                allowed_terms.update(self._terms_by_namespace.get(ns, set()))
            terms = terms & allowed_terms

        # Filter by IC threshold if IC values are available
        if self._ic_values and self.config.min_literature_ic > 0:
            min_ic = self.config.min_literature_ic
            terms = {t for t in terms if self._ic_values.get(t, 0.0) >= min_ic}

        return terms

    def predict_one(
        self,
        protein_id: str,
        embedding: np.ndarray,
    ) -> pd.DataFrame:
        """Predict GO terms for a single protein.

        Args:
            protein_id: ID for the protein being predicted.
            embedding: Embedding vector for the protein.

        Returns:
            DataFrame with columns ['protein_id', 'go_term', 'score'].
        """
        # Retrieve neighbors
        distances, indices = self.retrieve_neighbors(embedding)

        # Build evidence
        evidence_list = self.build_evidence(distances, indices)

        # Aggregate scores (pass IC values for weighting if available)
        term_scores = aggregate_scores(
            evidence_list,
            self.config,
            ic_values=self._ic_values if self._ic_values else None,
        )

        # Propagate to ancestors if ontology is available
        if self.config.propagate_ancestors and self._ontology is not None:
            term_scores = propagate_to_ancestors(term_scores, self._ontology)

        # Convert to DataFrame
        if not term_scores:
            return pd.DataFrame(columns=["protein_id", "go_term", "score"])

        records = [
            {"protein_id": protein_id, "go_term": term, "score": score}
            for term, score in term_scores.items()
        ]
        return pd.DataFrame(records)

    def predict(
        self,
        embeddings: dict[str, np.ndarray],
        show_progress: bool = False,
    ) -> pd.DataFrame:
        """Predict GO terms for multiple proteins.

        Args:
            embeddings: Dict mapping protein IDs to embedding vectors.
            show_progress: Whether to show progress bar.

        Returns:
            DataFrame with columns ['protein_id', 'go_term', 'score'].
        """
        all_predictions = []

        items: Any = embeddings.items()
        if show_progress:
            try:
                from rich.progress import track

                items = track(list(items), description="Predicting...")
            except ImportError:
                pass

        for protein_id, embedding in items:
            preds = self.predict_one(protein_id, embedding)
            all_predictions.append(preds)

        if not all_predictions:
            return pd.DataFrame(columns=["protein_id", "go_term", "score"])

        return pd.concat(all_predictions, ignore_index=True)

    def get_params(self) -> RetrievalPredictorParams:
        """Get predictor parameters.

        Returns:
            RetrievalPredictorParams with predictor configuration.
        """
        return RetrievalPredictorParams(
            k=self.k,
            alpha=self.config.alpha,
            literature_discount=self.config.literature_discount,
            min_score_threshold=self.config.min_score_threshold,
            normalize_scores=self.config.normalize_scores,
            propagate_ancestors=self.config.propagate_ancestors,
            has_literature=self._publication_cache is not None,
            has_ontology=self._ontology is not None,
        )

    def get_neighbor_explanation(
        self,
        embedding: np.ndarray,
        top_n: int = 5,
    ) -> list[NeighborExplanation]:
        """Get explanation of nearest neighbors for interpretability.

        Args:
            embedding: Query embedding vector.
            top_n: Number of top neighbors to return.

        Returns:
            List of NeighborExplanation with neighbor info.
        """
        distances, indices = self.retrieve_neighbors(embedding)
        evidence_list = self.build_evidence(distances, indices)

        explanations = []
        for evidence in evidence_list[:top_n]:
            explanations.append(
                NeighborExplanation(
                    protein_id=evidence.protein_id,
                    similarity=evidence.similarity,
                    num_go_terms=len(evidence.go_terms),
                    num_pmids=len(evidence.pmids),
                    num_literature_terms=len(evidence.literature_terms),
                    top_go_terms=list(evidence.go_terms)[:10],
                )
            )

        return explanations


def create_retrieval_predictor(
    train_embeddings: dict[str, np.ndarray],
    annotations: pd.DataFrame,
    ontology: nx.DiGraph | None = None,
    publication_cache: PublicationCache | None = None,
    abstract_cache: AbstractCache | None = None,
    go_extractor: GOExtractor | None = None,
    k: int = 50,
    alpha: float = 0.7,
    literature_discount: float = 0.5,
) -> RetrievalAugmentedPredictor:
    """Factory function to create a fully configured retrieval predictor.

    Args:
        train_embeddings: Training protein embeddings.
        annotations: Training annotations DataFrame.
        ontology: GO ontology graph (optional).
        publication_cache: Publication cache (optional).
        abstract_cache: Abstract cache (optional).
        go_extractor: GO term extractor (optional).
        k: Number of neighbors.
        alpha: Weight for label-based scores.
        literature_discount: Discount for literature scores.

    Returns:
        Configured and fitted RetrievalAugmentedPredictor.
    """
    config = AggregationConfig(
        alpha=alpha,
        literature_discount=literature_discount,
    )

    predictor = RetrievalAugmentedPredictor(k=k, config=config)

    # Set up ontology
    if ontology is not None:
        predictor.set_ontology(ontology)

    # Set up literature enrichment if all components available
    if publication_cache and abstract_cache and go_extractor:
        predictor.set_literature_enrichment(
            publication_cache,
            abstract_cache,
            go_extractor,
        )

    # Fit the predictor
    predictor.fit(train_embeddings, annotations)

    return predictor


def load_retrieval_predictor_from_cache(
    data_dir: Path | str,
    k: int = 50,
    alpha: float = 0.7,
    use_literature: bool = False,
) -> RetrievalAugmentedPredictor:
    """Load a retrieval predictor from cached data.

    Args:
        data_dir: Directory containing cached embeddings and data.
        k: Number of neighbors.
        alpha: Weight for label-based scores.
        use_literature: Whether to use literature enrichment.

    Returns:
        Configured and fitted RetrievalAugmentedPredictor.
    """
    import obonet
    import pandas as pd

    data_dir = Path(data_dir)

    # Load embeddings from cache
    embeddings_file = data_dir / "embeddings.npy"
    protein_ids_file = data_dir / "embedding_protein_ids.txt"

    if not embeddings_file.exists() or not protein_ids_file.exists():
        msg = f"Embeddings not found in {data_dir}. Run 'cafa6 embeddings' first."
        raise FileNotFoundError(msg)

    import numpy as np

    embeddings_array = np.load(embeddings_file)
    with protein_ids_file.open() as f:
        protein_ids = [line.strip() for line in f if line.strip()]

    # Convert to dict
    train_embeddings = {pid: embeddings_array[i] for i, pid in enumerate(protein_ids)}
    logger.info(f"Loaded {len(train_embeddings)} embeddings from cache")

    # Load annotations
    annotations_file = data_dir / "Train" / "train_terms.tsv"
    if not annotations_file.exists():
        msg = f"Annotations not found: {annotations_file}"
        raise FileNotFoundError(msg)

    annotations = pd.read_csv(
        annotations_file,
        sep="\t",
        header=0,
        names=["protein_id", "go_term", "aspect"],
    )
    # Rename columns to match expected format
    annotations = annotations.rename(columns={"EntryID": "protein_id", "term": "go_term"})
    # Re-read with proper handling
    annotations = pd.read_csv(annotations_file, sep="\t")
    annotations = annotations.rename(columns={"EntryID": "protein_id", "term": "go_term"})
    logger.info(f"Loaded {len(annotations)} annotations")

    # Load ontology
    ontology_file = data_dir / "Train" / "go-basic.obo"
    ontology = None
    if ontology_file.exists():
        ontology = obonet.read_obo(str(ontology_file))
        logger.info(f"Loaded ontology with {len(ontology)} terms")

    # Optional: Load literature enrichment
    publication_cache = None
    abstract_cache = None
    go_extractor = None

    if use_literature:
        from cafa_6_protein.pubmed import (
            AbstractCache,
            GOExtractor,
            PublicationCache,
        )

        publications_file = data_dir / "publications.parquet"
        abstracts_file = data_dir / "abstracts.db"

        if publications_file.exists() and abstracts_file.exists():
            publication_cache = PublicationCache(data_dir)
            abstract_cache = AbstractCache(data_dir)
            # Load GO dictionary from OBO file for text extraction
            go_extractor = GOExtractor.from_obo(ontology_file)
            logger.info("Loaded literature enrichment caches")

    return create_retrieval_predictor(
        train_embeddings=train_embeddings,
        annotations=annotations,
        ontology=ontology,
        publication_cache=publication_cache,
        abstract_cache=abstract_cache,
        go_extractor=go_extractor,
        k=k,
        alpha=alpha,
    )
