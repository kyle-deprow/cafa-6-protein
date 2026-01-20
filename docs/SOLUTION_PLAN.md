# CAFA 6 Solution Plan

## Approach: Retrieval-Augmented Protein Function Prediction

A non-parametric approach that combines sequence similarity with literature evidence for interpretable, scalable predictions.

---

## Core Idea

```
Test protein
  → PLM embedding → top-k similar training proteins
  → Their GO labels (from training data)
  → Their publications (from PubMed cache)
  → GO terms extracted from literature
  → Aggregate: similarity-weighted labels + literature evidence
  → Final predictions with citation-backed confidence
```

**Why this works**:
1. **Homology signal**: Similar sequences → similar functions (well-established)
2. **Literature signal**: Papers describe functions before they're curated into GO
3. **Interpretable**: Each prediction cites similar proteins + supporting papers
4. **Scalable**: No LLM calls at inference (all local computation)
5. **Adaptive**: New knowledge added by updating caches, not retraining

---

## Dataset Summary

| Metric | Value |
|--------|-------|
| Training proteins | 82,404 |
| Training annotations | 537,027 |
| Unique GO terms | 26,125 |
| Test proteins | 224,309 |
| GO terms with IA weights | 40,122 |

**Key insight**: 70-80% of GO terms are rare (≤10 proteins). High-IA terms (rare, specific) score higher in evaluation but are hardest to predict.

---

## Architecture

### Stage 1: Embedding Index (Offline)
```python
# Build FAISS index of training protein embeddings
embeddings = load_t5_embeddings(train_proteins)  # 1024-dim per protein
index = faiss.IndexFlatIP(1024)  # Inner product for cosine sim
index.add(normalize(embeddings))
```

### Stage 2: Retrieval (Per Test Protein)
```python
def retrieve_evidence(test_protein, k=50):
    query = normalize(get_embedding(test_protein))
    distances, indices = index.search(query, k)
    
    evidence = []
    for dist, idx in zip(distances, indices):
        neighbor = train_proteins[idx]
        evidence.append({
            "protein": neighbor,
            "similarity": dist,
            "go_terms": neighbor.annotations,
            "pmids": publication_cache.get_pmids(neighbor.id),
        })
    return evidence
```

### Stage 3: Literature Enrichment
```python
def enrich_with_literature(evidence, go_extractor):
    for item in evidence:
        for pmid in item["pmids"]:
            abstract = abstract_cache.get_abstract(pmid)
            if abstract:
                extracted_terms = go_extractor.extract(abstract["title"] + " " + abstract["abstract"])
                item["literature_terms"] = extracted_terms
    return evidence
```

### Stage 4: Score Aggregation
```python
def aggregate_predictions(evidence, alpha=0.7):
    scores = defaultdict(float)
    
    for item in evidence:
        sim = item["similarity"]
        
        # Label-based scores (from training annotations)
        for go_term in item["go_terms"]:
            scores[go_term] += alpha * sim
        
        # Literature-based scores (from abstract extraction)
        for go_term in item.get("literature_terms", []):
            scores[go_term] += (1 - alpha) * sim * 0.5  # Lower weight for extracted
    
    # Normalize to [0, 1]
    max_score = max(scores.values()) if scores else 1
    return {term: score / max_score for term, score in scores.items()}
```

### Stage 5: Post-Processing
```python
def post_process(predictions, ontology):
    # Propagate to ancestors (parent ≥ max child)
    propagated = propagate_to_parents(predictions, ontology)
    
    # Apply threshold
    filtered = {t: s for t, s in propagated.items() if s >= 0.01}
    
    return filtered
```

---

## Implementation Status

### ✅ Completed
- [x] Data loaders (FASTA, TSV, OBO)
- [x] Evaluation metrics (weighted F1)
- [x] Frequency baseline predictor
- [x] kNN predictor
- [x] Ontology utilities (ancestor propagation)
- [x] CLI (`info`, `baseline`, `cv`, `validate`, `predict`)
- [x] **PubMed pipeline**:
  - [x] PublicationCache (Parquet) - protein → PMIDs (82,404 proteins, 194,771 PMIDs)
  - [x] AbstractCache (SQLite) - PMID → abstract
  - [x] UniProt client (fetch publications)
  - [x] NCBI client (fetch abstracts)
  - [x] GO extractor (Aho-Corasick, 40k terms)
  - [x] CLI command: `cafa6 pubmed`
- [x] **Score Aggregation (Stage 4)**:
  - [x] RetrievalAugmentedPredictor - full retrieval pipeline
  - [x] NeighborEvidence - structured evidence from kNN
  - [x] AggregationConfig - configurable alpha, literature discount
  - [x] aggregate_scores() - combine label + literature scores
  - [x] Ancestor propagation integration
  - [x] Factory function: create_retrieval_predictor()
- [x] **Embedding Pipeline**:
  - [x] EmbeddingDownloader - UniProt T5 embeddings (1024-dim)
  - [x] EmbeddingIndex - FAISS index for similarity search
  - [x] 82,395 training proteins with embeddings (321.9 MB)
  - [x] FAISS index built and saved
  - [x] CLI command: `cafa6 embeddings`
- [x] **Prediction Pipeline**:
  - [x] load_retrieval_predictor_from_cache() - load from cached data
  - [x] CLI command: `cafa6 predict` - generate predictions
  - [x] Literature enrichment via --literature flag
  - [x] Pydantic models for type safety

### 🔄 In Progress
- [ ] Fetch abstracts for remaining ~185k PMIDs (running in background)

### ⏳ Next Steps
- [ ] Cross-validation on retrieval system
- [ ] Tune hyperparameters (k, alpha, literature_discount)
- [ ] Ensemble with frequency baseline
- [ ] Generate final submission


---

## Data Pipeline

```
Training Data                    External APIs
     │                                │
     ▼                                ▼
train_terms.tsv              UniProt API → publications.parquet
train_sequences.fasta        NCBI API   → abstracts.db
go-basic.obo                      │
     │                            ▼
     └──────────────────> GO Extractor (Aho-Corasick)
                                  │
                                  ▼
                          Literature GO scores
                                  │
     PLM Embeddings              │
          │                      │
          ▼                      ▼
     FAISS Index ──────> Retrieval + Aggregation
                                  │
                                  ▼
                          Final Predictions
```

---

## Expected Performance

| Component | Estimated F-max | Notes |
|-----------|-----------------|-------|
| Frequency baseline | 0.22-0.30 | Current CV results |
| kNN (embeddings) | 0.42-0.72 | Validated! BP=0.43, MF=0.71, CC=0.73 |
| + Literature scores | 0.45-0.75 | PubMed evidence (in progress) |
| + Ensemble tuning | 0.47-0.78 | Weight optimization |

**Target**: Top 10 (current threshold ~0.41) - **Already exceeding for MF/CC!**

---

## Commands

```bash
# Fetch publications for training proteins
uv run cafa6 pubmed

# Fetch with abstracts (run multiple times if needed)
uv run cafa6 pubmed --fetch-abstracts --max-abstracts 50000

# Cross-validate retrieval predictor
uv run cafa6 cv-retrieval -k 50 --max-val 1000

# Cross-validate with literature enrichment
uv run cafa6 cv-retrieval -k 50 --alpha 0.7 --literature --max-val 1000

# Generate predictions for test set
uv run cafa6 predict submission.tsv --k 50

# Generate with literature
uv run cafa6 predict submission.tsv --k 50 --alpha 0.7 --literature
```

---

## Key Files

| File | Purpose |
|------|---------|
| `src/cafa_6_protein/pubmed/cache.py` | Publication + Abstract caching |
| `src/cafa_6_protein/pubmed/uniprot.py` | UniProt API client |
| `src/cafa_6_protein/pubmed/ncbi.py` | NCBI E-utilities client |
| `src/cafa_6_protein/pubmed/extractor.py` | GO term extraction from text |
| `data/publications.parquet` | Protein → PMID cache |
| `data/abstracts.db` | PMID → Abstract cache |

---

## Technical Requirements

```toml
# Core
torch>=2.0
faiss-cpu>=1.7
pandas>=2.1
pyarrow>=14.0

# Biology
obonet>=1.1
biopython>=1.83

# Text processing
ahocorasick-rs>=1.0

# APIs
requests>=2.31

# CLI
typer>=0.9
rich>=13.0
```

---

## Validation Strategy

1. **5-fold CV** with CAFA-evaluator
2. **Time-shifted holdout**: Exclude proteins with recent annotations
3. **Track per-ontology F-max**: BP, MF, CC separately
4. **High-IA term performance**: Where competition is won

---

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| API rate limits | Aggressive caching, batch requests |
| Missing embeddings | Fall back to sequence-based kNN |
| Slow extraction | Aho-Corasick is O(n) in text length |
| Novel proteins | Literature signal helps when homology fails |
