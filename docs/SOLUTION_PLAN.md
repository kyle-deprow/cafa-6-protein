# CAFA 6 Solution Plan

## Data Analysis Summary

### Dataset Statistics

| Metric | Value |
|--------|-------|
| Training proteins | 82,404 |
| Training annotations | 537,027 |
| Unique GO terms (in training) | 26,125 |
| GO terms with IA weights | 40,122 |
| Test superset proteins | 224,309 |
| Unique species | 1,381 |

### Sequence Characteristics

| Metric | Training | Test |
|--------|----------|------|
| Count | 82,404 | 224,309 |
| Mean length | 525.8 aa | 429.2 aa |
| Median length | 409 aa | - |
| Min length | 3 aa | 2 aa |
| Max length | 35,213 aa | 35,213 aa |

**Key Observation**: Test set is 2.7x larger than training and has slightly shorter sequences on average. This suggests the test set includes more diverse/novel proteins.

### Ontology Distribution

| Aspect | Unique Terms | Annotated Proteins | Terms/Protein | Rare Terms (≤10) |
|--------|--------------|-------------------|---------------|------------------|
| **MF** | 6,616 | 58,001 | 2.21 | 5,215 (79%) |
| **BP** | 16,858 | 59,958 | 4.18 | 11,959 (71%) |
| **CC** | 2,651 | 60,292 | 2.62 | 1,694 (64%) |

**Key Observations**:
- BP has 2.5x more terms than MF, suggesting it's the hardest to predict
- ~70-80% of terms are rare (≤10 proteins), creating a severe long-tail problem
- Most common terms: GO:0005515 (protein binding, 33,713 proteins), GO:0005634 (nucleus, 13,283)

### GO Ontology Structure

| Metric | Value |
|--------|-------|
| Total nodes | 40,122 |
| Total edges | 77,229 |
| MF terms | 10,131 |
| BP terms | 25,950 |
| CC terms | 4,041 |

### Information Accretion Weights

| Metric | Value |
|--------|-------|
| Range | 0.0 - 15.88 |
| Mean | 2.65 |
| Terms with weight 0 | 11,828 (root/common terms) |

**Implication**: High-IA terms (rare, specific) are worth more in evaluation. However, they're also hardest to predict due to limited training examples.

**Critical Insight**: The most frequent terms have near-zero IA weights:
- GO:0005515 (protein binding, 33,713 proteins): IA = 0.20
- GO:0005886 (plasma membrane, 10,150 proteins): IA = 0.03

Meanwhile, 9,049 terms have IA > 5, and 1,271 have IA > 10. **Optimizing for frequency is counterproductive** — we must optimize for IA-weighted value.

### Species Distribution (Top 10)

| TaxonID | Species | Proteins |
|---------|---------|----------|
| 9606 | Human | 17,162 (21%) |
| 10090 | Mouse | 12,508 (15%) |
| 3702 | Arabidopsis | 11,863 (14%) |
| 559292 | S. cerevisiae | 5,520 (7%) |
| 10116 | Rat | 4,909 (6%) |
| 284812 | S. pombe | 4,636 (6%) |
| 83333 | E. coli K-12 | 3,466 (4%) |
| 7227 | D. melanogaster | 3,201 (4%) |
| 6239 | C. elegans | 2,540 (3%) |
| 83332 | M. tuberculosis | 1,530 (2%) |

**Key Insight**: 50% of training data comes from just 3 species (human, mouse, Arabidopsis). Transfer learning from model organisms to less-studied species will be critical.

---

## Problem Characteristics & Challenges

### 1. Extreme Multi-Label Classification
- 26,125 possible labels (GO terms in training)
- Average protein has 6.5 labels
- Maximum: 233 labels on one protein
- Labels are hierarchically organized (DAG structure)

### 2. Severe Class Imbalance
- 79% of MF terms appear in ≤10 proteins
- Power-law distribution of term frequencies
- Most predictive signal comes from rare, high-IA terms

### 3. Hierarchical Label Structure
- Predictions must be consistent with GO DAG
- If predicting child term, must also predict all ancestors
- Parent score ≥ max(child scores)

### 4. Prospective Evaluation (Time-Shift)
- Test set = proteins gaining NEW experimental annotations after deadline
- Public LB uses a small held-out sample (not representative)
- Final evaluation happens 4+ months later
- **Implication**: Overfit to LB = likely to fail on final test

### 5. Open-World Assumption
- Absence of annotation ≠ negative label
- A protein may have functions not yet discovered
- Cannot use standard negative sampling strategies

### 6. Literature Signal is Underexploited
- Proteins gaining NEW annotations often have recent papers describing their function
- Electronic annotations (IEA) that become experimentally confirmed = easy wins
- A simple GOA UniProt mapping achieves **0.269 LB** with zero ML
- **Implication**: Literature mining and annotation transfer should be core components, not afterthoughts

---

## Critical Design Decisions

### Target Term Selection: IA-Weighted, Not Frequency-Based

**Wrong approach** (what most competitors do):
- Select top-K terms by frequency
- Problem: High-frequency terms have low IA weights → low scoring impact

**Correct approach**:
- Score each term: `selection_score = IA(term) × sqrt(frequency(term))`
- Select top-K by selection_score
- This balances learnability (need examples) with value (IA weight)

**Recommended term counts**:
- MF: Top 1000 by selection_score
- BP: Top 2000 by selection_score
- CC: Top 600 by selection_score

### GOA Baseline Must Be Ensembled Into Every Model

The 0.269 baseline from GOA mapping exploits annotation transfer:
1. Many test proteins already have electronic (IEA) annotations
2. Some of these will be experimentally confirmed during evaluation
3. Predicting these with confidence = 1.0 is often correct

**Every submission should blend**: `final_score = α × ML_score + (1-α) × GOA_score`

### Loss Function: Focal Loss > BCE

Standard BCE treats all examples equally. With 79% rare terms, this is suboptimal.

**Use Focal Loss with IA weights**:
```python
focal_loss = -α × (1 - p)^γ × log(p) × IA_weight
```
Where γ=2 focuses learning on hard examples (rare terms).

### Taxonomy: Phylogenetic Embeddings, Not One-Hot

**Problem with one-hot**: 1,381 species → sparse, no generalization between related species.

**Solution**: Encode taxonomy as path in tree of life:
1. Build phylogenetic tree from NCBI taxonomy
2. Learn 32-dim embedding for each node
3. Species embedding = mean of path embeddings from root

This allows Arabidopsis knowledge to transfer to other plants, etc.

---

## Tiered Solution Architecture

### Tier 0: GOA Baseline + kNN (Target: 0.30-0.35)
**Approach**: "Free Lunch from Existing Annotations"

**Implementation**:
```python
# Step 1: Direct GOA mapping
for protein in test_proteins:
    goa_terms = fetch_goa_annotations(protein.uniprot_id)  # IEA, ISS, etc.
    for term in goa_terms:
        predictions.append((protein, term, 1.0))

# Step 2: kNN in embedding space
for protein in test_proteins:
    neighbors = faiss_index.search(protein.t5_embedding, k=50)
    for neighbor, distance in neighbors:
        for term in neighbor.annotations:
            weight = 1.0 / (1.0 + distance)
            knn_scores[protein, term] += weight
    # Normalize scores to [0, 1]
    
# Step 3: Blend
final_score = 0.4 * goa_score + 0.6 * knn_score
```

**Why this works**:
- GOA captures annotations about to be confirmed (IEA → EXP transition)
- kNN exploits sequence similarity → function similarity
- Together they cover both known and novel proteins

**This is our safety net** — always blend into final predictions.

---

### Tier 1: PubMed Literature Mining (Target: 0.38-0.42) ⭐ PRIORITY
**Approach**: "Read the Papers Like Scientists Do"

**Key Insight**: Proteins gaining NEW experimental annotations often have recent papers. Mining PubMed abstracts can predict function before it's curated into GO.

**Pipeline**:
```
For each test protein:
  1. Fetch UniProt cross-references → PubMed IDs
  2. Retrieve abstracts from PubMed API (recent 2 years prioritized)
  3. Extract GO term candidates via:
     a. Direct GO term mentions in text
     b. LLM extraction: "What molecular functions does this protein have?"
     c. Named entity recognition for function keywords
  4. Score candidates by:
     - Paper recency (newer = higher weight)
     - Journal impact factor
     - Number of supporting papers
     - LLM confidence score
```

**LLM Prompt Template**:
```
Given this abstract about protein {uniprot_id}:
"{abstract}"

Extract the following if mentioned:
1. Molecular functions (what the protein does chemically)
2. Biological processes (what cellular processes it participates in)
3. Cellular components (where in the cell it is located)

For each, provide a confidence score 0-1.
```

**Data Sources**:
- UniProt cross-references (direct protein → PubMed links)
- PubMed API / Europe PMC API
- Semantic Scholar API (for citation context)

**Implementation Priority**: HIGH — this is unique signal that most competitors miss.

**Pros**:
- Captures cutting-edge research not yet in databases
- Can predict truly novel functions
- Provides interpretable evidence (paper citations)
- LLMs are good at this extraction task

**Cons**:
- API rate limits (need caching)
- Not all proteins have papers
- Requires LLM inference (cost/time)

**Expected Score**: 0.38-0.42 (standalone), but **massive value when ensembled**

---

### Tier 2: Optimized MLP Ensemble (Target: 0.44-0.46)
**Approach**: "Standing on Giants' Shoulders"

**Architecture**:
```
Input Features:
  - T5 embeddings (1024-dim, primary signal)
  - ESM2-3B embeddings (2560-dim, complementary)
  - Phylogenetic taxonomy embedding (32-dim, learned)
  - InterPro domain presence (variable, binary features)
       ↓
Feature fusion: Concat → Dense(4096) → LayerNorm → GELU
       ↓
Shared backbone: 2048 → 1024 → 512
       ↓
Three heads with cross-attention:
  - MF head: 512 → 256 → top_k_MF terms
  - BP head: 512 → 256 → top_k_BP terms  
  - CC head: 512 → 256 → top_k_CC terms
  - Cross-attention between heads (CC informs MF, etc.)
       ↓
Sigmoid activation per term
       ↓
Post-processing: Parent propagation, taxon constraints, blend with Tier 0+1
```

**Key Decisions**:
- Use pre-computed T5 embeddings from UniProt (proven best in CAFA 5)
- Supplement with ESM2-3B for complementary signal
- **Target Selection**: Top-K by `IA × sqrt(frequency)`, NOT pure frequency
- **Loss**: Focal loss with IA weights (γ=2, handles class imbalance)
- **Taxonomy**: Phylogenetic embeddings, not one-hot
- 5-fold CV with species-stratified, time-shifted validation split

**Target Selection** (by `IA × sqrt(frequency)`):
- MF: Top 1000 terms
- BP: Top 2000 terms 
- CC: Top 600 terms

**Pros**:
- Proven effective (4th place CAFA 5 used similar approach)
- Computationally efficient (embeddings pre-computed)
- Easy to iterate and debug
- Strong baseline to build upon

**Cons**:
- Limited by embedding quality (can't improve embeddings)
- May miss rare terms not in top-K

**Expected Score**: 0.44-0.46 (with proper ensembling)

---

### Tier 3: Retrieval-Augmented Prediction (Target: 0.45-0.47)
**Approach**: "Search Then Verify"

**Key Insight**: Proteins with similar sequences have similar functions. Instead of predicting from scratch, find similar proteins and transfer/refine their annotations.

**Stage 1: Retrieval**
```
For each test protein:
  1. Compute PLM embedding
  2. Find k=100 nearest neighbors in training set (FAISS)
  3. Collect their GO annotations with distance-weighted scores
  4. Initial prediction = weighted vote of neighbors' annotations
```

**Stage 2: Refinement**
```
Input: [protein_embedding, neighbor_annotations, initial_predictions, pubmed_scores]
       ↓
Transformer encoder (cross-attention between protein and candidate terms)
       ↓
Refined scores for each candidate term
       ↓
Post-processing: Parent propagation, calibration
```

**Key Innovations**:
1. **Retrieval-Augmented Prediction**: Doesn't predict from scratch; uses similar proteins as evidence
2. **Uncertainty from Neighbors**: If neighbors disagree, lower confidence
3. **Handles Rare Terms**: If a neighbor has a rare term, can propagate it
4. **Efficient**: Stage 1 is very fast; Stage 2 only scores ~1000 candidates per protein
5. **Literature Integration**: PubMed scores from Tier 1 inform refinement

**Expected Score**: 0.45-0.47 (robust, lower variance)

---

### Tier 4 (LOW PRIORITY): Graph-Based Approaches
**Approach**: "Structure-Aware Prediction"

**⚠️ Deprioritized**: CAFA 5 showed hierarchy-aware losses didn't outperform simpler approaches. The GO DAG structure is already captured by parent propagation post-processing.

**If time permits**, explore:
- Graph Attention Network over GO DAG for term embeddings
- Contrastive learning between proteins and GO terms
- Zero-shot prediction of unseen terms via term text embeddings

**Expected Score**: 0.40-0.48 (high variance — likely not worth the complexity)

---

## Recommended Strategy: Revised Timeline

### Phase 1: Foundation (Days 1-3)
**Goal**: Working submission + validation infrastructure

- [ ] Set up CAFA-evaluator locally with correct settings
- [ ] Implement submission generation pipeline
- [ ] Download T5 embeddings from UniProt FTP
- [ ] Implement Tier 0 (GOA baseline + kNN)
- [ ] **Submit Tier 0** to verify pipeline works
- **Target Score**: 0.30-0.35

### Phase 2: PubMed Mining (Days 4-10) ⭐ HIGH PRIORITY
**Goal**: Unique competitive advantage through literature mining

- [ ] Build UniProt → PubMed cross-reference database
- [ ] Implement PubMed/Europe PMC API fetcher with caching
- [ ] Extract GO terms from abstracts (regex + NER)
- [ ] Set up LLM extraction pipeline (Claude/GPT-4 for function extraction)
- [ ] Score and rank extracted terms
- [ ] Ensemble with Tier 0
- [ ] **Submit Tier 0 + Tier 1 blend**
- **Target Score**: 0.38-0.42

### Phase 3: MLP Optimization (Days 11-18)
**Goal**: Strong neural baseline with proper features

- [ ] Implement IA-weighted target selection
- [ ] Build phylogenetic taxonomy embeddings
- [ ] Fetch InterPro domain features
- [ ] Train MLP with focal loss
- [ ] 5-fold CV with species stratification
- [ ] Ensemble Tier 0 + 1 + 2
- [ ] **Submit optimized ensemble**
- **Target Score**: 0.44-0.46

### Phase 4: Retrieval + Polish (Days 19-25)
**Goal**: Retrieval-augmented refinement + final optimization

- [ ] Implement Tier 3 retrieval system with FAISS
- [ ] Train refinement model
- [ ] Calibrate probability scores
- [ ] Extensive ensembling experiments
- [ ] Taxon constraint filtering
- [ ] **Final submission**
- **Target Score**: 0.46-0.48

---

## Additional Signal Sources (Ranked by Expected Value)

### 1. PubMed Abstract Mining ⭐⭐⭐ (HIGH VALUE)
**Status**: PRIORITY — implement in Phase 2

**Data Sources**:
- UniProt cross-references: `uniprot.org/uniprotkb/{id}.json` → "references" field
- PubMed API: `eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi`
- Europe PMC API: `europepmc.org/restfulWebService`
- Semantic Scholar: For citation context and related papers

**LLM Models for Extraction**:
- Claude 3.5 Sonnet (best accuracy)
- GPT-4o (fast, good accuracy)
- Local: Llama 3.1 70B or Mistral Large (if API costs are concern)

### 2. GOA UniProt Electronic Annotations ⭐⭐⭐ (HIGH VALUE)
**Status**: Implement in Phase 1

**Data Source**: `ftp.ebi.ac.uk/pub/databases/GO/goa/UNIPROT/goa_uniprot_all.gaf.gz`

**Usage**:
- Extract IEA, ISS, ISO annotations for test proteins
- These often become experimentally confirmed → easy wins

### 3. InterPro Domain Features ⭐⭐ (MEDIUM VALUE)
**Status**: Implement in Phase 3

**Data Sources**:
- InterPro API: `ebi.ac.uk/interpro/api/entry/interpro/protein/uniprot/{id}`
- InterPro2GO mapping: `ftp.ebi.ac.uk/pub/databases/interpro/`

**Usage**:
- Binary features for domain presence
- Domain → GO term transfer via InterPro2GO

### 4. Phylogenetic Taxonomy Embeddings ⭐⭐ (MEDIUM VALUE)
**Status**: Implement in Phase 3

**Data Source**: NCBI Taxonomy database

**Implementation**:
```python
# Build taxonomy tree
tree = NCBITaxonomy.load()
for species_id in all_species:
    path = tree.get_lineage(species_id)  # [root, ..., species]
    # Embed each node, take mean
    species_embedding = mean([node_embeddings[n] for n in path])
```

### 5. Protein-Protein Interaction Networks ⭐ (LOW VALUE)
**Status**: Deprioritized — CAFA 5 showed STRING features didn't help

### 6. 3D Structure Features ⭐ (LOW VALUE FOR EFFORT)
**Status**: Deprioritized — high compute cost, marginal gains

---

## Novel Ideas Worth Exploring

### 1. Annotation Velocity Prediction
**Hypothesis**: Proteins with recent IEA annotations + recent papers are likely to gain experimental confirmation.

**Implementation**: 
- Track annotation dates (requires multiple UniProt snapshots)
- Train a model to predict which IEA → EXP
- Up-weight predictions for high-velocity proteins

### 2. Cross-Ontology Attention
**Hypothesis**: CC → MF → BP information flow (location informs function informs process).

**Implementation**:
- Attention layers between ontology heads
- Joint prediction with consistency regularization

### 3. Uncertainty-Aware Calibration
**Hypothesis**: Novel proteins (far from training) should have lower confidence.

**Implementation**:
- Compute distance to nearest training protein
- Scale prediction confidence: `calibrated = raw_score × (1 - novelty_penalty)`

### 4. Paper Citation Network
**Hypothesis**: Proteins cited together may share functions.

**Implementation**:
- Build citation graph from PubMed
- Propagate GO terms through co-citation links

---

## Validation Strategy

### Time-Shifted Holdout (CRITICAL)

**Problem**: Our training data is from UniProt 2025_03 (June 2025). We need to simulate the prospective evaluation.

**Solution**: Download an older UniProt release and use the difference as validation.

```bash
# Download UniProt 2024_06 (1 year older)
wget ftp://ftp.uniprot.org/pub/databases/uniprot/previous_releases/release-2024_06/knowledgebase/

# Extract annotations
python scripts/extract_annotations.py --release 2024_06 --output data/validation/

# Validation set = (2025_03 annotations) - (2024_06 annotations)
# These are proteins that gained annotations in the past year
```

**If older releases unavailable**: 
- Use 80/20 split where 20% = proteins with fewest annotations (proxy for "about to be annotated")
- Or split by species: train on model organisms, validate on less-studied species

### CAFA-Evaluator Settings

```bash
cafa-evaluator \
  --prediction predictions.tsv \
  --ground_truth ground_truth.tsv \
  --ontology data/Train/go-basic.obo \
  --ia data/IA.tsv \
  --max_terms 500 \
  --prop fill \
  --norm cafa \
  --no_orphans
```

### Validation Metrics to Track

| Metric | Target | Why It Matters |
|--------|--------|----------------|
| Overall F1 | 0.45+ | Competition metric |
| MF F1 | 0.50+ | Usually easier |
| BP F1 | 0.40+ | Hardest ontology |
| CC F1 | 0.45+ | Medium difficulty |
| High-IA F1 (IA > 5) | 0.30+ | Where real points come from |
| Rare species F1 | 0.35+ | Generalization test |

### Robustness Checks
- Performance on rare terms (IA > 5) — **this is where competition is won/lost**
- Performance on novel proteins (low sequence identity to training)
- Performance on under-represented species
- Calibration plots (predicted probability vs actual frequency)

---

## Technical Requirements

### Compute Resources
- GPU: 1x A100 (40GB) for training, or 2x RTX 3090
- Storage: ~50GB for embeddings + data
- RAM: 64GB recommended for loading all embeddings

### Key Dependencies
```
# Core ML
torch>=2.0
transformers>=4.35
faiss-gpu>=1.7
scikit-learn>=1.4

# Biology
obonet>=1.1
biopython>=1.83

# Data
pandas>=2.1
numpy>=1.26
pyarrow>=14.0  # for parquet files
h5py>=3.10    # for T5 embeddings

# PubMed Mining
requests>=2.31
beautifulsoup4>=4.12
xmltodict>=0.13
rate-limiter>=1.0

# LLM Extraction
anthropic>=0.18  # Claude API
openai>=1.12    # GPT-4 API (backup)

# Evaluation
cafa-evaluator  # pip install git+https://github.com/BioComputingUP/CAFA-evaluator
```

### Pre-computed Embeddings to Download
```bash
# T5 embeddings from UniProt FTP (~20GB)
wget -r -np ftp://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/embeddings/

# ESM2-3B embeddings from Kaggle (~30GB)
kaggle datasets download -d andreylalaley/esm2-embeddings-cafa6

# ESM-Cambrian embeddings (backup, ~15GB)
kaggle datasets download -d dalloliogm/esm-cambrian-embeddings
```

### External Data to Fetch
```bash
# GOA UniProt annotations
wget ftp://ftp.ebi.ac.uk/pub/databases/GO/goa/UNIPROT/goa_uniprot_all.gaf.gz

# InterPro2GO mapping
wget ftp://ftp.ebi.ac.uk/pub/databases/interpro/current_release/interpro2go

# NCBI Taxonomy (for phylogenetic embeddings)
wget ftp://ftp.ncbi.nlm.nih.gov/pub/taxonomy/taxdump.tar.gz
```

---

## Success Criteria

| Milestone | Target Score | Timeline | Components |
|-----------|--------------|----------|------------|
| Working submission | 0.25+ | Day 3 | Tier 0 (GOA + kNN) |
| PubMed pipeline working | 0.35+ | Day 10 | Tier 0 + Tier 1 |
| Optimized MLP | 0.44+ | Day 18 | Tier 0 + 1 + 2 |
| Full ensemble | 0.46+ | Day 25 | All tiers |
| Calibrated final | 0.47+ | Day 28 | Polished ensemble |

**Competition Goal**: Top 3 finish (currently ~0.44 threshold)

**Key Differentiator**: PubMed mining is our unique edge — most competitors rely only on embeddings.

---

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| PubMed API rate limits | Cache aggressively, use Europe PMC as backup |
| LLM extraction cost | Use Claude Haiku for bulk, Sonnet for hard cases |
| T5 embeddings not available | Fall back to ESM2-3B (Kaggle dataset) |
| Overfitting to LB | Trust validation more than LB (prospective nature) |
| Time crunch | Phase 1+2 are MVP, Phase 3+4 are enhancements |
