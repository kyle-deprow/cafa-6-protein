# CAFA 6 Protein Function Prediction - Research Notes

## Competition Overview

**Goal**: Predict Gene Ontology (GO) terms for proteins based on their amino acid sequences across three sub-ontologies:
- **MF** (Molecular Function): What the protein does at a molecular level
- **BP** (Biological Process): Which biological processes it participates in  
- **CC** (Cellular Component): Where in the cell it is located

**Evaluation**: Maximum weighted F1-measure using Information Accretion (IA) weights for each GO term. Final score is arithmetic mean of F1 scores across MF, BP, and CC.

**Key Challenge**: This is a **prospective (time-shifted) competition** - the test set consists of proteins that will gain experimental annotations AFTER the submission deadline. This creates a distribution shift between public leaderboard and final evaluation.

---

## Current CAFA 6 Leaderboard (Top 10)

| Rank | Team | Score |
|------|------|-------|
| 1 | Mixture of Experts | 0.4481 |
| 2 | WePredictProteins | 0.4393 |
| 3 | Guoliang&Born4 | 0.4333 |
| 4 | chya | 0.4291 |
| 5 | mirrandax | 0.4231 |
| 6 | RDL | 0.4183 |
| 7 | HoehndorfLab | 0.4171 |
| 8 | Diogo | 0.4162 |
| 9 | Stefan Stefanov | 0.4156 |
| 10 | T1_INT34057 | 0.4091 |

---

## Key Approaches from CAFA 5 & CAFA 6 Discussions

### 1. Protein Language Model (PLM) Embeddings

**Most Effective Embeddings (ranked by community feedback):**

| Model | Dimensions | Performance Notes |
|-------|------------|-------------------|
| **ProtT5** (T5) | Varies | Consistently top performer; available pre-computed from UniProt |
| **ESM2-t36** | 2560 | Strong performer, good balance |
| **ESM2-t48** | 5120 | Largest ESM2, best single model |
| **Ankh** | - | Works well in ensembles |
| **ESM-Cambrian** | 320 | Successor to ESM2, newer model |
| **ProtBert** | - | Underperformed; may work if compressed with UMAP/tSNE to 3 dims |

**Key Insight**: T5 embeddings outperform ESM2 in local validation according to some competitors.

**Pre-computed Resources**:
- T5 embeddings: Available directly from UniProt FTP (HDF5 format)
- ESM-Cambrian: Kaggle dataset by dalloliogm
- ESM2 (5120-dim): Shared by andreylalaley

### 2. Simple Baseline: GOA UniProt Direct Mapping (~0.269 LB)

A simple baseline that achieves ~0.269 on public LB without any ML:
1. Download GOA UniProt GAF file (Gene Association File)
2. Filter out negated ("NOT") annotations
3. Map test proteins directly to their existing GO terms
4. Submit with confidence = 1.0

**Why it works**: Electronic annotations (IEA) may become experimentally confirmed during the evaluation period.

### 3. Winning CAFA 5 Approaches

#### 3rd Place Solution (tito)
- **Embeddings**: T5, ESM2-t36, ESM2-t48, and combinations
- **Additional Features**:
  - Taxonomy data (one-hot encoded, only 90 taxa in test)
  - Non-experimental annotations as features (11 evidence codes, processed via 1D-CNN)
- **Architecture**: Neural network with 4 model types based on embedding combinations
- **Validation**: Time-shifted holdout (used latest UniProtKB annotations)
- **Ensemble**: Simple averaging of 4 model types

#### 4th Place Solution (Synthetic Goose)  
- **Embeddings**: ProtT5, ESM2, Ankh
- **Additional Features**:
  - Taxonomy binary matrix
  - **TF-IDF of paper abstracts** associated with each protein (from PubMed)
- **Architecture**: Dense NN in Keras with separate heads for each ontology
- **Labels**: Top 1500 BPO, 800 CCO, 800 MFO terms (selected by IA × frequency)
- **Loss**: Binary Cross Entropy with IA weights as class_weight
- **Training**: 5-fold CV with prediction averaging
- **Key Insight**: Ridge regression sometimes outperformed NN on PLM embeddings (possible underfitting)

#### 13th Place Solution
- **Embeddings**: ESM2-3B with Z-score normalization
- **Target Selection**: Top 500 GO terms by frequency or threshold (BP>250, MF/CC>50)
- **Architecture**: Multiple MLPs with 10-fold CV
- **Post-processing**:
  - **Taxon constraints** filtering (eliminate incompatible terms for species)
  - Parent-child propagation (ensure parent ≥ max(children))
  - Merge with GO annotations for test proteins never experimentally confirmed
- **Ensemble**: Average 6 MLPs, then average with exported GO annotations

### 4. Multi-Modal & Advanced Approaches

**GOCurator (CAFA 5 Winner)**:
- Integrated multiple information sources:
  - 3D protein structures
  - Textual descriptions
  - Scientific literature
- Used **learning-to-rank** framework

**Structure-Based Approaches** (from NVIDIA BioNeMo discussion):
- GPU-accelerated MSA generation with MMseqs2 (177x speedup vs JackHMMER)
- Structure prediction with OpenFold2 or Boltz2
- Equivariant layers for 3D graph operations (cuEquivariance)

---

## What Didn't Work (Negative Results)

| Approach | Issue |
|----------|-------|
| ProtBert embeddings | Low performance; only worked when compressed to 3 dims via UMAP/tSNE |
| STRING database features | Did not help |
| Anc2Vec label embeddings | Did not improve performance |
| Hierarchy-aware loss | Not better than standard BCE |
| Simply increasing model capacity | Rapid overfitting |

---

## Validation Strategy

**Critical Insight**: Standard cross-validation doesn't work well due to time-shift nature.

**Recommended Approach** (from CAFA 5 winners):
1. Use experimental annotations from the **latest** UniProtKB release as validation set
2. Exclude any protein-term pairs that were already in training data
3. Mimic the prospective evaluation by creating time-shifted splits

**CAFA Evaluator Settings** (to match leaderboard):
```bash
-max_terms 500 -prop fill -norm cafa -no_orphans
```

---

## Key Technical Details

### GO Ontology Structure
- Directed Acyclic Graph (DAG)
- Predictions must be **propagated to root** (parents get max of children's scores)
- Root terms: BPO=GO:0008150, CCO=GO:0005575, MFO=GO:0003674

### Submission Format
- Tab-separated: `protein_id  GO_term  score`
- Score in (0, 1.000] with up to 3 significant figures
- Max 1500 terms per protein across all ontologies
- No header

### Information Accretion (IA) Weights
- Provided in IA.tsv
- Used for weighted precision/recall calculation
- Root terms have weight 0
- Rare (deep) terms have higher weights

---

## Ideas for Brainstorming

### Quick Wins
1. ✅ Use pre-computed T5 embeddings from UniProt
2. ✅ Include GOA UniProt annotations as baseline/features
3. ✅ Add taxonomy one-hot encoding as features
4. ✅ Train separate models per ontology (MF, BP, CC)

### Medium Effort
1. 🔧 Ensemble multiple PLM embeddings (T5 + ESM2 + Ankh)
2. 🔧 Use non-experimental annotations as additional features
3. 🔧 Implement taxon constraints filtering
4. 🔧 Add TF-IDF features from paper abstracts (PubMed)
5. 🔧 Use IA weights as loss weights during training

### Advanced / Experimental
1. 🔬 Fine-tune protein language models on GO prediction task
2. 🔬 Use 3D structure predictions (AlphaFold2/ESMFold) as features
3. 🔬 Learning-to-rank framework (like GOCurator)
4. 🔬 Graph Neural Networks on GO ontology structure
5. 🔬 Multi-task learning across ontologies
6. 🔬 LLM-based text predictions from protein context
7. 🔬 Knowledge distillation from larger models

### Novel Ideas to Explore
1. **Contrastive Learning**: Pre-train embeddings to cluster proteins with similar GO terms
2. **Label Hierarchy Encoding**: Use graph attention over GO DAG structure
3. **Temporal Modeling**: Predict which annotations are "about to be confirmed"
4. **Protein-Protein Interaction**: Use STRING or other PPI data for message passing
5. **Multi-species Transfer**: Transfer learning from well-annotated species to less-studied ones
6. **Uncertainty Quantification**: Calibrated probabilities for prospective predictions

---

## Resources

### Data Sources
- UniProt T5 embeddings: `ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/embeddings/`
- GOA UniProt GAF: `ftp.ebi.ac.uk/pub/databases/GO/goa/UNIPROT/`
- ESM-Cambrian embeddings: Kaggle dataset by dalloliogm

### Tools
- **CAFA-evaluator**: Official evaluation code from GitHub (BioComputingUP/CAFA-evaluator)
- **obonet**: Python library for parsing GO OBO files
- **BioNeMo**: NVIDIA framework for protein language models

### Key Papers
- Jiang et al. (2016) - Expanded evaluation of protein function prediction (evaluation methodology)
- Clark & Radivojac (2013) - Information accretion weighting rationale
- NetGO - Text-based function prediction using paper abstracts

---

## Implementation Progress

### Phase 1: Baseline Infrastructure ✅
- [x] Data loaders for FASTA, TSV, and OBO files
- [x] Evaluation metrics (weighted F1, precision, recall)
- [x] Frequency baseline predictor
- [x] kNN predictor with distance-weighted scoring
- [x] Ontology utilities (ancestor propagation, GO roots)
- [x] Ensemble blending predictor
- [x] CLI with commands: `info`, `baseline`, `cv`, `validate`
- [x] 127 tests passing

**Baseline CV Results** (5-fold):
- BP F-max: 0.22
- CC F-max: 0.59
- MF F-max: 0.54

### Phase 2: PubMed Mining Pipeline ✅
- [x] Publication cache (Parquet format for protein→PMID mappings)
- [x] Abstract cache (SQLite for PMID→abstract text)
- [x] UniProt API client for fetching protein publications
- [x] NCBI E-utilities client for fetching PubMed abstracts
- [x] GO dictionary from OBO with Aho-Corasick matching
- [x] GO term extractor from text
- [x] Full e2e pipeline tested

**Next Steps**:
- [ ] Fetch publications for all training proteins
- [ ] Build text features (TF-IDF of abstracts)
- [ ] Integrate with baseline predictors

### Phase 3: PLM Embeddings (Planned)
- [ ] Download pre-computed T5 embeddings from UniProt
- [ ] ESM2 embeddings via API or pre-computed
- [ ] MLP classifier with embedding features
- [ ] Ensemble with baseline predictors

---

## Next Steps

1. [x] Set up baseline with frequency predictor
2. [ ] Set up baseline with pre-computed T5 embeddings
3. [ ] Implement proper time-shifted validation
4. [x] Add PubMed text mining infrastructure
5. [ ] Build ensemble of multiple PLM embeddings
6. [ ] Experiment with taxon constraints and parent-child propagation
7. [ ] Explore multi-modal features (text, structure)
