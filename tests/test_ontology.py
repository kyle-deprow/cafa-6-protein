"""Tests for GO ontology utilities including parent propagation.

TDD tests for GO DAG operations needed for CAFA submission.
"""

from pathlib import Path
from tempfile import NamedTemporaryFile

import pandas as pd
import pytest

from cafa_6_protein.data.ontology import (
    load_go_ontology,
    propagate_to_parents,
    get_ancestors,
    GO_ROOTS,
)


# Minimal OBO content for testing
SAMPLE_OBO_CONTENT = """format-version: 1.2
ontology: go

[Term]
id: GO:0003674
name: molecular_function
namespace: molecular_function
def: "A molecular process that can be carried out by the action of a single macromolecular machine." []

[Term]
id: GO:0005488
name: binding
namespace: molecular_function
def: "The selective, non-covalent, often stoichiometric, interaction of a molecule with one or more specific sites on another molecule." []
is_a: GO:0003674 ! molecular_function

[Term]
id: GO:0005515
name: protein binding
namespace: molecular_function
def: "Binding to a protein." []
is_a: GO:0005488 ! binding

[Term]
id: GO:0005524
name: ATP binding
namespace: molecular_function
def: "Binding to ATP, adenosine 5'-triphosphate." []
is_a: GO:0005488 ! binding

[Term]
id: GO:0008150
name: biological_process
namespace: biological_process
def: "A biological process is the execution of a genetically-encoded biological module." []

[Term]
id: GO:0006412
name: translation
namespace: biological_process
def: "The cellular metabolic process in which a protein is formed." []
is_a: GO:0008150 ! biological_process

[Term]
id: GO:0005575
name: cellular_component
namespace: cellular_component
def: "A location, relative to cellular compartments and structures." []

[Term]
id: GO:0005737
name: cytoplasm
namespace: cellular_component
def: "The contents of a cell excluding the plasma membrane and nucleus." []
is_a: GO:0005575 ! cellular_component

[Term]
id: GO:0005829
name: cytosol
namespace: cellular_component
def: "The part of the cytoplasm that does not contain organelles." []
is_a: GO:0005737 ! cytoplasm
"""


class TestLoadGoOntology:
    """Tests for loading GO ontology from OBO files."""

    def test_load_go_ontology_basic(self, tmp_path: Path) -> None:
        """Test loading a basic OBO file."""
        obo_file = tmp_path / "test.obo"
        obo_file.write_text(SAMPLE_OBO_CONTENT)

        graph = load_go_ontology(obo_file)

        assert graph is not None
        assert "GO:0003674" in graph.nodes()  # MF root
        assert "GO:0008150" in graph.nodes()  # BP root
        assert "GO:0005575" in graph.nodes()  # CC root

    def test_load_go_ontology_has_edges(self, tmp_path: Path) -> None:
        """Test that parent-child relationships are loaded."""
        obo_file = tmp_path / "test.obo"
        obo_file.write_text(SAMPLE_OBO_CONTENT)

        graph = load_go_ontology(obo_file)

        # GO:0005515 (protein binding) is_a GO:0005488 (binding)
        # In networkx, edges point from child to parent
        assert graph.has_edge("GO:0005515", "GO:0005488")


class TestGetAncestors:
    """Tests for getting ancestor terms in GO DAG."""

    def test_get_ancestors_basic(self, tmp_path: Path) -> None:
        """Test getting ancestors of a term."""
        obo_file = tmp_path / "test.obo"
        obo_file.write_text(SAMPLE_OBO_CONTENT)
        graph = load_go_ontology(obo_file)

        # cytosol -> cytoplasm -> cellular_component
        ancestors = get_ancestors(graph, "GO:0005829")

        assert "GO:0005737" in ancestors  # cytoplasm
        assert "GO:0005575" in ancestors  # cellular_component (root)

    def test_get_ancestors_root_term(self, tmp_path: Path) -> None:
        """Test that root terms have no ancestors."""
        obo_file = tmp_path / "test.obo"
        obo_file.write_text(SAMPLE_OBO_CONTENT)
        graph = load_go_ontology(obo_file)

        ancestors = get_ancestors(graph, "GO:0003674")

        assert len(ancestors) == 0

    def test_get_ancestors_missing_term(self, tmp_path: Path) -> None:
        """Test handling of terms not in ontology."""
        obo_file = tmp_path / "test.obo"
        obo_file.write_text(SAMPLE_OBO_CONTENT)
        graph = load_go_ontology(obo_file)

        ancestors = get_ancestors(graph, "GO:9999999")

        assert ancestors == set()


class TestPropagateToParents:
    """Tests for propagating predictions to parent terms."""

    def test_propagate_basic(self, tmp_path: Path) -> None:
        """Test basic parent propagation."""
        obo_file = tmp_path / "test.obo"
        obo_file.write_text(SAMPLE_OBO_CONTENT)
        graph = load_go_ontology(obo_file)

        predictions = pd.DataFrame({
            "protein_id": ["P12345", "P12345"],
            "go_term": ["GO:0005829", "GO:0005515"],  # cytosol, protein binding
            "score": [0.9, 0.8],
        })

        propagated = propagate_to_parents(predictions, graph)

        # Should include original terms plus ancestors
        p12345_terms = propagated[propagated["protein_id"] == "P12345"]["go_term"].unique()
        
        assert "GO:0005829" in p12345_terms  # original: cytosol
        assert "GO:0005737" in p12345_terms  # parent: cytoplasm
        assert "GO:0005575" in p12345_terms  # root: cellular_component
        assert "GO:0005515" in p12345_terms  # original: protein binding
        assert "GO:0005488" in p12345_terms  # parent: binding
        assert "GO:0003674" in p12345_terms  # root: molecular_function

    def test_propagate_max_score_rule(self, tmp_path: Path) -> None:
        """Test that parent scores are max of children's scores."""
        obo_file = tmp_path / "test.obo"
        obo_file.write_text(SAMPLE_OBO_CONTENT)
        graph = load_go_ontology(obo_file)

        # cytosol (0.9) and another child of cytoplasm would propagate max to cytoplasm
        predictions = pd.DataFrame({
            "protein_id": ["P12345"],
            "go_term": ["GO:0005829"],  # cytosol
            "score": [0.9],
        })

        propagated = propagate_to_parents(predictions, graph)

        # Parent cytoplasm should have score >= cytosol's score
        cytoplasm_row = propagated[
            (propagated["protein_id"] == "P12345") & 
            (propagated["go_term"] == "GO:0005737")
        ]
        assert len(cytoplasm_row) == 1
        assert cytoplasm_row.iloc[0]["score"] >= 0.9

    def test_propagate_multiple_proteins(self, tmp_path: Path) -> None:
        """Test propagation for multiple proteins."""
        obo_file = tmp_path / "test.obo"
        obo_file.write_text(SAMPLE_OBO_CONTENT)
        graph = load_go_ontology(obo_file)

        predictions = pd.DataFrame({
            "protein_id": ["P12345", "P67890"],
            "go_term": ["GO:0005829", "GO:0006412"],
            "score": [0.9, 0.7],
        })

        propagated = propagate_to_parents(predictions, graph)

        # Both proteins should have propagated terms
        assert len(propagated[propagated["protein_id"] == "P12345"]) >= 1
        assert len(propagated[propagated["protein_id"] == "P67890"]) >= 1

    def test_propagate_preserves_original_scores(self, tmp_path: Path) -> None:
        """Test that original prediction scores are preserved."""
        obo_file = tmp_path / "test.obo"
        obo_file.write_text(SAMPLE_OBO_CONTENT)
        graph = load_go_ontology(obo_file)

        predictions = pd.DataFrame({
            "protein_id": ["P12345"],
            "go_term": ["GO:0005829"],
            "score": [0.9],
        })

        propagated = propagate_to_parents(predictions, graph)

        # Original term should keep its score
        original_row = propagated[
            (propagated["protein_id"] == "P12345") & 
            (propagated["go_term"] == "GO:0005829")
        ]
        assert len(original_row) == 1
        assert original_row.iloc[0]["score"] == pytest.approx(0.9)

    def test_propagate_empty_predictions(self, tmp_path: Path) -> None:
        """Test propagation with empty predictions."""
        obo_file = tmp_path / "test.obo"
        obo_file.write_text(SAMPLE_OBO_CONTENT)
        graph = load_go_ontology(obo_file)

        predictions = pd.DataFrame(columns=["protein_id", "go_term", "score"])

        propagated = propagate_to_parents(predictions, graph)

        assert len(propagated) == 0


class TestGoRoots:
    """Tests for GO root term constants."""

    def test_go_roots_defined(self) -> None:
        """Test that GO root terms are properly defined."""
        assert GO_ROOTS["MF"] == "GO:0003674"
        assert GO_ROOTS["BP"] == "GO:0008150"
        assert GO_ROOTS["CC"] == "GO:0005575"
