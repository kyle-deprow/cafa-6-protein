"""Pytest fixtures for CAFA 6 tests."""

import pandas as pd
import pytest


@pytest.fixture
def sample_sequences() -> pd.DataFrame:
    """Sample protein sequences for testing."""
    return pd.DataFrame(
        {
            "protein_id": ["P12345", "P67890", "Q11111"],
            "sequence": [
                "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKRQQIAAALMPVTMVNDF",
                "MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHFDLSH",
                "MNIFEMLRIDEGLRLKIYKDTEGYYTIGIGHLLTKSPSLNAAKSELDKAIGRNTNGVITKDEAEKLFNQDVDAAVRGILRNAKLKPVYDSLDAVRRAALINMVFQMGETGVAGFTNSLRMLQQKRWDEAAVNLAKSRWYNQTPNRAKRVITTFRTGTWDAYKNL",
            ],
        }
    )


@pytest.fixture
def sample_annotations() -> pd.DataFrame:
    """Sample GO annotations for testing."""
    return pd.DataFrame(
        {
            "protein_id": ["P12345", "P12345", "P67890", "Q11111"],
            "go_term": ["GO:0005524", "GO:0016887", "GO:0005833", "GO:0003677"],
            "score": [1.0, 1.0, 1.0, 1.0],
        }
    )


@pytest.fixture
def sample_weights() -> dict[str, float]:
    """Sample information accretion weights for testing."""
    return {
        "GO:0005524": 2.5,  # ATP binding
        "GO:0016887": 3.2,  # ATPase activity
        "GO:0005833": 4.1,  # hemoglobin complex
        "GO:0003677": 2.8,  # DNA binding
    }
