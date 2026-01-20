"""PubMed mining module for GO term extraction from literature."""

from cafa_6_protein.pubmed.cache import AbstractCache, PublicationCache
from cafa_6_protein.pubmed.extractor import GODictionary, GOExtractor, extract_go_terms
from cafa_6_protein.pubmed.ncbi import NCBIClient, parse_efetch_response
from cafa_6_protein.pubmed.uniprot import UniProtClient, parse_uniprot_response

__all__ = [
    "AbstractCache",
    "GODictionary",
    "GOExtractor",
    "NCBIClient",
    "PublicationCache",
    "UniProtClient",
    "extract_go_terms",
    "parse_efetch_response",
    "parse_uniprot_response",
]
