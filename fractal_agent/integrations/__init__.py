"""
Fractal Agent External Knowledge Integration

Provides web search and document processing capabilities.
"""

from .search.web_search import (
    WebSearchClient,
    SearchResult,
    SearchResponse,
    CitationManager,
    search_web
)

from .documents.document_parser import (
    DocumentParser,
    PDFParser,
    DOCXParser,
    ParsedDocument,
    DocumentMetadata,
    parse_document
)

__all__ = [
    'WebSearchClient',
    'SearchResult',
    'SearchResponse',
    'CitationManager',
    'search_web',
    'DocumentParser',
    'PDFParser',
    'DOCXParser',
    'ParsedDocument',
    'DocumentMetadata',
    'parse_document',
]
