"""
Web Search Integration Package
"""

from .web_search import (
    WebSearchClient,
    SearchResult,
    SearchResponse,
    CitationManager,
    search_web
)

__all__ = [
    'WebSearchClient',
    'SearchResult',
    'SearchResponse',
    'CitationManager',
    'search_web',
]
