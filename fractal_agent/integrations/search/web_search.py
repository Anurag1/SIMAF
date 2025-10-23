"""
Web Search Integration for Fractal Agent

Provides web search capabilities using DuckDuckGo (free) and Google (API key required).
Includes citation management for tracking sources.
"""

import logging
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import requests
from bs4 import BeautifulSoup
import time

logger = logging.getLogger(__name__)


class SearchEngine(Enum):
    """Supported search engines"""
    DUCKDUCKGO = "duckduckgo"
    GOOGLE = "google"


@dataclass
class SearchResult:
    """Single search result"""
    title: str
    url: str
    snippet: str
    source: str = ""
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "title": self.title,
            "url": self.url,
            "snippet": self.snippet,
            "source": self.source,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata
        }


@dataclass
class SearchResponse:
    """Search response containing multiple results"""
    query: str
    results: List[SearchResult]
    engine: str
    total_results: int
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "query": self.query,
            "results": [r.to_dict() for r in self.results],
            "engine": self.engine,
            "total_results": self.total_results,
            "timestamp": self.timestamp.isoformat()
        }


class CitationManager:
    """Manages citations for external sources"""

    def __init__(self):
        self.citations: List[Dict[str, Any]] = []
        self._citation_map: Dict[str, int] = {}

    def add_citation(self, url: str, title: str, snippet: str = "",
                     accessed_date: Optional[datetime] = None) -> int:
        """
        Add a citation and return citation number

        Args:
            url: Source URL
            title: Page/article title
            snippet: Relevant excerpt
            accessed_date: When the source was accessed

        Returns:
            Citation number (1-indexed)
        """
        if url in self._citation_map:
            return self._citation_map[url]

        citation_num = len(self.citations) + 1
        self.citations.append({
            "number": citation_num,
            "url": url,
            "title": title,
            "snippet": snippet,
            "accessed_date": accessed_date or datetime.now()
        })
        self._citation_map[url] = citation_num
        return citation_num

    def get_citation(self, citation_num: int) -> Optional[Dict[str, Any]]:
        """Get citation by number"""
        if 1 <= citation_num <= len(self.citations):
            return self.citations[citation_num - 1]
        return None

    def format_citations(self, style: str = "numbered") -> str:
        """
        Format all citations

        Args:
            style: Citation style ('numbered', 'apa', 'mla')

        Returns:
            Formatted citation list
        """
        if not self.citations:
            return ""

        if style == "numbered":
            lines = ["References:"]
            for cite in self.citations:
                lines.append(
                    f"[{cite['number']}] {cite['title']}\n"
                    f"    {cite['url']}\n"
                    f"    Accessed: {cite['accessed_date'].strftime('%Y-%m-%d')}"
                )
            return "\n".join(lines)

        elif style == "apa":
            lines = ["References:"]
            for cite in self.citations:
                date_str = cite['accessed_date'].strftime('%Y, %B %d')
                lines.append(
                    f"{cite['title']}. Retrieved {date_str}, from {cite['url']}"
                )
            return "\n".join(lines)

        else:
            return self.format_citations("numbered")

    def clear(self):
        """Clear all citations"""
        self.citations.clear()
        self._citation_map.clear()


class WebSearchClient:
    """Client for performing web searches"""

    def __init__(self,
                 default_engine: SearchEngine = SearchEngine.DUCKDUCKGO,
                 google_api_key: Optional[str] = None,
                 google_cse_id: Optional[str] = None):
        """
        Initialize web search client

        Args:
            default_engine: Default search engine to use
            google_api_key: Google Custom Search API key (optional)
            google_cse_id: Google Custom Search Engine ID (optional)
        """
        self.default_engine = default_engine
        self.google_api_key = google_api_key
        self.google_cse_id = google_cse_id
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })

    def search(self,
               query: str,
               max_results: int = 10,
               engine: Optional[SearchEngine] = None) -> SearchResponse:
        """
        Perform web search

        Args:
            query: Search query
            max_results: Maximum number of results to return
            engine: Search engine to use (defaults to default_engine)

        Returns:
            SearchResponse with results
        """
        engine = engine or self.default_engine

        try:
            if engine == SearchEngine.DUCKDUCKGO:
                return self._search_duckduckgo(query, max_results)
            elif engine == SearchEngine.GOOGLE:
                return self._search_google(query, max_results)
            else:
                raise ValueError(f"Unsupported search engine: {engine}")
        except Exception as e:
            logger.error(f"Search failed for query '{query}': {e}")
            return SearchResponse(
                query=query,
                results=[],
                engine=engine.value,
                total_results=0
            )

    def _search_duckduckgo(self, query: str, max_results: int) -> SearchResponse:
        """Search using DuckDuckGo HTML scraping"""
        try:
            # DuckDuckGo HTML search
            url = "https://html.duckduckgo.com/html/"
            data = {"q": query}

            response = self.session.post(url, data=data, timeout=10)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, 'html.parser')
            results = []

            # Parse search results
            for result_div in soup.find_all('div', class_='result')[:max_results]:
                try:
                    # Extract title and URL
                    title_elem = result_div.find('a', class_='result__a')
                    if not title_elem:
                        continue

                    title = title_elem.get_text(strip=True)
                    url = title_elem.get('href', '')

                    # Extract snippet
                    snippet_elem = result_div.find('a', class_='result__snippet')
                    snippet = snippet_elem.get_text(strip=True) if snippet_elem else ""

                    results.append(SearchResult(
                        title=title,
                        url=url,
                        snippet=snippet,
                        source="DuckDuckGo"
                    ))
                except Exception as e:
                    logger.warning(f"Failed to parse result: {e}")
                    continue

            return SearchResponse(
                query=query,
                results=results,
                engine="duckduckgo",
                total_results=len(results)
            )

        except Exception as e:
            logger.error(f"DuckDuckGo search failed: {e}")
            raise

    def _search_google(self, query: str, max_results: int) -> SearchResponse:
        """Search using Google Custom Search API"""
        if not self.google_api_key or not self.google_cse_id:
            raise ValueError("Google API key and CSE ID required for Google search")

        try:
            url = "https://www.googleapis.com/customsearch/v1"
            params = {
                "key": self.google_api_key,
                "cx": self.google_cse_id,
                "q": query,
                "num": min(max_results, 10)  # API limit is 10 per request
            }

            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()

            results = []
            for item in data.get("items", []):
                results.append(SearchResult(
                    title=item.get("title", ""),
                    url=item.get("link", ""),
                    snippet=item.get("snippet", ""),
                    source="Google",
                    metadata={
                        "displayLink": item.get("displayLink", ""),
                        "formattedUrl": item.get("formattedUrl", "")
                    }
                ))

            total = int(data.get("searchInformation", {}).get("totalResults", 0))

            return SearchResponse(
                query=query,
                results=results,
                engine="google",
                total_results=total
            )

        except Exception as e:
            logger.error(f"Google search failed: {e}")
            raise

    def search_with_citations(self,
                             query: str,
                             citation_manager: CitationManager,
                             max_results: int = 10) -> SearchResponse:
        """
        Perform search and automatically add citations

        Args:
            query: Search query
            citation_manager: CitationManager to add citations to
            max_results: Maximum results

        Returns:
            SearchResponse with results
        """
        response = self.search(query, max_results)

        # Add citations for all results
        for result in response.results:
            citation_num = citation_manager.add_citation(
                url=result.url,
                title=result.title,
                snippet=result.snippet
            )
            result.metadata['citation_number'] = citation_num

        return response


# Convenience function
def search_web(query: str,
               max_results: int = 10,
               engine: SearchEngine = SearchEngine.DUCKDUCKGO,
               google_api_key: Optional[str] = None,
               google_cse_id: Optional[str] = None) -> SearchResponse:
    """
    Convenience function for performing web search

    Args:
        query: Search query
        max_results: Maximum number of results
        engine: Search engine to use
        google_api_key: Google API key (if using Google)
        google_cse_id: Google CSE ID (if using Google)

    Returns:
        SearchResponse with results
    """
    client = WebSearchClient(
        default_engine=engine,
        google_api_key=google_api_key,
        google_cse_id=google_cse_id
    )
    return client.search(query, max_results)
