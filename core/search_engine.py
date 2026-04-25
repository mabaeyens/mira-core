"""Search engine module with Ollama native and DuckDuckGo fallback."""

import logging
from typing import List, Dict, Optional
from .config import MAX_SEARCH_RESULTS, SEARCH_TIMEOUT, USE_NATIVE_SEARCH

logger = logging.getLogger(__name__)

try:
    from ollama import web_search as ollama_web_search
    OLLAMA_NATIVE_AVAILABLE = True
except ImportError:
    OLLAMA_NATIVE_AVAILABLE = False
    logger.warning("Ollama native web_search not available. Using DuckDuckGo fallback.")

try:
    from ddgs import DDGS
    DDGS_AVAILABLE = True
except ImportError:
    DDGS_AVAILABLE = False
    logger.error("ddgs not installed. Please run: uv add ddgs")


class SearchEngine:
    """Handles web search with fallback mechanisms."""
    
    def __init__(self, use_native: bool = USE_NATIVE_SEARCH):
        self.use_native = use_native and OLLAMA_NATIVE_AVAILABLE
        self.ddgs = DDGS() if DDGS_AVAILABLE else None
        
        if not self.use_native and not self.ddgs:
            raise RuntimeError("No search engine available. Install ddgs.")
    
    def search(self, query: str, max_results: int = MAX_SEARCH_RESULTS) -> List[Dict]:
        """
        Search the web and return formatted results.
        
        Args:
            query: Search query string
            max_results: Maximum number of results to return
            
        Returns:
            List of dicts with 'title', 'url', 'snippet' keys
        """
        logger.info("Searching for: %s", query)

        # Try native Ollama search first if enabled
        if self.use_native:
            try:
                results = ollama_web_search(query=query, max_results=max_results)
                if results:
                    logger.info("Found %d results via Ollama native", len(results))
                    return self._format_ollama_results(results)
            except Exception as e:
                logger.warning("Ollama native search failed: %s. Falling back to DuckDuckGo.", e)

        # Fallback to DuckDuckGo
        if self.ddgs:
            try:
                results = list(self.ddgs.text(query, max_results=max_results, timeout=SEARCH_TIMEOUT))
                if results:
                    logger.info("Found %d results via DuckDuckGo", len(results))
                    return self._format_ddgs_results(results)
            except Exception as e:
                logger.error("DuckDuckGo search failed: %s", e)
                return []
        
        logger.warning("No search results found.")
        return []
    
    def _format_ollama_results(self, results: List) -> List[Dict]:
        """Format Ollama native search results."""
        formatted = []
        for r in results:
            formatted.append({
                "title": r.get("title", "No title"),
                "url": r.get("url", ""),
                "snippet": r.get("content", r.get("snippet", ""))
            })
        return formatted
    
    def _format_ddgs_results(self, results: List) -> List[Dict]:
        """Format DuckDuckGo search results."""
        formatted = []
        for r in results:
            formatted.append({
                "title": r.get("title", "No title"),
                "url": r.get("href", r.get("link", "")),
                "snippet": r.get("body", r.get("snippet", ""))
            })
        return formatted
    
    def get_search_summary(self, results: List[Dict]) -> str:
        """Create a human-readable summary of search results."""
        if not results:
            return "No search results found."
        
        lines = []
        for i, r in enumerate(results[:5], 1):
            lines.append(f"{i}. **{r['title']}**\n   {r['snippet'][:200]}...\n   [{r['url']}]")
        
        return "\n\n".join(lines)