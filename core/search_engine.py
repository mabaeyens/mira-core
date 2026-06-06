"""Search engine module with Ollama native, Brave, and DuckDuckGo fallback."""

import logging
import re
from typing import List, Dict, Optional
import httpx
from .config import MAX_SEARCH_RESULTS, SEARCH_TIMEOUT, USE_NATIVE_SEARCH, BRAVE_API_KEY

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
        self.brave_key = BRAVE_API_KEY
        self.ddgs = DDGS() if DDGS_AVAILABLE else None

        if not self.use_native and not self.brave_key and not self.ddgs:
            raise RuntimeError("No search engine available. Install ddgs or set BRAVE_API_KEY.")
    
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

        # Try Brave Search if API key is configured
        if self.brave_key:
            try:
                resp = httpx.get(
                    "https://api.search.brave.com/res/v1/web/search",
                    params={"q": query, "count": max_results},
                    headers={"Accept": "application/json", "X-Subscription-Token": self.brave_key},
                    timeout=SEARCH_TIMEOUT,
                )
                resp.raise_for_status()
                results = self._format_brave_results(resp.json())
                if results:
                    logger.info("Found %d results via Brave Search", len(results))
                    return results
            except Exception as e:
                logger.warning("Brave search failed: %s. Falling back to DuckDuckGo.", e)

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
    
    @staticmethod
    def _clean_snippet(text: str) -> str:
        """Normalize whitespace in a snippet — fixes concatenated words from HTML stripping."""
        return re.sub(r'\s+', ' ', text).strip()

    def _format_ollama_results(self, results: List) -> List[Dict]:
        """Format Ollama native search results."""
        formatted = []
        for r in results:
            formatted.append({
                "title": r.get("title", "No title"),
                "url": r.get("url", ""),
                "snippet": self._clean_snippet(r.get("content", r.get("snippet", "")))
            })
        return formatted

    @staticmethod
    def _format_brave_results(data: dict) -> List[Dict]:
        """Format Brave Search API results."""
        formatted = []
        for r in data.get("web", {}).get("results", []):
            formatted.append({
                "title": r.get("title", "No title"),
                "url": r.get("url", ""),
                "snippet": SearchEngine._clean_snippet(r.get("description", "")),
            })
        return formatted

    def _format_ddgs_results(self, results: List) -> List[Dict]:
        """Format DuckDuckGo search results."""
        formatted = []
        for r in results:
            formatted.append({
                "title": r.get("title", "No title"),
                "url": r.get("href", r.get("link", "")),
                "snippet": self._clean_snippet(r.get("body", r.get("snippet", "")))
            })
        return formatted
    
    def format_tool_result(self, results: List[Dict]) -> str:
        """Format search results for the model tool response — full snippets, URL co-located."""
        if not results:
            return "No search results found."
        lines = []
        for i, r in enumerate(results, 1):
            lines.append(
                f"[{i}] {r['title']}\n"
                f"URL: {r['url']}\n"
                f"Snippet: {r['snippet']}"
            )
        return "\n\n".join(lines)