"""Tool definitions for Ollama API."""

SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": "Search the web for current information. Use this when the user asks about recent events, news, prices, or anything after April 2024.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query string"
                },
                "num_results": {
                    "type": "integer",
                    "description": "Number of results to return (default 5)",
                    "default": 5
                }
            },
            "required": ["query"]
        }
    }
}

FETCH_TOOL = {
    "type": "function",
    "function": {
        "name": "fetch_url",
        "description": (
            "Fetch the full text content of a web page. "
            "Use this when a web_search result looks relevant but the snippet "
            "is too short to answer the question — fetch the page to read the details."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "The full URL to fetch"
                }
            },
            "required": ["url"]
        }
    }
}

TOOLS = [SEARCH_TOOL, FETCH_TOOL]