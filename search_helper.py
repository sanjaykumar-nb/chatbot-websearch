from duckduckgo_search import DDGS

def web_search(query: str, max_results=3):
    """Get web results with DuckDuckGo"""
    with DDGS() as ddgs:
        results = [r for r in ddgs.text(query, max_results=max_results)]
        return [f"Source {i+1}: {res['body']}" for i, res in enumerate(results)]