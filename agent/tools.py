"""
Implements simple tools:
- WebSearchTool: stub uses Wikipedia REST (or your own function) to keep it keyless by default.
- RetrievalTool: BM25 over candidate_contexts.
- CalculatorTool: eval simple expressions (safe subset).

Each tool exposes:
    name (str)
    call(query: str, **kwargs) -> str
"""
from rank_bm25 import BM25Okapi
from typing import List, Dict
import math, re, requests

class WebSearchTool:
    name = "web_search"
    def call(self, query: str, topk: int = 3) -> str:
        # Simple, keyless Wikipedia summary as a placeholder:
        # For real search, plug SerpAPI/Bing here.
        try:
            r = requests.get(f"https://en.wikipedia.org/api/rest_v1/page/summary/{query.replace(' ', '%20')}", timeout=5)
            if r.status_code == 200:
                js = r.json()
                return js.get("extract", "")[:1200]
        except Exception:
            pass
        return "No results."

class RetrievalTool:
    name = "retrieve"
    def __init__(self, corpus: List[str]):
        self.corpus = corpus
        self.tokenized = [c.split() for c in corpus]
        self.bm25 = BM25Okapi(self.tokenized)

    def call(self, query: str, topk: int = 3) -> str:
        scores = self.bm25.get_scores(query.split())
        idxs = sorted(range(len(scores)), key=lambda i: -scores[i])[:topk]
        chunks = [self.corpus[i] for i in idxs]
        return "\n".join(chunks)

class CalculatorTool:
    name = "calculator"
    def call(self, expr: str) -> str:
        if not re.fullmatch(r"[0-9\+\-\*\/\.\(\) ]{1,100}", expr):
            return "Unsafe expression."
        try:
            return str(eval(expr, {"__builtins__": {}}, {}))
        except Exception:
            return "Math error."
