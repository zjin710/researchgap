"""Fetch papers from the OpenAlex API (free, no key required)."""
from __future__ import annotations

import requests

API = "https://api.openalex.org/works"
MAILTO = "researchgap-hackathon@example.com"  # polite-pool identifier; replace with your team email


def _abstract_from_inverted(inv: dict | None) -> str:
    """OpenAlex ships abstracts as {word: [positions]}; rebuild the text."""
    if not inv:
        return ""
    words: list[tuple[int, str]] = []
    for word, positions in inv.items():
        for p in positions:
            words.append((p, word))
    return " ".join(w for _, w in sorted(words))


def fetch_papers(query: str, since_year: int = 2019, limit: int = 200) -> list[dict]:
    """Return up to `limit` papers matching `query` that have abstracts.

    Each paper: {id, title, year, abstract, url}
    """
    papers: list[dict] = []
    page = 1
    while len(papers) < limit and page <= 10:
        params = {
            "search": query,
            "filter": f"from_publication_date:{since_year}-01-01,has_abstract:true",
            "per-page": 100,
            "page": page,
            "select": "id,title,publication_year,abstract_inverted_index,doi",
            "mailto": MAILTO,
        }
        r = requests.get(API, params=params, timeout=30)
        r.raise_for_status()
        results = r.json().get("results", [])
        if not results:
            break
        for w in results:
            abstract = _abstract_from_inverted(w.get("abstract_inverted_index"))
            if len(abstract) < 200:
                continue
            papers.append(
                {
                    "id": w["id"],
                    "title": w.get("title") or "Untitled",
                    "year": w.get("publication_year"),
                    "abstract": abstract,
                    "url": w.get("doi") or w["id"],
                }
            )
        page += 1
    return papers[:limit]
