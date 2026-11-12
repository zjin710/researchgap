"""Turn scored gaps into suggested research questions.

Free-tier budget is tiny (~20 Gemini requests/day), so the top gaps are drafted in ONE
request (`suggest_questions`). `template_question` is the free offline fallback.
"""
from __future__ import annotations

import json

from .extract import _client, llm_generate

BATCH_PROMPT = """A research-gap analysis of the field "{field}" found {n} method x domain pairings that
the sampled literature covers thinly. For EACH one, write advice for a student choosing a thesis topic.

Return a JSON array with exactly {n} objects, in order, each with these keys:
- "n": the number shown in brackets (integer)
- "question": one concrete research question (1 sentence)
- "why": why it is worth studying (2 sentences, grounded in the evidence given)
- "risk": one risk that could explain why it is unexplored (1 sentence)

Be honest: never claim nobody has ever studied it, only that the sampled literature shows little.

{items}
"""


def _item(k: int, gap: dict, nearest: list[dict]) -> str:
    near = "; ".join(f"{p['title']} ({p['year']})" for p in nearest) or "none"
    return (
        f"[{k}] method: {gap['method']} | domain: {gap['domain']} | papers found: {gap['papers']}\n"
        f"    neighbour activity (0-1): {gap['neighbour_activity']}; recent growth (0-1): {gap['recent_trend']}; "
        f"authors' future-work mentions: {gap['future_work_mentions']}\n"
        f"    closest existing papers: {near}"
    )


def template_question(field: str, gap: dict) -> str:
    """Free offline fallback (no Gemini request)."""
    return (
        f"**Research question:** Can {gap['method']} be applied effectively to {gap['domain']} within {field}?\n\n"
        f"**Why:** The sampled literature has {gap['papers']} paper(s) on this pairing, while each side is active on its own "
        f"(neighbour activity {gap['neighbour_activity']}).\n\n"
        f"**Risk:** The pairing may be unexplored because suitable data is hard to obtain."
    )


def _format(entry: dict) -> str:
    return (
        f"**Research question:** {str(entry.get('question', '')).strip()}\n\n"
        f"**Why:** {str(entry.get('why', '')).strip()}\n\n"
        f"**Risk:** {str(entry.get('risk', '')).strip()}"
    )


def suggest_questions(field: str, items: list[tuple[dict, list[dict]]], api_key: str | None = None) -> list[str | None]:
    """ONE Gemini request for all `items` = [(gap, nearest_papers), ...].

    Returns markdown aligned with `items` (None where Gemini skipped one).
    Raises on request failure; the caller decides how to fall back.
    """
    from google.genai import types

    if not items:
        return []
    prompt = BATCH_PROMPT.format(
        field=field, n=len(items), items="\n\n".join(_item(k, g, n) for k, (g, n) in enumerate(items, start=1))
    )
    resp = llm_generate(_client(api_key), prompt, types.GenerateContentConfig(response_mime_type="application/json"))
    data = json.loads(resp.text)
    if isinstance(data, dict):
        data = next((v for v in data.values() if isinstance(v, list)), [])
    out: list[str | None] = [None] * len(items)
    for pos, entry in enumerate(data):
        if not isinstance(entry, dict) or not str(entry.get("question", "")).strip():
            continue
        try:
            k = int(entry.get("n", pos + 1)) - 1
        except (TypeError, ValueError):
            k = pos
        if 0 <= k < len(items):
            out[k] = _format(entry)
    return out
