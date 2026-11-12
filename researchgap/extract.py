"""Structured extraction of paper attributes with Gemini (JSON output) + disk cache.

Built for the Gemini FREE tier, which is tiny (about 5 requests/minute AND about
20 requests/DAY per model on current keys), so every request must count:
- BATCHING: ~25 abstracts per request (50 papers = 2 requests; 200 papers = 8);
- CACHE: papers analysed once are never sent again, even across runs;
- PACING: requests are spaced to stay under RG_RPM (default 4 per minute);
- 429 per-minute: wait for the retry delay Google reports, then retry;
- 429 per-day: do not wait; try another model, else stop and KEEP partial results;
- 503 / timeouts: short timeout + bounded retries, then fall back to other Flash models;
- a run that is failing everywhere stops early with a readable error.
Everything runs in the calling thread, so Streamlit progress updates are safe.
The API key is passed explicitly (never stored in os.environ), so visitors of a
deployed app cannot overwrite each other's keys.
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import re
import threading
import time
from pathlib import Path

CACHE_DIR = Path(os.environ.get("RG_CACHE_DIR", Path(__file__).resolve().parent.parent / "cache"))
# Model names change; Google retired gemini-2.5-flash for new users. Override with RG_GEMINI_MODEL.
MODEL = os.environ.get("RG_GEMINI_MODEL", "gemini-3.8-flash")

RPM = float(os.environ.get("RG_RPM", "4"))  # requests per minute we allow ourselves; 0 = unlimited (paid plans)
BATCH_SIZE = int(os.environ.get("RG_BATCH_SIZE", "25"))  # abstracts per request
ABSTRACT_CHARS = int(os.environ.get("RG_ABSTRACT_CHARS", "1000"))
TIMEOUT_MS = int(os.environ.get("RG_TIMEOUT_MS", "90000"))  # per request (batches are big)
MAX_ATTEMPTS = int(os.environ.get("RG_MAX_ATTEMPTS", "3"))  # per model
BACKOFF_S = 1.5  # doubles each attempt (for 5xx/timeouts)
MAX_WAIT_S = 70  # never sleep longer than this for one retry
MAX_FALLBACK_MODELS = 6
RETRYABLE = {429, 500, 502, 503, 504}

BATCH_PROMPT = """You are analysing {n} academic paper abstracts for a research-gap map of the field "{field}".

Return a JSON array with exactly {n} objects, one per paper, in order. Each object has these keys:
- "n": the paper number shown in brackets below (integer)
- "method": the main technique/approach, 1-3 words, Title Case, generic (e.g. "Random Forest", "Survey", "Deep Learning", "Case Study")
- "domain": the application area or population studied, 1-3 words, Title Case (e.g. "Healthcare", "Higher Education", "Banking")
- "region": country/region studied, or "Global/Unspecified"
- "limitations": list of up to 3 short strings: limitations or future-work the authors state (empty list if none stated)

Only use what each abstract says. Do not guess.

{papers}
"""

_lock = threading.Lock()
_chain: list[str] | None = None  # models to try, best first; None = not initialised
_next_slot = 0.0


class GeminiUnavailable(RuntimeError):
    """Every model we tried failed (overloaded, retired or out of quota)."""

    def __init__(self, message: str, daily_quota: bool = False):
        super().__init__(message)
        self.daily_quota = daily_quota


# ---------- cache ----------
def _key(field: str, paper_id: str) -> str:
    return hashlib.sha1(f"{MODEL}|{field}|{paper_id}".encode()).hexdigest()


def _cache_path(field: str, paper_id: str) -> Path:
    return CACHE_DIR / f"{_key(field, paper_id)}.json"


def _client(api_key: str | None = None):
    from google import genai

    key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not key:
        raise RuntimeError("Add a Gemini API key to run Live mode (or use Demo mode).")
    try:
        from google.genai import types

        opts = types.HttpOptions(timeout=TIMEOUT_MS, retry_options=types.HttpRetryOptions(attempts=1))
        return genai.Client(api_key=key, http_options=opts)
    except Exception:  # very old SDK without these options
        return genai.Client(api_key=key)


def _clean(d: dict) -> dict:
    out = {
        "method": str(d.get("method") or "Unspecified").strip().title(),
        "domain": str(d.get("domain") or "Unspecified").strip().title(),
        "region": str(d.get("region") or "Global/Unspecified").strip(),
        "limitations": [str(x).strip() for x in (d.get("limitations") or [])][:3],
    }
    return out


# ---------- error helpers ----------
def _code(e: Exception):
    return getattr(e, "code", None)


def _is_retryable(e: Exception) -> bool:
    return _code(e) in RETRYABLE or "timeout" in type(e).__name__.lower()


def _text_is_daily_quota(s: str) -> bool:
    """A per-DAY quota will not clear in a minute, so waiting is pointless."""
    return "perday" in s.lower().replace("_", "").replace(" ", "")


def _is_daily_quota(e: Exception) -> bool:
    return _code(e) == 429 and _text_is_daily_quota(str(e))


def _retry_delay_s(s: str):
    """Seconds Google asks us to wait ('Please retry in 11.5s' / retryDelay '11s'), or None."""
    m = re.search(r"retry in ([\d.]+)\s*s", s) or re.search(r"retryDelay['\"]?\s*:\s*['\"]?([\d.]+)s", s)
    return float(m.group(1)) if m else None


def _retry_delay(e: Exception):
    return _retry_delay_s(str(e))


def friendly_error(s: str) -> str:
    """Short human wording for the errors people actually hit; unknown errors pass through."""
    if "429" in s and _text_is_daily_quota(s):
        lim = re.search(r"limit: ?(\d+)", s)
        model = re.search(r"model: ?([\w.\-]+)", s)
        wait = _retry_delay_s(s)
        out = "429 Daily free-tier quota used up"
        if lim:
            out += f" ({lim.group(1)} requests/day" + (f" for {model.group(1).rstrip('.')})" if model else ")")
        if wait:
            mins = max(1, round(wait / 60))
            out += f"; it resets in about {mins} min" if mins < 120 else f"; it resets in about {mins / 60:.0f} h"
        return out + "."
    if "429" in s:
        return "429 Per-minute rate limit reached. " + s[:200]
    return s[:400]


def _wait_for_slot():
    """Space request starts at least 60/RPM seconds apart (shared by all callers)."""
    global _next_slot
    if RPM <= 0:
        return
    gap = 60.0 / RPM
    with _lock:
        now = time.monotonic()
        start = max(now, _next_slot)
        _next_slot = start + gap
    if start > now:
        time.sleep(start - now)


def _discover_fallbacks(client, exclude: str) -> list[str]:
    """Other Flash text models this key can call, newest-looking name first."""
    skip = ("image", "tts", "audio", "live", "embed", "vision", "robotics", "computer", "customtools")
    found: list[str] = []
    try:
        for m in client.models.list():
            name = (getattr(m, "name", "") or "").removeprefix("models/")
            actions = getattr(m, "supported_actions", None) or []
            if "gemini" not in name or "flash" not in name or name == exclude:
                continue
            if any(s in name for s in skip):
                continue
            if actions and "generateContent" not in actions:
                continue
            found.append(name)
    except Exception:
        return []
    return sorted(set(found), reverse=True)[:MAX_FALLBACK_MODELS]


# Models confirmed to have positive free-tier quotas
# Priority: higher daily quota first, then regular Flash models.
FREE_TIER_FALLBACK_PRIORITY = [
    "gemini-3.1-flash-lite",
    "gemini-3.5-flash-lite",
    "gemini-2.5-flash-lite",
    "gemini-3.6-flash",
    "gemini-3-flash",
    "gemini-2.5-flash",
]

def _discover_fallbacks(client, exclude: str) -> list[str]:
    """Discover supported Flash models with confirmed free-tier quotas."""

    skip = (
        "image", "tts", "audio", "live", "embed",
        "vision", "robotics", "computer", "customtools",
        "omni", "pro"
    )

    exclude = (exclude or "").removeprefix("models/").lower()

    found = set()

    try:
        for m in client.models.list():
            name = (
                getattr(m, "name", "") or ""
            ).removeprefix("models/")

            normalized = name.lower()

            if normalized == exclude:
                continue

            if normalized not in FREE_TIER_FALLBACK_PRIORITY:
                continue

            if any(s in normalized for s in skip):
                continue

            actions = getattr(m, "supported_actions", None) or []

            if actions and "generateContent" not in actions:
                continue

            found.add(normalized)

    except Exception as e:
        print(f"Model discovery failed: {e}")
        return []

    # Preserve explicitly configured priority
    ordered = [
        model
        for model in FREE_TIER_FALLBACK_PRIORITY
        if model in found
    ]

    return ordered[:MAX_FALLBACK_MODELS]


def llm_generate(client, contents, config=None):
    """generate_content with pacing, bounded retries and model fallback. Never waits unboundedly.

    Raises GeminiUnavailable (a RuntimeError) summarising every model's failure, or the
    original error for problems retrying cannot fix (bad key, bad request).
    """
    global _chain
    with _lock:
        if _chain is None:
            _chain = [MODEL]
        chain = list(_chain)

    last: Exception | None = None
    failures: list[tuple[str, Exception]] = []
    i = 0
    while i < len(chain):
        model = chain[i]
        for attempt in range(MAX_ATTEMPTS):
            _wait_for_slot()
            try:
                resp = client.models.generate_content(model=model, contents=contents, config=config)
                if i > 0:  # a fallback worked: try it first from now on
                    with _lock:
                        _chain = [model] + [m for m in chain if m != model]
                return resp
            except Exception as e:
                last = e
                if _is_daily_quota(e):
                    break
                if _is_retryable(e) and attempt < MAX_ATTEMPTS - 1:
                    d = _retry_delay(e) if _code(e) == 429 else None
                    wait = min(d + 1, MAX_WAIT_S) if d else BACKOFF_S * (2 ** attempt) + random.random() * 0.5
                    time.sleep(wait)
                    continue
                break
        # this model is exhausted (or permanently unavailable)
        if _is_retryable(last) or _code(last) == 404:
            failures.append((model, last))
            if len(chain) == 1:  # discover fallbacks once, only when needed
                chain += _discover_fallbacks(client, exclude=chain[0])
                with _lock:
                    _chain = list(chain)
            i += 1
            continue
        raise last  # bad key, bad request, etc.: retrying other models won't help
    summary = "; ".join(f"{m}: {friendly_error(str(e))}" for m, e in failures)
    raise GeminiUnavailable(summary, daily_quota=any(_is_daily_quota(e) for _, e in failures)) from last


# ---------- extraction ----------
def _format_batch(papers: list[dict]) -> str:
    return "\n\n".join(
        f"[{k}] Title: {p['title']}\nAbstract: {p['abstract'][:ABSTRACT_CHARS]}" for k, p in enumerate(papers, start=1)
    )


def _parse_batch(text: str, count: int) -> list[dict | None]:
    """Map Gemini's JSON array back to papers by the 'n' field. Missing/invalid items become None."""
    data = json.loads(text)
    if isinstance(data, dict):  # tolerate {"papers": [...]} style wrappers
        data = next((v for v in data.values() if isinstance(v, list)), [])
    out: list[dict | None] = [None] * count
    for pos, item in enumerate(data):
        if not isinstance(item, dict):
            continue
        try:
            k = int(item.get("n", pos + 1)) - 1
        except (TypeError, ValueError):
            k = pos
        if 0 <= k < count:
            out[k] = _clean(item)
    return out


def extract_batch(client, field: str, papers: list[dict]) -> list[dict | None]:
    """One request for up to BATCH_SIZE papers. Returns attributes aligned with `papers`."""
    from google.genai import types

    resp = llm_generate(
        client,
        BATCH_PROMPT.format(n=len(papers), field=field, papers=_format_batch(papers)),
        # No temperature/top_p/top_k: Gemini 3.x docs say to remove these sampling parameters.
        types.GenerateContentConfig(response_mime_type="application/json"),
    )
    try:
        return _parse_batch(resp.text, len(papers))
    except Exception as e:
        raise ValueError(f"Gemini returned unparsable JSON ({type(e).__name__})") from e


def extract_papers(field: str, papers: list[dict], progress=None, batch_size: int | None = None, api_key: str | None = None):
    """Extract attributes for every paper.

    Returns (results, errors). `results` is every paper merged with its attributes
    (papers that could not be analysed get "Unspecified"). `errors` has one readable
    string per paper that was not analysed; errors[0] is the real cause.

    If Gemini fails (daily quota, overload...) after SOME papers were analysed or
    cached, the partial results are returned, not thrown away: re-run later and only
    the missing papers are sent. Raises RuntimeError only if nothing at all could be
    analysed. Papers already cached are never re-sent.

    `progress(done, total)` is called from the calling thread after each batch.
    """
    client = _client(api_key)
    bs = batch_size or BATCH_SIZE
    results: list[dict | None] = [None] * len(papers)
    errors: list[str] = []

    todo: list[int] = []
    for i, p in enumerate(papers):
        cp = _cache_path(field, p["id"])
        if cp.exists():
            results[i] = {**p, **json.loads(cp.read_text())}
        else:
            todo.append(i)
    good = len(papers) - len(todo)
    done = good
    if progress:
        progress(done, len(papers))

    consecutive_failures = 0
    stop_reason: str | None = None
    for start in range(0, len(todo), bs):
        idxs = todo[start : start + bs]
        if stop_reason:  # an earlier batch showed Gemini is out of reach: don't burn more requests
            for i in idxs:
                results[i] = {**papers[i], **_clean({})}
                errors.append(f"Not analysed (stopped early): {stop_reason}")
            done += len(idxs)
            if progress:
                progress(done, len(papers))
            continue

        batch_err = None
        daily = False
        try:
            parsed = extract_batch(client, field, [papers[i] for i in idxs])
        except Exception as e:  # whole request failed
            parsed = [None] * len(idxs)
            raw = str(e) if isinstance(e, GeminiUnavailable) else f"{type(e).__name__}: {friendly_error(str(e))}"
            batch_err = raw
            daily = getattr(e, "daily_quota", False) or _is_daily_quota(e)

        consecutive_failures = consecutive_failures + 1 if batch_err else 0
        for i, attrs in zip(idxs, parsed):
            if attrs is None:
                errors.append(batch_err or "ValueError: paper missing from Gemini's response")
                results[i] = {**papers[i], **_clean({})}
            else:
                CACHE_DIR.mkdir(parents=True, exist_ok=True)
                _cache_path(field, papers[i]["id"]).write_text(json.dumps(attrs))
                results[i] = {**papers[i], **attrs}
                good += 1
        done += len(idxs)
        if progress:
            progress(done, len(papers))
        if batch_err and (daily or consecutive_failures >= 2):
            stop_reason = batch_err

    if good == 0 and papers:
        raise RuntimeError(f"Gemini is failing on every request. First error: {errors[0]}")
    return results, errors
