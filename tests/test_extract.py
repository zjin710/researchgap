import json
import re

import pytest
from google.genai import errors

import researchgap.extract as ex

# The real 429 your free-tier key returned (shortened): includes the retry hint Google sends.
REAL_429 = (
    "You exceeded your current quota. Quota exceeded for metric: "
    "generativelanguage.googleapis.com/generate_content_free_tier_requests, limit: 5, model: gemini-3.8-flash. "
    "Please retry in 11.526240183s."
)


# The real per-DAY 429 your free-tier key returned (shortened).
DAILY_429 = (
    "You exceeded your current quota. Quota exceeded for metric: "
    "generativelanguage.googleapis.com/generate_content_free_tier_requests, limit: 20, model: gemini-3.8-flash, "
    "description: Generate content requests per day limit exceeded on Free Tier. "
    "Please retry in 1h4m36.891342643s. quotaId: GenerateRequestsPerDayPerProjectPerModel-FreeTier "
    "details: [{'retryDelay': '3876s'}]"
)


def papers(n, start=0):
    return [{"id": f"W{i}", "title": f"t{i}", "year": 2024, "abstract": "x" * 300, "url": "u"} for i in range(start, start + n)]


class Resp:
    def __init__(self, text):
        self.text = text


def good_reply(contents, drop=()):
    """Answer like Gemini: a JSON array with one object per '[k] Title:' in the prompt."""
    ks = [int(k) for k in re.findall(r"^\[(\d+)\] Title:", contents, re.M)]
    return json.dumps(
        [
            {"n": k, "method": "Random Forest", "domain": "Healthcare", "region": "Malaysia", "limitations": ["small sample"]}
            for k in ks
            if k not in drop
        ]
    )


def api_error(code, message="simulated"):
    cls = errors.ServerError if code >= 500 else errors.ClientError
    return cls(code, {"error": {"code": code, "message": message, "status": "X"}}, None)


def patch_client(monkeypatch, behaviour=lambda model, n, contents: None, reply=good_reply, listed=("gemini-3.7-flash", "gemini-3.7-flash-image")):
    """behaviour(model, call_no, contents) may raise. Returns the list of models called."""
    log = []

    class M:
        def __init__(self, name):
            self.name = f"models/{name}"
            self.supported_actions = ["generateContent"]

    class Models:
        def generate_content(self, model, contents, config=None):
            log.append(model)
            behaviour(model, len(log), contents)
            return Resp(reply(contents))

        def list(self):
            return [M(n) for n in listed]

    class Client:
        models = Models()

    monkeypatch.setattr(ex, "_client", lambda api_key=None: Client())
    return log


# ---------- batching (the fix for the 5-requests-per-minute free tier) ----------
def test_batches_many_papers_per_request(monkeypatch):
    log = patch_client(monkeypatch)
    results, errs = ex.extract_papers("f", papers(25), batch_size=10)
    assert len(log) == 3  # 25 papers -> 3 requests, not 25
    assert errs == [] and len(results) == 25
    assert all(r["method"] == "Random Forest" and r["limitations"] == ["small sample"] for r in results)


def test_default_batch_size_keeps_50_papers_within_free_tier(monkeypatch):
    log = patch_client(monkeypatch)
    ex.extract_papers("f", papers(50))
    assert len(log) <= 2  # free tier allows only ~20 requests per DAY, so 50 papers must cost very few


def test_results_stay_aligned_when_gemini_reorders(monkeypatch):
    def reordered(contents):
        ks = [int(k) for k in re.findall(r"^\[(\d+)\] Title:", contents, re.M)]
        return json.dumps([{"n": k, "method": f"M{k}", "domain": "D", "region": "R", "limitations": []} for k in reversed(ks)])

    patch_client(monkeypatch, reply=reordered)
    results, _ = ex.extract_papers("f", papers(4))
    assert [r["method"] for r in results] == ["M1", "M2", "M3", "M4"]


def test_cached_papers_are_not_resent(monkeypatch):
    log = patch_client(monkeypatch)
    ex.extract_papers("f", papers(10))
    assert len(log) == 1
    ex.extract_papers("f", papers(10))
    assert len(log) == 1  # second run: everything from cache
    ex.extract_papers("f", papers(10) + papers(3, start=10))
    assert len(log) == 2  # only the 3 new papers sent


def test_progress_reports_each_batch_in_order(monkeypatch):
    patch_client(monkeypatch)
    seen = []
    ex.extract_papers("f", papers(25), batch_size=10, progress=lambda d, n: seen.append((d, n)))
    assert seen == [(0, 25), (10, 25), (20, 25), (25, 25)]


def test_missing_items_are_errors_and_not_cached(monkeypatch, tmp_path):
    patch_client(monkeypatch, reply=lambda c: good_reply(c, drop=(2,)))
    results, errs = ex.extract_papers("f", papers(3))
    assert len(errs) == 1 and "missing" in errs[0]
    assert [r["method"] for r in results] == ["Random Forest", "Unspecified", "Random Forest"]
    assert len(list(tmp_path.glob("*.json"))) == 2  # the missing one will be retried next run


def test_unparsable_json_is_error_and_not_cached(monkeypatch, tmp_path):
    patch_client(monkeypatch, reply=lambda c: "not json")
    with pytest.raises(RuntimeError, match="unparsable"):
        ex.extract_papers("f", papers(3))
    assert list(tmp_path.glob("*.json")) == []


# ---------- 429 / quota ----------
def test_real_429_message_is_parsed():
    assert ex._retry_delay(api_error(429, REAL_429)) == pytest.approx(11.526, rel=1e-3)
    assert ex._retry_delay(ValueError("nothing here")) is None
    assert ex._retry_delay(Exception("{'retryDelay': '11s'}")) == 11


def test_429_waits_the_retry_delay_then_succeeds(monkeypatch, fast):
    def first_429(model, n, contents):
        if n == 1:
            raise api_error(429, REAL_429)

    log = patch_client(monkeypatch, first_429)
    results, errs = ex.extract_papers("f", papers(3))
    assert errs == [] and len(log) == 2 and log[0] == log[1] == ex.MODEL
    assert any(11.5 < s < 13 for s in fast)  # waited about 11.5 s + 1 s, as Google asked


def test_daily_quota_is_not_waited_out_and_falls_back(monkeypatch, fast):
    def daily(model, n, contents):
        if model == ex.MODEL:
            raise api_error(429, "quotaId: GenerateRequestsPerDayPerProjectPerModel-FreeTier. Please retry in 30s")

    log = patch_client(monkeypatch, daily)
    results, errs = ex.extract_papers("f", papers(3))
    assert errs == [] and log == [ex.MODEL, "gemini-3.7-flash"]
    assert not any(s > 5 for s in fast)  # no long wait


def test_pacing_spaces_requests(monkeypatch, fast):
    monkeypatch.setattr(ex, "RPM", 4)  # one request every 15 s
    clock = {"t": 1000.0}
    monkeypatch.setattr(ex.time, "monotonic", lambda: clock["t"])
    monkeypatch.setattr(ex.time, "sleep", lambda s: (fast.append(s), clock.__setitem__("t", clock["t"] + s)))
    patch_client(monkeypatch)
    ex.extract_papers("f", papers(30), batch_size=10)  # 3 requests
    assert [round(s) for s in fast] == [15, 15]  # first goes immediately, next two wait 15 s each


# ---------- overload / retired models / bad key ----------
def test_transient_503_is_retried_on_same_model(monkeypatch):
    def once(model, n, contents):
        if n == 1:
            raise api_error(503)

    log = patch_client(monkeypatch, once)
    _, errs = ex.extract_papers("f", papers(1))
    assert errs == [] and log == [ex.MODEL, ex.MODEL]


def test_overloaded_primary_falls_back_and_remembers(monkeypatch):
    def primary_down(model, n, contents):
        if model == ex.MODEL:
            raise api_error(503)

    log = patch_client(monkeypatch, primary_down)
    _, errs = ex.extract_papers("f", papers(30), batch_size=10)
    assert errs == []
    assert set(log) == {ex.MODEL, "gemini-3.7-flash"}  # the -image variant was filtered out
    assert log.count(ex.MODEL) == ex.MAX_ATTEMPTS  # later batches go straight to the working model


def test_retired_model_404_falls_back(monkeypatch):
    def gone(model, n, contents):
        if model == ex.MODEL:
            raise api_error(404)

    log = patch_client(monkeypatch, gone)
    _, errs = ex.extract_papers("f", papers(2))
    assert errs == [] and log.count(ex.MODEL) == 1  # 404 is not retried


def test_timeouts_are_retryable(monkeypatch):
    class ReadTimeout(Exception):
        pass

    def slow(model, n, contents):
        if n == 1:
            raise ReadTimeout("read timed out")

    log = patch_client(monkeypatch, slow)
    _, errs = ex.extract_papers("f", papers(1))
    assert errs == [] and len(log) == 2


def test_everything_down_fails_fast_not_forever(monkeypatch):
    def down(model, n, contents):
        raise api_error(503)

    log = patch_client(monkeypatch, down)
    with pytest.raises(RuntimeError, match="503"):
        ex.extract_papers("f", papers(50), batch_size=10)
    # gave up after 2 failed batches (each: main model + one fallback model, MAX_ATTEMPTS tries each),
    # instead of grinding through all 5 batches
    assert len(log) == 2 * 2 * ex.MAX_ATTEMPTS


def test_bad_key_is_not_retried_across_models(monkeypatch):
    def denied(model, n, contents):
        raise api_error(403)

    log = patch_client(monkeypatch, denied)
    with pytest.raises(RuntimeError, match="403"):
        ex.extract_papers("f", papers(3))
    assert log == [ex.MODEL]  # one call, no retries, no fallbacks


# ---------- per-day quota: keep partial results, never burn more requests ----------
def test_daily_quota_mid_run_keeps_partial_results(monkeypatch):
    def quota_after_two(model, n, contents):
        if n > 2:
            raise api_error(429, DAILY_429)

    log = patch_client(monkeypatch, quota_after_two)
    results, errs = ex.extract_papers("f", papers(50), batch_size=10)  # must NOT raise
    assert [r["method"] for r in results[:20]] == ["Random Forest"] * 20  # first 2 batches kept
    assert all(r["method"] == "Unspecified" for r in results[20:])
    assert len(errs) == 30
    assert "Daily free-tier quota used up" in errs[0] and "20 requests/day" in errs[0] and "65 min" in errs[0]
    assert len(log) == 4  # 2 good + 1 failed batch tried on 2 models; batches 4 and 5 never sent


def test_rerun_after_quota_resets_sends_only_missing_papers(monkeypatch):
    state = {"blocked": True}

    def quota(model, n, contents):
        if state["blocked"] and n > 2:
            raise api_error(429, DAILY_429)

    log = patch_client(monkeypatch, quota)
    ex.extract_papers("f", papers(50), batch_size=10)
    state["blocked"] = False
    before = len(log)
    results, errs = ex.extract_papers("f", papers(50), batch_size=10)
    assert errs == [] and all(r["method"] == "Random Forest" for r in results)
    assert len(log) - before == 3  # only the 30 missing papers, as 3 batches


def test_daily_quota_with_nothing_analysed_raises_readable_error(monkeypatch):
    def quota(model, n, contents):
        raise api_error(429, DAILY_429)

    patch_client(monkeypatch, quota)
    with pytest.raises(RuntimeError, match="Daily free-tier quota used up"):
        ex.extract_papers("f", papers(30), batch_size=10)


def test_friendly_error_wording():
    msg = ex.friendly_error(f"429 RESOURCE_EXHAUSTED. {DAILY_429}")
    assert msg == "429 Daily free-tier quota used up (20 requests/day for gemini-3.8-flash); it resets in about 65 min."
    assert ex.friendly_error("503 UNAVAILABLE high demand").startswith("503")
    assert "Per-minute" in ex.friendly_error("429 RESOURCE_EXHAUSTED. Please retry in 11.5s")


# ---------- api key handling ----------
def test_key_is_never_stored_in_environment(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    import os

    ex._client("visitor-key-123")
    assert "GEMINI_API_KEY" not in os.environ and "GOOGLE_API_KEY" not in os.environ
    with pytest.raises(RuntimeError, match="API key"):
        ex._client(None)
