import json

import pytest

import researchgap.questions as q


def gap(method, domain, papers=0):
    return {"method": method, "domain": domain, "papers": papers, "score": 50.0,
            "neighbour_activity": 0.5, "recent_trend": 0.4, "future_work_mentions": 2}


NEAR = [{"title": "Some paper", "year": 2024, "url": "u"}]


class Resp:
    def __init__(self, text):
        self.text = text


def patch_llm(monkeypatch, reply):
    calls = []

    class Models:
        def generate_content(self, model, contents, config=None):
            calls.append(contents)
            return Resp(reply(contents))

        def list(self):
            return []

    class Client:
        models = Models()

    monkeypatch.setattr(q, "_client", lambda api_key=None: Client())
    return calls


def test_many_gaps_cost_one_request(monkeypatch):
    calls = patch_llm(
        monkeypatch,
        lambda c: json.dumps([{"n": k, "question": f"Q{k}?", "why": f"W{k}.", "risk": f"R{k}."} for k in range(1, 6)]),
    )
    items = [(gap(f"M{k}", f"D{k}"), NEAR) for k in range(1, 6)]
    out = q.suggest_questions("field", items, api_key="k")
    assert len(calls) == 1  # one request for all five gaps
    assert [o.splitlines()[0] for o in out] == [f"**Research question:** Q{k}?" for k in range(1, 6)]
    assert "**Why:** W3." in out[2] and "**Risk:** R3." in out[2]
    assert all(f"method: M{k}" in calls[0] for k in range(1, 6))


def test_missing_or_reordered_answers_are_aligned(monkeypatch):
    patch_llm(
        monkeypatch,
        lambda c: json.dumps([{"n": 3, "question": "Q3?", "why": "w", "risk": "r"}, {"n": 1, "question": "Q1?", "why": "w", "risk": "r"}]),
    )
    out = q.suggest_questions("f", [(gap("A", "B"), NEAR)] * 3, api_key="k")
    assert out[0].startswith("**Research question:** Q1?") and out[1] is None and out[2].startswith("**Research question:** Q3?")


def test_request_failure_is_raised_for_the_caller(monkeypatch):
    def boom(c):
        raise ValueError("nope")

    patch_llm(monkeypatch, boom)
    with pytest.raises(ValueError):
        q.suggest_questions("f", [(gap("A", "B"), NEAR)], api_key="k")


def test_template_needs_no_gemini():
    t = q.template_question("my field", gap("Survey", "Banking", papers=1))
    assert "Survey" in t and "Banking" in t and "my field" in t
