import pytest

import researchgap.extract as ex


@pytest.fixture(autouse=True)
def fast(monkeypatch, tmp_path):
    """No real sleeping, no pacing, fresh cache + model chain for every test. Returns the recorded sleeps."""
    monkeypatch.setattr(ex, "RPM", 0)  # pacing off unless a test turns it on
    monkeypatch.setattr(ex, "BACKOFF_S", 0)
    monkeypatch.setattr(ex, "_chain", None)
    monkeypatch.setattr(ex, "_next_slot", 0.0)
    monkeypatch.setattr(ex, "CACHE_DIR", tmp_path)
    sleeps = []
    monkeypatch.setattr(ex.time, "sleep", lambda s: sleeps.append(s))
    return sleeps
