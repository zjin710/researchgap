import pandas as pd

from researchgap.gaps import build_grid, score_gaps
from researchgap.openalex import _abstract_from_inverted


def make_df():
    rows = []
    def add(m, d, n, year=2024, lim=None):
        for i in range(n):
            rows.append({"id": f"{m}{d}{i}", "title": "t", "year": year, "url": "u",
                         "method": m, "domain": d, "limitations": lim or []})
    add("A", "X", 5)
    add("A", "Y", 4)
    add("B", "X", 4)
    add("B", "Y", 0)           # planted gap: both neighbours active
    add("C", "Z", 3)           # C and Z only meet each other
    add("A", "Z", 2)
    add("B", "Z", 2)
    add("C", "X", 2)
    add("C", "Y", 2, lim=["future work on method B in domain Y"])
    return pd.DataFrame(rows)


def test_grid_counts():
    df = make_df()
    grid, methods, domains = build_grid(df)
    assert grid.loc["A", "X"] == 5
    assert grid.loc["B", "Y"] == 0


def test_gap_found_and_ranked():
    df = make_df()
    grid, _, _ = build_grid(df)
    gaps = score_gaps(df, grid)
    assert gaps, "expected at least one gap"
    assert (gaps[0]["method"], gaps[0]["domain"]) == ("B", "Y")
    assert all(0 <= g["score"] <= 100 for g in gaps)
    assert gaps == sorted(gaps, key=lambda g: g["score"], reverse=True)


def test_future_work_boosts_score():
    df = make_df()
    grid, _, _ = build_grid(df)
    with_fw = score_gaps(df, grid)[0]["score"]
    df2 = df.copy()
    df2["limitations"] = [[] for _ in range(len(df2))]
    without_fw = score_gaps(df2, grid)[0]["score"]
    assert with_fw > without_fw


def test_dense_cells_not_gaps():
    df = make_df()
    grid, _, _ = build_grid(df)
    assert not any(g["papers"] > 1 for g in score_gaps(df, grid))


def test_abstract_rebuild():
    inv = {"hello": [0], "world": [1], "again": [2]}
    assert _abstract_from_inverted(inv) == "hello world again"
    assert _abstract_from_inverted(None) == ""
