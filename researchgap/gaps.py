"""Build the method x domain grid and score the empty/thin cells as research gaps."""
from __future__ import annotations

from collections import Counter

import pandas as pd


def top_values(series: pd.Series, n: int, min_count: int = 1) -> list[str]:
    counts = series.value_counts()
    counts = counts[counts >= min_count]
    return [v for v in counts.index if v not in ("Unspecified", "")][:n]


def build_grid(df: pd.DataFrame, max_methods: int = 10, max_domains: int = 10):
    """Return (grid, methods, domains); grid is a DataFrame of paper counts."""
    methods = top_values(df["method"], max_methods, min_count=2)
    domains = top_values(df["domain"], max_domains, min_count=2)
    sub = df[df["method"].isin(methods) & df["domain"].isin(domains)]
    grid = (
        pd.crosstab(sub["method"], sub["domain"])
        .reindex(index=methods, columns=domains, fill_value=0)
    )
    return grid, methods, domains


def _recent_share(df: pd.DataFrame, mask: pd.Series, recent_from: int) -> float:
    s = df[mask]
    if s.empty:
        return 0.0
    return float((s["year"] >= recent_from).mean())


def _future_work_hits(df: pd.DataFrame, method: str, domain: str) -> int:
    """Count limitation statements in papers about this method OR this domain
    that mention the other axis. Cheap proxy for 'authors say this is unexplored'."""
    hits = 0
    m, d = method.lower(), domain.lower()
    for _, row in df.iterrows():
        text = " ".join(row["limitations"]).lower()
        if not text:
            continue
        if (row["method"] == method and d in text) or (row["domain"] == domain and m in text):
            hits += 1
    return hits


def score_gaps(df: pd.DataFrame, grid: pd.DataFrame, max_gaps: int = 15) -> list[dict]:
    """Rank cells with count <= 1 by a 0-100 confidence that they're a real gap.

    A zero cell isn't automatically interesting: it may be impossible or pointless.
    We boost cells whose row AND column are both active (neighbour support),
    whose row/column are growing recently, and which authors mention as future work.
    """
    if grid.empty:
        return []
    recent_from = int(df["year"].max()) - 2
    row_tot = grid.sum(axis=1)
    col_tot = grid.sum(axis=0)
    rmax, cmax = max(row_tot.max(), 1), max(col_tot.max(), 1)

    gaps = []
    for method in grid.index:
        for domain in grid.columns:
            count = int(grid.loc[method, domain])
            if count > 1:
                continue
            neighbour = (row_tot[method] / rmax + col_tot[domain] / cmax) / 2  # 0-1
            trend = (
                _recent_share(df, df["method"] == method, recent_from)
                + _recent_share(df, df["domain"] == domain, recent_from)
            ) / 2
            fw = _future_work_hits(df, method, domain)
            fw_score = min(fw, 3) / 3
            emptiness = 1.0 if count == 0 else 0.6
            score = 100 * emptiness * (0.5 * neighbour + 0.25 * trend + 0.25 * fw_score)
            gaps.append(
                {
                    "method": method,
                    "domain": domain,
                    "papers": count,
                    "score": round(score, 1),
                    "neighbour_activity": round(neighbour, 2),
                    "recent_trend": round(trend, 2),
                    "future_work_mentions": fw,
                }
            )
    gaps.sort(key=lambda g: g["score"], reverse=True)
    return gaps[:max_gaps]


def nearest_papers(df: pd.DataFrame, method: str, domain: str, n: int = 4) -> list[dict]:
    """Papers sharing either axis with the gap, best overlap first."""
    s = df[(df["method"] == method) | (df["domain"] == domain)].copy()
    s["_r"] = s["year"].fillna(0)
    return s.sort_values("_r", ascending=False).head(n)[["title", "year", "url"]].to_dict("records")


def limitation_themes(df: pd.DataFrame, n: int = 8) -> list[tuple[str, int]]:
    c = Counter(x.lower() for lim in df["limitations"] for x in lim)
    return c.most_common(n)
