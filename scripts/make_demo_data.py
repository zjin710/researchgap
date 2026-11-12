"""Generate a SYNTHETIC demo dataset so the app works offline with no API key.

The papers here are NOT real publications. They are seeded, illustrative records
with deliberately planted gaps, used only for Demo mode. Live mode uses OpenAlex + Gemini.
"""
import json
import random
from pathlib import Path

random.seed(7)

FIELD = "Student retention analytics"
METHODS = ["Random Forest", "Logistic Regression", "Deep Learning", "Survey", "Case Study", "Clustering", "Survival Analysis"]
DOMAINS = ["Undergraduate", "Online Learning", "STEM Courses", "Postgraduate", "Vocational Training", "Community College"]

# relative popularity of each (method, domain) cell; zeros are the planted gaps
W = {
    "Random Forest":       [9, 8, 6, 1, 0, 3],
    "Logistic Regression": [8, 6, 5, 2, 1, 4],
    "Deep Learning":       [6, 9, 4, 0, 0, 1],
    "Survey":              [7, 3, 4, 6, 4, 5],
    "Case Study":          [4, 2, 3, 5, 5, 4],
    "Clustering":          [5, 4, 2, 0, 1, 2],
    "Survival Analysis":   [2, 0, 1, 0, 0, 2],
}
REGIONS = ["Malaysia", "United States", "Global/Unspecified", "China", "United Kingdom", "India"]
LIMS = {
    "Postgraduate": "limited evidence for postgraduate cohorts",
    "Vocational Training": "future work should cover vocational training",
    "Survival Analysis": "survival analysis was not explored",
    "Deep Learning": "deep learning approaches left for future work",
    "Community College": "community college populations are under-represented",
}

papers = []
n = 0
for m, row in W.items():
    for d, w in zip(DOMAINS, row):
        for _ in range(w):
            n += 1
            lims = []
            if random.random() < 0.35:
                lims.append(random.choice(list(LIMS.values())))
            if random.random() < 0.3:
                lims.append("small single-institution sample")
            papers.append(
                {
                    "id": f"demo:{n}",
                    "title": f"[Sample #{n}] {m} applied to {d}",
                    "year": random.choice([2019, 2020, 2021, 2022, 2023, 2024, 2025, 2025, 2026]),
                    "abstract": "Synthetic demo record.",
                    "url": "https://example.org/synthetic-demo",
                    "method": m,
                    "domain": d,
                    "region": random.choice(REGIONS),
                    "limitations": lims,
                }
            )

out = Path(__file__).resolve().parent.parent / "data" / "demo_student_retention.json"
out.write_text(json.dumps({"field": FIELD, "synthetic": True, "papers": papers}, indent=1))
print(f"wrote {len(papers)} synthetic papers to {out}")
