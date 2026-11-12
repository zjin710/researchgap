# ResearchGap

**Find what hasn't been studied yet.** ResearchGap maps the white space in a research field. It pulls recent papers from OpenAlex, uses Gemini to extract each paper's method, domain, region and stated limitations, then plots a **method x domain** heatmap where empty or thin cells are candidate research gaps. Click a gap to get a suggested research question, the evidence behind it, and papers to start reading.

Hackathon challenge area: **3. Data-Driven Research & Intelligent Decision Making**

## How it works

1. **Fetch**: `researchgap/openalex.py` queries the OpenAlex API (free, no key) for papers with abstracts.
2. **Extract**: `researchgap/extract.py` sends each abstract to Gemini with a JSON-output prompt and caches results on disk, so reruns are instant and cheap.
3. **Score**: `researchgap/gaps.py` builds the grid and ranks cells with 0-1 papers by a 0-100 confidence made of:
   - *neighbour activity*: both the method and the domain are busy elsewhere (an empty cell between two active axes is more interesting than one between dead ones)
   - *recent growth*: share of recent papers on either axis
   - *future-work mentions*: authors' own stated limitations that point at the gap
4. **Explain**: `researchgap/questions.py` asks Gemini for a research question, a rationale and a risk, grounded in that evidence (with an offline fallback).

**Honest limits:** the confidence score is a heuristic, and the app only sees the papers it fetched. A gap means "little in this sample", not "nobody has ever done it". Verify with a full literature search.

## Run locally

```bash
pip install -r requirements.txt
export GEMINI_API_KEY=your_key      # get one at https://aistudio.google.com/apikey
streamlit run app.py
```

- **Demo mode** works with no key and no internet. It uses a *synthetic* dataset (`data/demo_student_retention.json`, produced by `scripts/make_demo_data.py`) with planted gaps. Those records are not real publications.
- **Live mode** needs the key and internet access to `api.openalex.org`. The default model is `gemini-3.8-flash`; if Google retires it, set `RG_GEMINI_MODEL` to a current Flash model name from AI Studio.

**Built for the Gemini free tier, which is tiny** (about 5 requests/minute and about 20 requests/day per model on current keys):
- ~25 abstracts go into each request (50 papers = 2 requests), and the top research questions are drafted in 1 request.
- Every analysed paper is cached on disk, so re-running a field costs nothing and only missing papers are sent.
- If the daily quota runs out mid-run, the papers analysed so far are kept and shown; run again after it resets (the message says when).
- 429 per-minute replies are waited out, 503 overloads are retried, and other Flash models are tried if one is unavailable.
- Tunable via `RG_RPM` (0 on a paid plan), `RG_BATCH_SIZE`, `RG_TIMEOUT_MS`, `RG_MAX_ATTEMPTS`, `RG_GEMINI_MODEL`.

**Save real results for the demo:** in Live mode, open *Download / save data* and click *Save as demo dataset*. This writes `data/saved_<field>.json` (papers + research questions). Commit it to GitHub: in Demo mode it appears in the *Dataset* dropdown and works with no API key and no quota, which is how you should show the deployed app to judges.

Tests: `python -m pytest -q`

## Deploy (demo link)

Streamlit Community Cloud: push this repo (including your `data/saved_*.json` files) to GitHub and create an app pointing at `app.py`. Judges can use Demo mode with the saved datasets. Do **not** put your own key in *Secrets* unless you want every visitor's Live run to spend your quota; visitors can paste their own key into the Live sidebar instead. Never commit a key.

## Project layout

```
app.py                     Streamlit UI
researchgap/openalex.py    paper fetching
researchgap/extract.py     Gemini extraction + cache
researchgap/gaps.py        grid + gap scoring
researchgap/questions.py   research-question generation
scripts/make_demo_data.py  synthetic demo data
tests/                     unit tests for scoring and parsing
```

## Roadmap

More sources (Semantic Scholar, arXiv), a trend slider over publication years, a one-page PDF topic brief per gap, and institution-level dashboards for research offices.
