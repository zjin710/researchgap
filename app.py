"""ResearchGap - Streamlit app. Run: streamlit run app.py"""
import datetime
import json
import os
import re
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from researchgap.extract import BATCH_SIZE, friendly_error
from researchgap.gaps import build_grid, limitation_themes, nearest_papers, score_gaps
from researchgap.questions import suggest_questions, template_question

st.set_page_config(page_title="ResearchGap", page_icon="🔍", layout="wide")

DATA_DIR = Path(__file__).parent / "data"
TOP_Q = 5  # research questions drafted automatically (in ONE Gemini request)

st.title("🔍 ResearchGap")
st.caption("Find what hasn't been studied yet: a map of the white space in a research field.")


# ---------- helpers ----------
def list_datasets():
    """data/*.json files (synthetic sample + analyses saved from Live mode): {label: path}."""
    out = {}
    for p in sorted(DATA_DIR.glob("*.json")):
        try:
            d = json.loads(p.read_text())
            tag = "synthetic sample" if d.get("synthetic") else "saved real analysis"
            out[f"{d['field']} ({tag})"] = p
        except Exception:
            continue
    return out


@st.cache_data(show_spinner=False)
def load_dataset(path: str):
    d = json.loads(Path(path).read_text())
    return d["field"], pd.DataFrame(d["papers"]), d.get("questions", {})


def resolve_key(typed: str):
    """Typed key > Streamlit secrets > environment. Never written to os.environ (it would leak across visitors)."""
    if typed:
        return typed
    try:
        if "GEMINI_API_KEY" in st.secrets:
            return st.secrets["GEMINI_API_KEY"]
    except Exception:
        pass
    return os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")


def qkey(field, gap):
    return (field, gap["method"], gap["domain"])


# ---------- sidebar ----------
datasets = list_datasets()
with st.sidebar:
    st.header("Data source")
    mode = st.radio("Mode", ["Demo (saved data)", "Live (OpenAlex + Gemini)"])
    api_key = None
    if mode.startswith("Demo"):
        if not datasets:
            st.error("No demo data found in the data/ folder.")
            st.stop()
        dataset_label = st.selectbox("Dataset", list(datasets))
    else:
        field_in = st.text_input("Research field / keywords", "student retention prediction")
        limit = st.slider("Papers to analyse", 50, 200, 50, step=50)
        n_req = -(-limit // BATCH_SIZE) + 1  # abstract batches + 1 request for the research questions
        
        since = st.number_input("Published since", 2015, 2026, 2019)
        typed = st.text_input(
            "Gemini API key",
            type="password",
            value="",
            help="Used only for your session. Or set GEMINI_API_KEY in the environment / Streamlit secrets.",
        )
        api_key = resolve_key(typed)
        run = st.button("Analyse field", type="primary")
    st.divider()
    st.header("Grid size")
    max_m = st.slider("Max methods", 4, 14, 8)
    max_d = st.slider("Max domains", 4, 14, 8)


# ---------- live analysis ----------
def run_live(field, limit, since, api_key):
    from researchgap.extract import extract_papers
    from researchgap.openalex import fetch_papers

    with st.status("Fetching papers from OpenAlex...", expanded=True) as status:
        papers = fetch_papers(field, since_year=int(since), limit=limit)
        st.write(f"Found {len(papers)} papers with abstracts.")
        if not papers:
            status.update(label="No papers found", state="error")
            return None, None
        bar = st.progress(0.0, text="Extracting attributes with Gemini...")
        extracted, errors = extract_papers(
            field,
            papers,
            api_key=api_key,
            progress=lambda i, n: bar.progress(i / n, text=f"Extracting attributes with Gemini... {i}/{n} papers"),
        )
        status.update(label="Done", state="complete")
    notice = None
    if errors:
        notice = (
            f"Analysed {len(papers) - len(errors)} of {len(papers)} papers. {errors[0][:400]} "
            "Papers analysed so far are saved: run again later and only the missing ones are sent."
        )
    return pd.DataFrame(extracted), notice


q_store = st.session_state.setdefault("questions", {})
live = mode.startswith("Live")

if not live:
    path = datasets[dataset_label]
    field, df, saved_q = load_dataset(str(path))
    df = df.copy()
    for k, v in saved_q.items():  # questions saved with the dataset: shown with no Gemini request
        method, domain = k.split("|", 1)
        q_store.setdefault((field, method, domain), v)
    if "synthetic sample" in dataset_label:
        st.info("This is a **synthetic** dataset with planted gaps so the app works offline. "
                "Use Live mode to analyse real papers, then save the result as a dataset.", icon="ℹ️")
else:
    if run:
        if not api_key:
            st.error("Add a Gemini API key in the sidebar first.")
        else:
            try:
                df_live, notice = run_live(field_in, limit, since, api_key)
            except Exception as e:  # network / key / quota problems
                st.error(f"Live analysis failed: {type(e).__name__}: {e}")
                df_live, notice = None, None
            if df_live is not None:
                st.session_state["live"] = (field_in, df_live)
                st.session_state["live_notice"] = notice
                st.session_state.pop("q_error", None)
                st.session_state.setdefault("q_attempted", set()).discard(field_in)
    if "live" not in st.session_state:
        st.write("Enter a field, add your Gemini key, and press **Analyse field**.")
        st.stop()
    field, df = st.session_state["live"]
    df = df.copy()
    if st.session_state.get("live_notice"):
        st.warning(st.session_state["live_notice"])

df["year"] = pd.to_numeric(df["year"], errors="coerce").fillna(0).astype(int)

# ---------- grid + gaps ----------
grid, methods, domains = build_grid(df, max_m, max_d)
if grid.empty or grid.shape[0] < 2 or grid.shape[1] < 2:
    st.warning("Not enough overlapping methods/domains to build a grid. Try a broader field or more papers.")
    st.stop()
gaps = score_gaps(df, grid)

analysed = int((df["method"] != "Unspecified").sum())
c1, c2, c3 = st.columns(3)
c1.metric("Papers analysed", analysed if analysed == len(df) else f"{analysed} of {len(df)}")
c2.metric("Methods x domains", f"{grid.shape[0]} x {grid.shape[1]}")
c3.metric("Candidate gaps", len(gaps))

st.subheader(f"Method x Domain map: {field}")
gap_cells = {(g["method"], g["domain"]): g["score"] for g in gaps}
text = [[("" if grid.loc[m, d] else "gap") if (m, d) in gap_cells and grid.loc[m, d] == 0 else str(grid.loc[m, d])
         for d in grid.columns] for m in grid.index]
fig = go.Figure(
    go.Heatmap(
        z=grid.values,
        x=list(grid.columns),
        y=list(grid.index),
        text=text,
        texttemplate="%{text}",
        colorscale="Blues",
        hovertemplate="%{y} x %{x}<br>%{z} papers<extra></extra>",
        colorbar=dict(title="Papers"),
    )
)
# outline the top gaps
for (m, d), s in list(gap_cells.items())[:8]:
    xi, yi = list(grid.columns).index(d), list(grid.index).index(m)
    fig.add_shape(type="rect", x0=xi - .5, x1=xi + .5, y0=yi - .5, y1=yi + .5,
                  line=dict(color="#d9480f", width=3))
fig.update_layout(height=520, margin=dict(l=10, r=10, t=10, b=10), yaxis=dict(autorange="reversed"))
st.plotly_chart(fig, width="stretch")
st.caption("Orange outlines mark the highest-confidence gaps (cells with 0-1 papers where both the method and the domain are active elsewhere).")

# ---------- gap detail ----------
st.subheader("Top research gaps")
if not gaps:
    st.success("No thin cells found: this grid looks well covered.")
    st.stop()

# Draft questions for the top gaps with ONE Gemini request (Live mode only; free tier is ~20 requests/day).
if live and api_key:
    attempted = st.session_state.setdefault("q_attempted", set())
    need = [g for g in gaps[:TOP_Q] if qkey(field, g) not in q_store]
    if need and field not in attempted:
        attempted.add(field)
        try:
            with st.spinner(f"Drafting research questions for the top {len(need)} gaps (1 Gemini request)..."):
                texts = suggest_questions(field, [(g, nearest_papers(df, g["method"], g["domain"])) for g in need], api_key)
            for g, t in zip(need, texts):
                if t:
                    q_store[qkey(field, g)] = t
            st.session_state.pop("q_error", None)
        except Exception as e:
            st.session_state["q_error"] = friendly_error(str(e))

left, right = st.columns([2, 3])
with left:
    tbl = pd.DataFrame(gaps)[["method", "domain", "papers", "score", "future_work_mentions"]]
    tbl.columns = ["Method", "Domain", "Papers", "Confidence", "Future-work mentions"]
    st.dataframe(tbl, hide_index=True, width="stretch")
    labels = [f"{g['method']} x {g['domain']} ({g['score']})" for g in gaps]
    pick = st.selectbox("Explore a gap", labels)
gap = gaps[labels.index(pick)]

with right:
    st.markdown(f"### {gap['method']} x {gap['domain']}")
    a, b, c = st.columns(3)
    a.metric("Confidence", f"{gap['score']}/100")
    b.metric("Neighbour activity", gap["neighbour_activity"])
    c.metric("Recent growth", gap["recent_trend"])
    near = nearest_papers(df, gap["method"], gap["domain"])
    k = qkey(field, gap)
    if k in q_store:
        st.markdown(q_store[k])
    else:
        st.markdown(template_question(field, gap))
        st.caption("Template question (no Gemini request used).")
        if live and api_key and st.button("Draft this one with Gemini (1 request)"):
            try:
                with st.spinner("Drafting..."):
                    t = suggest_questions(field, [(gap, near)], api_key)[0]
                if t:
                    q_store[k] = t
                    st.rerun()
            except Exception as e:
                st.session_state["q_error"] = friendly_error(str(e))
    if st.session_state.get("q_error") and live:
        st.info(f"Gemini could not draft questions: {st.session_state['q_error']} Showing template questions instead.")
        if st.button("Try Gemini again"):
            st.session_state["q_attempted"].discard(field)
            st.session_state.pop("q_error", None)
            st.rerun()
    st.markdown("**Start reading:**")
    for p in near:
        st.markdown(f"- [{p['title']}]({p['url']}) ({p['year']})")
    st.caption("Confidence is a heuristic, not proof. A gap may exist because the pairing is impractical. "
               "Always verify with a full literature search.")

with st.expander("What authors say they left for future work"):
    for t, n in limitation_themes(df):
        st.write(f"- {t} ({n})")

with st.expander("Download / save data"):
    st.download_button("Gaps (CSV)", pd.DataFrame(gaps).to_csv(index=False), "gaps.csv")
    st.download_button("Papers (CSV)", df.drop(columns=["abstract"], errors="ignore").to_csv(index=False), "papers.csv")
    if live:
        st.caption("Save this analysis (with its research questions) so it appears in Demo mode with no API key. "
                   "Run this locally, then commit the new file in `data/` to GitHub.")
        if st.button("Save as demo dataset"):
            slug = re.sub(r"[^a-z0-9]+", "_", field.lower()).strip("_")[:40] or "field"
            out = DATA_DIR / f"saved_{slug}.json"
            DATA_DIR.mkdir(exist_ok=True)
            payload = {
                "field": field,
                "synthetic": False,
                "saved": datetime.date.today().isoformat(),
                "papers": json.loads(df.drop(columns=["abstract"], errors="ignore").to_json(orient="records")),
                "questions": {f"{m}|{d}": v for (f, m, d), v in q_store.items() if f == field},
            }
            out.write_text(json.dumps(payload, indent=1))
            load_dataset.clear()
            st.success(f"Saved to data/{out.name}. Switch to Demo mode to see it.")
