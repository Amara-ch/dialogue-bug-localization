"""
Streamlit demo: Dialogue-Driven Bug Localization.

Run: streamlit run app.py
"""
import streamlit as st
import pandas as pd

from src.inference import get_localizer, DEFAULT_ALPHA

st.set_page_config(
    page_title="Dialogue Bug Localizer",
    page_icon="🔍",
    layout="wide",
)

st.title("🔍 Dialogue-Driven Bug Localization")
st.caption("Hybrid CodeBERT + Graph Neural Network + BM25 — final-year research demo")

# ---- Load model (cached) ----
@st.cache_resource(show_spinner="Loading model (first time ~30s)...")
def load_model():
    return get_localizer()

loc = load_model()

# ---- Sidebar ----
with st.sidebar:
    st.header("Settings")
    repo = st.selectbox("Repository", loc.available_repos())
    top_k = st.slider("Top-K files", 3, 15, 5)
    alpha = st.slider(
        "Neural weight (α)",
        0.0, 1.0, DEFAULT_ALPHA, 0.1,
        help="0 = pure BM25, 1 = pure neural model, 0.4 = best from our sweep",
    )

    with st.expander(f"📁 Files in {repo}"):
        for f in loc.repo_files(repo):
            st.text(f)

# ---- Main ----
st.subheader("🐛 Bug Description (dialogue / issue text)")

examples = {
    "(empty)": "",
    "DataFrame merge bug": (
        "When merging two DataFrames using pd.merge with how='outer', "
        "the resulting DataFrame loses the original index order and "
        "raises a KeyError on duplicate column names."
    ),
    "Flask routing error": (
        "Flask app returns 404 for URL patterns containing trailing slashes "
        "when strict_slashes=False is set in the route decorator."
    ),
    "Requests SSL issue": (
        "requests.get() raises SSLError when verifying certificates "
        "behind a corporate proxy, even with verify=True."
    ),
}

choice = st.selectbox("Load example", list(examples.keys()))
dialogue = st.text_area(
    "Describe the bug:",
    value=examples[choice],
    height=180,
    placeholder="e.g. The login button doesnt work when CSRF token expires...",
)

go = st.button("🔍 Locate Bug Files", type="primary", use_container_width=True)

if go:
    if not dialogue.strip():
        st.warning("Please enter a bug description.")
    else:
        with st.spinner("Running hybrid inference..."):
            results = loc.predict(dialogue, repo, top_k=top_k, alpha=alpha)

        st.success(f"Top {len(results)} suspected files in **{repo}**")

        df = pd.DataFrame(results)
        df.index = [f"#{i+1}" for i in range(len(df))]
        df_disp = df.rename(columns={
            "file": "File path",
            "score": "Hybrid score",
            "neural": "Neural",
            "bm25": "BM25",
        })
        st.dataframe(
            df_disp.style.format({
                "Hybrid score": "{:.3f}",
                "Neural": "{:.3f}",
                "BM25": "{:.3f}",
            }).background_gradient(subset=["Hybrid score"], cmap="Greens"),
            use_container_width=True,
        )

        st.subheader("📊 Score Comparison")
        chart_df = df[["file", "neural", "bm25", "score"]].set_index("file")
        st.bar_chart(chart_df)

        with st.expander("ℹ️ How to read this"):
            st.markdown("""
            - **Hybrid score** = α × Neural + (1-α) × BM25  (normalized)
            - **Neural** = CodeBERT + GCN fusion model output
            - **BM25** = classical keyword-matching baseline
            - α = 0.4 (best from our test-set sweep, MRR=0.607)
            """)

st.markdown("---")
st.caption(
    "Model: CodeBERT (12-layer) + 3-layer GCN + Multi-head cross-attention. "
    "Trained on 109 multi-domain bugs (15 GitHub repos)."
)
