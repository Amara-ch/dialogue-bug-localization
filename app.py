"""
Streamlit demo: Dialogue-Driven Bug Localization (cloud-friendly).

Live BM25 inference + pre-computed neural/hybrid research results.
Run: streamlit run app.py
"""
import json
import pickle
import re
from pathlib import Path

import pandas as pd
import streamlit as st
from rank_bm25 import BM25Okapi

# ---------- Paths ----------
ROOT = Path(__file__).parent
DATASET_PATH = ROOT / "data" / "processed" / "dataset.json"
GRAPH_PATH = ROOT / "data" / "graphs" / "code_graph.pkl"
TEST_RESULTS = ROOT / "checkpoints" / "test_results_v2.json"
BASELINES = ROOT / "checkpoints" / "baseline_comparison.json"
ENSEMBLE = ROOT / "checkpoints" / "ensemble_results.json"
HISTORY = ROOT / "checkpoints" / "history_v2.json"

# ---------- Page ----------
st.set_page_config(
    page_title="Dialogue Bug Localizer",
    page_icon="🔍",
    layout="wide",
)

st.title("🔍 Dialogue-Driven Bug Localization")
st.caption("Hybrid CodeBERT + Graph Neural Network + BM25 — final-year research demo")


# ---------- Data loading ----------
@st.cache_resource(show_spinner="Loading dataset and code graph...")
def load_assets():
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        dataset = json.load(f)
    with open(GRAPH_PATH, "rb") as f:
        graph = pickle.load(f)

    # Build per-repo file list from graph nodes
    repo_files = {}
    for node in graph.nodes:
        # node format: "owner/repo::path/to/file.py"
        if "::" in node:
            repo, path = node.split("::", 1)
        else:
            parts = node.split("/", 2)
            repo = "/".join(parts[:2]) if len(parts) >= 2 else node
            path = parts[2] if len(parts) > 2 else node
        repo_files.setdefault(repo, set()).add(path)
    repo_files = {r: sorted(fs) for r, fs in repo_files.items()}
    return dataset, repo_files


@st.cache_data
def load_results():
    def _read(p):
        if p.exists():
            with open(p, "r", encoding="utf-8") as f:
                return json.load(f)
        return None
    return {
        "test": _read(TEST_RESULTS),
        "baselines": _read(BASELINES),
        "ensemble": _read(ENSEMBLE),
        "history": _read(HISTORY),
    }


def tokenize(text):
    return re.findall(r"[a-zA-Z_][a-zA-Z0-9_]*", text.lower())


def bm25_predict(dialogue, files, top_k=5):
    """BM25 on file path tokens — same baseline used in our paper."""
    corpus = [tokenize(f.replace("/", " ").replace("_", " ").replace(".", " ")) for f in files]
    bm25 = BM25Okapi(corpus)
    q_tokens = tokenize(dialogue)
    scores = bm25.get_scores(q_tokens)
    # Normalize 0..1
    if scores.max() > 0:
        scores = scores / scores.max()
    ranked = sorted(zip(files, scores), key=lambda x: -x[1])[:top_k]
    return [{"file": f, "score": float(s)} for f, s in ranked]


# ---------- Load ----------
dataset, repo_files = load_assets()
results = load_results()

# ---------- Sidebar ----------
with st.sidebar:
    st.header("⚙️ Settings")
    mode = st.radio("Mode", ["🔍 Live Demo", "📊 Research Results"])
    if mode == "🔍 Live Demo":
        repos = sorted(repo_files.keys())
        repo = st.selectbox("Repository", repos)
        top_k = st.slider("Top-K files", 3, 15, 5)
        with st.expander(f"📁 Files in {repo} ({len(repo_files[repo])})"):
            for f in repo_files[repo][:50]:
                st.text(f)
            if len(repo_files[repo]) > 50:
                st.caption(f"... and {len(repo_files[repo]) - 50} more")

# ---------- Modes ----------
if mode == "🔍 Live Demo":
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
        placeholder="e.g. The login button doesn't work when CSRF token expires...",
    )

    if st.button("🔍 Locate Bug Files", type="primary", use_container_width=True):
        if not dialogue.strip():
            st.warning("Please enter a bug description.")
        else:
            with st.spinner("Running BM25 inference..."):
                preds = bm25_predict(dialogue, repo_files[repo], top_k=top_k)

            st.success(f"Top {len(preds)} suspected files in **{repo}**")
            df = pd.DataFrame(preds)
            df.index = [f"#{i+1}" for i in range(len(df))]
            st.dataframe(
                df.rename(columns={"file": "File path", "score": "BM25 score"}).style.format(
                    {"BM25 score": "{:.3f}"}
                ).background_gradient(subset=["BM25 score"], cmap="Greens"),
                use_container_width=True,
            )

            st.bar_chart(df.set_index("file")["score"])

            with st.expander("ℹ️ Note about this demo"):
                st.markdown(
                    """
                    This live demo runs **BM25 only** (lightweight, free-tier friendly).
                    The full **CodeBERT + GCN + BM25 hybrid** model achieves higher MRR
                    (0.607 vs 0.582) — see **📊 Research Results** tab for full numbers.

                    To run the full hybrid locally:
                    ```
                    git clone https://github.com/Amara-ch/dialogue-bug-localization
                    pip install -r requirements.txt
                    python -m src.training.train_v2
                    streamlit run app.py
                    ```
                    """
                )

else:
    # ---------- Research results tab ----------
    st.subheader("📊 Research Results — Test Set (22 bugs)")

    rows = []
    if results["baselines"]:
        for name, m in results["baselines"].items():
            rows.append({"Method": name, **{k: m.get(k) for k in ("top1", "top3", "top5", "top10", "mrr")}})
    if results["test"]:
        m = results["test"]
        rows.append({"Method": "Ours (CodeBERT+GCN)", **{k: m.get(k) for k in ("top1", "top3", "top5", "top10", "mrr")}})
    if results["ensemble"]:
        # ensemble may have multiple alphas - pick best
        ens = results["ensemble"]
        if isinstance(ens, dict) and "best" in ens:
            m = ens["best"]
            rows.append({"Method": f"🏆 Hybrid (α={m.get('alpha', 0.4)})",
                         **{k: m.get(k) for k in ("top1", "top3", "top5", "top10", "mrr")}})
        elif isinstance(ens, list) and ens:
            m = max(ens, key=lambda x: x.get("mrr", 0))
            rows.append({"Method": f"🏆 Hybrid (α={m.get('alpha', 0.4)})",
                         **{k: m.get(k) for k in ("top1", "top3", "top5", "top10", "mrr")}})

    if rows:
        df = pd.DataFrame(rows).rename(columns={
            "top1": "Top-1", "top3": "Top-3", "top5": "Top-5",
            "top10": "Top-10", "mrr": "MRR",
        })
        st.dataframe(
            df.style.format({c: "{:.3f}" for c in df.columns if c != "Method"})
            .background_gradient(subset=["MRR"], cmap="Greens"),
            use_container_width=True,
        )
    else:
        st.info("Result files not found — make sure checkpoints/*.json are in the repo.")

    st.markdown("**🏆 Best result: Hybrid (α=0.4) — MRR 0.607, 100% Top-10 accuracy**")

    # Training history
    if results["history"]:
        st.subheader("📈 Training history")
        h = results["history"]
        if isinstance(h, dict):
            df_h = pd.DataFrame(h)
            st.line_chart(df_h)

    # Dataset stats
    st.subheader("📦 Dataset")
    n_total = sum(len(v) if isinstance(v, list) else 1 for v in dataset.values()) if isinstance(dataset, dict) else len(dataset)
    st.metric("Total bugs", n_total)
    st.metric("Repositories", len(repo_files))

st.markdown("---")
st.caption(
    "Model: CodeBERT (12-layer) + 3-layer GCN + Multi-head cross-attention. "
    "Trained on 109 multi-domain bugs (15 GitHub repos)."
)