# dialogue-bug-localization
Dialogue-Based Bug Localization using BERT + GNN (Term Project)
<div align="center">

# 🧬 Dialogue-Driven Bug Localization

### *Hybrid CodeBERT + Graph Neural Network + BM25 for finding bugs from human conversations*

[![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.36-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Transformers](https://img.shields.io/badge/🤗_Transformers-4.40-yellow)](https://huggingface.co/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Live Demo](https://img.shields.io/badge/🚀_Live-Demo-6366f1)](https://dialogue-bug-localization.streamlit.app/)

**A hybrid AI system that reads natural-language bug reports and pinpoints the files most likely responsible — fusing transformer semantics, graph topology, and classical IR into one verdict.**

[🚀 Live Demo](https://dialogue-bug-localization.streamlit.app/) · [📖 Methodology](#-methodology) · [📊 Results](#-results) · [👤 About](#-about-the-researcher)

</div>

---

## 📋 Table of Contents

- [🎯 What is this?](#-what-is-this)
- [✨ Key Highlights](#-key-highlights)
- [🏗️ Architecture](#️-architecture)
- [📦 Installation](#-installation)
- [🚀 Quick Start](#-quick-start)
- [🔬 Methodology](#-methodology)
- [📊 Results](#-results)
- [📁 Project Structure](#-project-structure)
- [🧪 Reproducing the Results](#-reproducing-the-results)
- [📚 References](#-references)
- [👤 About the Researcher](#-about-the-researcher)
- [📜 License](#-license)

---

## 🎯 What is this?

Modern software repositories contain **thousands of files**. When a user reports a bug — usually inside a long, messy conversation full of stack traces, screenshots, and back-and-forth — a developer must manually trace through the codebase to locate the offending file. On large open-source projects this can take **2–4 hours per bug**.

This project introduces a **hybrid AI system** that automates that search: given a bug dialogue (issue text, conversation, stack trace), it returns the **top-K files** most likely to contain the bug — in under **2 seconds**, with **100% Top-10 accuracy** on unseen test bugs.

> **TL;DR** — Paste a bug report. Get a ranked list of suspect files. Engineer goes to fix the bug instead of hunting for it.

---

## ✨ Key Highlights

| Metric | Value | Significance |
|--------|-------|--------------|
| 🎯 **Mean Reciprocal Rank** | **0.607** | +4.3% over strongest baseline |
| 📈 **Top-10 Accuracy** | **100%** | Correct file always in top 10 |
| ⚡ **Inference time** | **< 2 sec** | >1000× faster than manual search |
| 📦 **Repositories** | **15** | Multi-domain generalization |
| 🐛 **Real bugs** | **109** | From actual GitHub issues |
| 💻 **Hardware** | **GTX 1650 (4GB)** | Trains on commodity GPU |

---

## 🏗️ Architecture

```
                ┌──────────────────────────┐
                │   Bug Dialogue / Issue   │
                └────────────┬─────────────┘
                             │
              ┌──────────────┼──────────────┐
              ▼              ▼              ▼
      ┌──────────────┐ ┌────────────┐ ┌────────────┐
      │   CodeBERT   │ │ Code Graph │ │    BM25    │
      │  Encoder     │ │  (NetworkX)│ │  Lexical   │
      │  (768-dim)   │ │            │ │  Matcher   │
      └──────┬───────┘ └─────┬──────┘ └─────┬──────┘
             │               │              │
             └───────┬───────┘              │
                     ▼                      │
            ┌─────────────────┐             │
            │  3-Layer GCN    │             │
            │  + Cross-Attn   │             │
            └────────┬────────┘             │
                     │                      │
                     └────────┬─────────────┘
                              ▼
                  ┌────────────────────────┐
                  │  Hybrid Fusion         │
                  │  α·Neural + (1-α)·BM25 │
                  │  α = 0.4               │
                  └────────────┬───────────┘
                               ▼
                   ┌─────────────────────┐
                   │  Top-K Ranked Files │
                   └─────────────────────┘
```

### Three signals, one verdict

| Channel | What it captures | Why it matters |
|---------|------------------|----------------|
| 🧠 **CodeBERT** | Semantic intent in 768-dim vectors | Understands paraphrases, synonyms, intent |
| 🕸️ **GCN** | File relationships via imports/calls | "Guilt by association" propagation |
| 🔍 **BM25** | Lexical keyword overlap | Catches obvious matches, interpretable |

---

## 📦 Installation

### Prerequisites

- Python **3.11** (3.10–3.12 also work)
- 8 GB RAM minimum (16 GB recommended)
- CUDA-capable GPU (optional, for training)

### Setup

```bash
# 1. Clone the repository
git clone https://github.com/Amara-ch/dialogue-bug-localization.git
cd dialogue-bug-localization

# 2. Create a virtual environment (recommended)
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

### Lightweight (Streamlit Cloud / demo only)

```bash
pip install streamlit pandas numpy networkx rank_bm25 matplotlib
```

### Full (training + inference with neural model)

```bash
pip install streamlit pandas numpy networkx rank_bm25 matplotlib \
            torch transformers torch-geometric scikit-learn pyyaml
```

---

## 🚀 Quick Start

### Run the web app

```bash
streamlit run app.py
```

Open your browser at `http://localhost:8501` and:

1. Click **🚀 Try the Model**
2. Pick a real bug from the test set (or paste your own)
3. Click **🔍 Locate Bug Files**
4. See ranked predictions with confidence scores

### Programmatic inference

```python
from src.inference import get_localizer

loc = get_localizer()

dialogue = """
When merging two DataFrames using pd.merge with how='outer', 
the resulting DataFrame loses the original index order and 
raises a KeyError on duplicate column names.
"""

results = loc.predict(dialogue, repo="pandas-dev/pandas", top_k=5, alpha=0.4)
for r in results:
    print(f"{r['score']:.3f}  {r['file']}")
```

**Expected output:**
```
0.842  pandas/core/reshape/merge.py
0.673  pandas/core/frame.py
0.512  pandas/core/indexes/base.py
0.448  pandas/core/reshape/concat.py
0.391  pandas/core/internals/managers.py
```

---

## 🔬 Methodology

### Six-stage pipeline

| Stage | Channel | Description |
|-------|---------|-------------|
| **1. Input preprocessing** | text | Strip Markdown/HTML, truncate to 512 BERT tokens |
| **2. CodeBERT encoder** | semantic | 12-layer transformer (110M params), top 4 layers fine-tuned |
| **3. Code graph construction** | structural | AST parsing → nodes (files), edges (imports, calls, inheritance) |
| **4. Graph Neural Network** | structural | 3-layer GCN + multi-head cross-attention, hidden dim 256 |
| **5. BM25 lexical scoring** | lexical | Okapi BM25 on file path tokens |
| **6. Hybrid fusion** | fusion | `α · neural + (1-α) · BM25`, α = 0.4 |

### The fusion equation

```
score(file_i | dialogue_d) = α · σ(MLP(h_d ⊕ h_i)) + (1 − α) · BM25(d, path_i)

where:
    h_d  = CodeBERT(dialogue) ∈ ℝ^768       # semantic embedding
    h_i  = GCN(graph, file_i) ∈ ℝ^256       # structural embedding
    σ    = sigmoid activation
    ⊕    = concatenation
    α    = 0.4   (validated on held-out split)
```

### Hyperparameters

| Parameter | Value |
|-----------|-------|
| Optimizer | AdamW (β₁=0.9, β₂=0.999, weight decay=0.01) |
| Learning rate | 2e-5 with linear warmup |
| Batch size | 8 (gradient accumulation = 2) |
| Epochs | 30 (early stopping on val MRR) |
| Loss | Pairwise margin ranking (margin = 0.3) |
| Hardware | NVIDIA GTX 1650 (4 GB) |
| Training time | ~6 hours |
| Random seed | 42 |

---

## 📊 Results

### Test-set performance (22 unseen bugs across 15 repositories)

| Method | Top-1 | Top-3 | Top-5 | Top-10 | MRR |
|--------|:-----:|:-----:|:-----:|:------:|:---:|
| TF-IDF | 0.227 | 0.500 | 0.636 | 0.864 | 0.391 |
| BM25 | 0.318 | 0.591 | 0.727 | 0.955 | 0.486 |
| BugLocator | 0.273 | 0.545 | 0.682 | 0.909 | 0.452 |
| CodeBERT (ours, neural only) | 0.273 | 0.545 | 0.682 | 0.864 | 0.452 |
| CodeBERT + GCN (ours) | 0.318 | 0.591 | 0.773 | 0.955 | 0.520 |
| **🏆 Hybrid (α = 0.4)** | **0.409** | **0.682** | **0.864** | **1.000** | **0.607** |

### Dataset composition

| Split | Bugs | Repositories |
|-------|------|--------------|
| Train | 81 | 15 |
| Validation | 5 | 5 |
| Test | 22 | 15 |
| **Total** | **109** | **15** |

### Repositories covered

`pandas-dev/pandas` · `psf/black` · `tornadoweb/tornado` · `scrapy/scrapy` · `ytdl-org/youtube-dl` · `matplotlib/matplotlib` · `ansible/ansible` · `tiangolo/fastapi` · `keras-team/keras` · `spotify/luigi` · `httpie/httpie` · `sanic-org/sanic` · `tqdm/tqdm` · `nvbn/thefuck` · `cool-RR/PySnooper`

---

## 📁 Project Structure

```
dialogue-bug-localization/
│
├── app.py                          # Streamlit web app (portfolio + demo)
├── requirements.txt                # Python dependencies
├── README.md                       # This file
├── runtime.txt                     # Python version pin (Streamlit Cloud)
│
├── configs/
│   └── config.yaml                 # Model & training hyperparameters
│
├── data/
│   ├── raw/                        # Raw GitHub issues + repo dumps
│   ├── processed/
│   │   └── dataset.json            # Tokenized samples + train/val/test splits
│   └── graphs/
│       └── code_graph.pkl          # Combined code graph (all repos)
│
├── src/
│   ├── data/
│   │   ├── collect_issues.py       # GitHub API scraper
│   │   └── preprocess.py           # Tokenization & cleaning
│   ├── graph/
│   │   └── build_graph.py          # AST → graph
│   ├── models/
│   │   ├── fusion_model.py         # Main hybrid model
│   │   └── gcn.py                  # GCN component
│   ├── training/
│   │   ├── train_v2.py             # Training loop
│   │   └── dataset.py              # PyTorch Dataset
│   ├── evaluation/
│   │   ├── evaluate.py             # Test-set metrics
│   │   └── baselines.py            # BM25, TF-IDF, BugLocator
│   └── inference.py                # Prediction API
│
├── checkpoints/                    # Trained models & results
│   ├── best_v2.pt                  # Best checkpoint (250 MB, gitignored)
│   ├── test_results_v2.json        # Test metrics
│   ├── baseline_comparison.json    # Baseline metrics
│   ├── ensemble_results.json       # α-sweep results
│   └── history_v2.json             # Training curves
│
└── notebooks/                      # Analysis notebooks (exploration)
```

---

## 🧪 Reproducing the Results

### 1. Download the data

```bash
python -m src.data.collect_issues   # Scrape GitHub issues
python -m src.data.preprocess        # Tokenize and split
python -m src.graph.build_graph      # Build code graph
```

### 2. Train the model

```bash
python -m src.training.train_v2 --config configs/config.yaml
```

### 3. Evaluate

```bash
python -m src.evaluation.evaluate --checkpoint checkpoints/best_v2.pt
python -m src.evaluation.baselines      # Run all baselines
```

### 4. Sweep α for hybrid fusion

```bash
python -m src.evaluation.alpha_sweep    # Tries α ∈ [0.0, 0.1, ..., 1.0]
```

---

## 📚 References

### Foundational papers

1. **CodeBERT** — Feng et al., EMNLP 2020 — [arxiv.org/abs/2002.08155](https://arxiv.org/abs/2002.08155)
2. **GCN** — Kipf & Welling, ICLR 2017 — [arxiv.org/abs/1609.02907](https://arxiv.org/abs/1609.02907)
3. **Attention Is All You Need** — Vaswani et al., NeurIPS 2017 — [arxiv.org/abs/1706.03762](https://arxiv.org/abs/1706.03762)
4. **BM25** — Robertson & Zaragoza, 2009
5. **BugLocator** — Zhou, Zhang & Lo, ICSE 2012
6. **SWE-bench** — Jimenez et al., ICLR 2024 — [arxiv.org/abs/2310.06770](https://arxiv.org/abs/2310.06770)

### Datasets

- **BugsInPy** — [github.com/soarsmu/BugsInPy](https://github.com/soarsmu/BugsInPy)
- **Defects4J** — [github.com/rjust/defects4j](https://github.com/rjust/defects4j)

### Tools

- 🤗 **Transformers** — [huggingface.co/docs/transformers](https://huggingface.co/docs/transformers)
- 🔥 **PyTorch Geometric** — [pytorch-geometric.readthedocs.io](https://pytorch-geometric.readthedocs.io/)
- 📊 **rank_bm25** — [github.com/dorianbrown/rank_bm25](https://github.com/dorianbrown/rank_bm25)

---

## 👤 About the Researcher

<div align="center">

### Amara Tariq
**AI / ML Researcher**
📍 Lahore, Pakistan 🇵🇰

</div>

I'm a Computer Science student and AI researcher passionate about applying deep learning to real-world software engineering problems. My research interests sit at the intersection of **NLP, graph learning, and developer productivity** — building AI systems that augment, rather than replace, human engineers.

**Open to:** research internships · ML engineering roles · open-source collaborations · PhD opportunities in AI4SE / NLP / Graph Learning.

### 📬 Get in touch

| Channel | Link |
|---------|------|
| ✉️ **Email** | [amaratariq9494@gmail.com](mailto:amaratariq9494@gmail.com) |
| 💼 **LinkedIn** | [linkedin.com/in/amara-tariq-2762ab331](https://www.linkedin.com/in/amara-tariq-2762ab331) |
| ⌥ **GitHub** | [github.com/Amara-ch](https://github.com/Amara-ch) |
| 🌐 **Live Demo** | [dialogue-bug-localization.streamlit.app](https://dialogue-bug-localization.streamlit.app/) |

---

## 🤝 Contributing

Contributions are welcome! Please feel free to:

- 🐛 [Open an issue](https://github.com/Amara-ch/dialogue-bug-localization/issues) for bugs or feature requests
- 🔀 Submit pull requests for improvements
- ⭐ Star the repo if you find it useful

---

## 📜 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgements

- Microsoft Research for releasing **CodeBERT**
- The **Hugging Face** team for the Transformers library
- The **PyTorch Geometric** team for excellent GNN tooling
- All the open-source maintainers whose repositories made this dataset possible

---

<div align="center">

**Built with ❤️ in Lahore · © 2026 Amara Tariq**

⭐ If this work helped you, please consider starring the repo!

</div>
