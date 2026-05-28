"""
Baselines for bug localization (no training needed):

  1. Random       - random score per file
  2. TF-IDF       - cosine(dialogue, file_path_text)
  3. BM25         - Okapi BM25 with file path as document

All evaluated on the SAME test split for fair comparison.
"""

import re
import json
import pickle
import random
from pathlib import Path, PurePosixPath

import yaml
import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Okapi

from src.training.dataset import BugLocalizationDataset
from src.training.metrics import topk_accuracy


TOKEN_RE = re.compile(r"[a-zA-Z]+")


def path_to_text(path):
    """Turn 'core/dtypes/common.py' -> 'core dtypes common'."""
    toks = []
    for seg in PurePosixPath(path).parts:
        toks += TOKEN_RE.findall(seg.lower())
    return " ".join(toks)


def tokenize(text):
    return TOKEN_RE.findall(text.lower())


def aggregate(metrics_list, n):
    agg = {"top1": 0, "top3": 0, "top5": 0, "top10": 0, "mrr": 0.0}
    for m, b in metrics_list:
        for k in agg:
            agg[k] += m[k] * b
    for k in agg:
        agg[k] /= max(1, n)
    return agg


def eval_scorer(test_ds, score_fn):
    """score_fn(sample, num_nodes) -> torch.FloatTensor [N]."""
    metrics_list, n = [], 0
    for s in test_ds.samples:
        repo = s["repo"]
        scores = score_fn(s, test_ds.num_nodes)

        # Build candidate mask + labels (same as Dataset)
        cand_mask = torch.zeros(test_ds.num_nodes, dtype=torch.bool)
        cand_mask[test_ds.repo_to_node_ids[repo]] = True
        labels = torch.zeros(test_ds.num_nodes, dtype=torch.float32)
        for ff in s["fix_files"]:
            key = f"{repo}::{ff}"
            if key in test_ds.node_to_id:
                labels[test_ds.node_to_id[key]] = 1.0

        scores = scores.masked_fill(~cand_mask, float("-inf"))
        m = topk_accuracy(scores.unsqueeze(0), labels.unsqueeze(0),
                          cand_mask.unsqueeze(0))
        metrics_list.append((m, 1))
        n += 1
    return aggregate(metrics_list, n)


def main():
    random.seed(42)
    np.random.seed(42)

    with open("configs/config.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    test_ds = BugLocalizationDataset(
        cfg["paths"]["processed"], cfg["paths"]["code_graph"], "test"
    )
    N = test_ds.num_nodes
    id_to_node = {v: k for k, v in test_ds.node_to_id.items()}
    paths_per_node = [id_to_node[i].split("::", 1)[1] for i in range(N)]
    print(f"Test samples: {len(test_ds)}  Nodes: {N}")

    # Pre-build per-node text representations
    node_texts = [path_to_text(p) for p in paths_per_node]
    node_tokens = [tokenize(t) for t in node_texts]

    # ---- 1. Random baseline ----
    def random_scorer(sample, N):
        return torch.rand(N)
    rnd = eval_scorer(test_ds, random_scorer)

    # ---- 2. TF-IDF baseline ----
    # Fit vectorizer on (dialogue + all node texts) so vocab is shared
    all_docs = [s["dialogue_text"] for s in test_ds.samples] + node_texts
    tfidf = TfidfVectorizer(token_pattern=r"[a-zA-Z]+", lowercase=True,
                            min_df=1, max_df=0.95)
    tfidf.fit(all_docs)
    node_vecs = tfidf.transform(node_texts)  # [N, V]

    def tfidf_scorer(sample, N):
        q = tfidf.transform([sample["dialogue_text"]])
        sim = cosine_similarity(q, node_vecs).ravel()  # [N]
        return torch.tensor(sim, dtype=torch.float32)
    tfidf_res = eval_scorer(test_ds, tfidf_scorer)

    # ---- 3. BM25 baseline ----
    bm25 = BM25Okapi(node_tokens)
    def bm25_scorer(sample, N):
        q_tok = tokenize(sample["dialogue_text"])
        sim = bm25.get_scores(q_tok)  # [N]
        return torch.tensor(sim, dtype=torch.float32)
    bm25_res = eval_scorer(test_ds, bm25_scorer)

    # ---- Load ours from test_results.json ----
    ours = None
    out_path = Path(cfg["paths"]["checkpoints"]) / "test_results.json"
    if out_path.exists():
        with open(out_path, "r", encoding="utf-8") as f:
            ours = json.load(f)["aggregate"]

    # ---- Print + save ----
    print("\n===== BASELINE COMPARISON (test set) =====")
    header = f"{'Method':<25} {'Top-1':>7} {'Top-3':>7} {'Top-5':>7} {'Top-10':>7} {'MRR':>7}"
    print(header)
    print("-" * len(header))
    def row(name, r):
        return (f"{name:<25} {r['top1']:>7.3f} {r['top3']:>7.3f} "
                f"{r['top5']:>7.3f} {r['top10']:>7.3f} {r['mrr']:>7.3f}")
    print(row("Random",  rnd))
    print(row("TF-IDF",  tfidf_res))
    print(row("BM25",    bm25_res))
    if ours is not None:
        print(row("Ours (CodeBERT+GCN)", ours))

    results = {"random": rnd, "tfidf": tfidf_res, "bm25": bm25_res, "ours": ours}
    save = Path(cfg["paths"]["checkpoints"]) / "baseline_comparison.json"
    with open(save, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved -> {save}")


if __name__ == "__main__":
    main()
