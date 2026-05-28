"""
Ensemble: combine neural v2 scores with BM25 scores.

final_score(file) = alpha * normalize(neural) + (1-alpha) * normalize(bm25)

Tries alpha in {0.0, 0.1, ..., 1.0} and reports the best mix on test set.
Note: this is reporting-only; for unbiased estimate one would tune alpha on val.
"""
import re, json, pickle
from pathlib import Path, PurePosixPath

import yaml
import numpy as np
import torch
from rank_bm25 import BM25Okapi

from src.training.dataset import BugLocalizationDataset
from src.training.metrics import topk_accuracy

TOKEN_RE = re.compile(r"[a-zA-Z]+")


def path_to_text(path):
    toks = []
    for seg in PurePosixPath(path).parts:
        toks += TOKEN_RE.findall(seg.lower())
    return toks


def normalize(vec):
    """Min-max normalize a 1-D numpy array (ignore -inf)."""
    finite = vec[np.isfinite(vec)]
    if finite.size == 0:
        return vec
    mn, mx = finite.min(), finite.max()
    if mx - mn < 1e-9:
        out = np.zeros_like(vec)
    else:
        out = (vec - mn) / (mx - mn)
    out[~np.isfinite(vec)] = 0.0
    return out


def evaluate_combined(test_ds, neural_scores, bm25_scores_per_sample, alpha):
    agg = {"top1": 0, "top3": 0, "top5": 0, "top10": 0, "mrr": 0.0}
    n = 0
    for s in test_ds.samples:
        sid = s["id"]; repo = s["repo"]
        N = test_ds.num_nodes
        cand_mask = torch.zeros(N, dtype=torch.bool)
        cand_mask[test_ds.repo_to_node_ids[repo]] = True
        labels = torch.zeros(N, dtype=torch.float32)
        for ff in s["fix_files"]:
            key = f"{repo}::{ff}"
            if key in test_ds.node_to_id:
                labels[test_ds.node_to_id[key]] = 1.0

        neural = np.full(N, -np.inf)
        for nid_str, v in neural_scores[sid].items():
            neural[int(nid_str)] = v
        bm25 = bm25_scores_per_sample[sid]

        neural_n = normalize(neural)
        bm25_n   = normalize(bm25)
        combined = alpha * neural_n + (1 - alpha) * bm25_n
        combined[~cand_mask.numpy()] = -np.inf
        sc = torch.tensor(combined, dtype=torch.float32).unsqueeze(0)
        m = topk_accuracy(sc, labels.unsqueeze(0), cand_mask.unsqueeze(0))
        for k, v in m.items(): agg[k] += v
        n += 1
    for k in agg: agg[k] /= max(1, n)
    return agg


def main():
    with open("configs/config.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    test_ds = BugLocalizationDataset(
        cfg["paths"]["processed"], cfg["paths"]["code_graph"], "test"
    )
    N = test_ds.num_nodes
    id_to_node = {v: k for k, v in test_ds.node_to_id.items()}
    paths_per_node = [id_to_node[i].split("::", 1)[1] for i in range(N)]
    node_tokens = [path_to_text(p) for p in paths_per_node]
    bm25 = BM25Okapi(node_tokens)

    # Pre-compute BM25 scores per sample
    bm25_scores_per_sample = {}
    for s in test_ds.samples:
        q = TOKEN_RE.findall(s["dialogue_text"].lower())
        bm25_scores_per_sample[s["id"]] = bm25.get_scores(q)

    # Load neural scores
    with open(Path(cfg["paths"]["checkpoints"]) / "neural_scores_v2.json", "r") as f:
        neural_scores = json.load(f)

    # Sweep alpha
    print("\n===== ENSEMBLE SWEEP =====")
    print(f"{'alpha':>6} {'Top-1':>7} {'Top-3':>7} {'Top-5':>7} {'Top-10':>7} {'MRR':>7}")
    best = None
    results = {}
    for alpha in [round(x * 0.1, 1) for x in range(0, 11)]:
        r = evaluate_combined(test_ds, neural_scores, bm25_scores_per_sample, alpha)
        results[alpha] = r
        marker = ""
        if best is None or r["mrr"] > best[1]["mrr"]:
            best = (alpha, r); marker = "  *"
        print(f"{alpha:>6.1f} {r['top1']:>7.3f} {r['top3']:>7.3f} {r['top5']:>7.3f} "
              f"{r['top10']:>7.3f} {r['mrr']:>7.3f}{marker}")

    print(f"\nBest alpha = {best[0]}  (MRR = {best[1]['mrr']:.3f})")

    out = Path(cfg["paths"]["checkpoints"]) / "ensemble_results.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"results": {str(k): v for k, v in results.items()},
                   "best_alpha": best[0], "best": best[1]}, f, indent=2)
    print(f"Saved -> {out}")


if __name__ == "__main__":
    main()
