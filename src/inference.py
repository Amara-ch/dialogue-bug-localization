"""
Inference helper for the Hybrid Bug Localizer.

Loads:
  - trained CodeBERT+GCN model (best_v2.pt)
  - code graph
  - BM25 over file paths
Combines neural + BM25 with alpha=0.4 (best from sweep).
"""
import re
import json
import pickle
from pathlib import Path, PurePosixPath
from functools import lru_cache

import yaml
import numpy as np
import torch
from transformers import AutoTokenizer
from rank_bm25 import BM25Okapi

from src.models.fusion_model import FusionModel
from src.training.dataset import build_graph_tensors

TOKEN_RE = re.compile(r"[a-zA-Z]+")
DEFAULT_ALPHA = 0.4


def _path_tokens(path):
    toks = []
    for seg in PurePosixPath(path).parts:
        toks += TOKEN_RE.findall(seg.lower())
    return toks


def _normalize(vec):
    v = np.array(vec, dtype=np.float32)
    finite = v[np.isfinite(v)]
    if finite.size == 0:
        return v
    mn, mx = finite.min(), finite.max()
    if mx - mn < 1e-9:
        out = np.zeros_like(v)
    else:
        out = (v - mn) / (mx - mn)
    out[~np.isfinite(v)] = 0.0
    return out


class BugLocalizer:
    def __init__(self, config_path="configs/config.yaml"):
        with open(config_path, "r", encoding="utf-8") as f:
            self.cfg = yaml.safe_load(f)

        # Load graph
        with open(self.cfg["paths"]["code_graph"], "rb") as f:
            g = pickle.load(f)
        self.node_to_id   = g["node_to_id"]
        self.id_to_node   = {v: k for k, v in self.node_to_id.items()}
        self.repo_of_node = g["repo_of_node"]
        self.num_nodes    = len(self.node_to_id)

        self.repo_to_node_ids = {}
        for node, nid in self.node_to_id.items():
            self.repo_to_node_ids.setdefault(self.repo_of_node[node], []).append(nid)

        # BM25 over all file paths
        self.paths_per_node = [self.id_to_node[i].split("::", 1)[1] for i in range(self.num_nodes)]
        self.node_tokens    = [_path_tokens(p) for p in self.paths_per_node]
        self.bm25 = BM25Okapi(self.node_tokens)

        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.cfg["model"]["bert_name"])
        self.max_len   = self.cfg["model"]["max_dialogue_tokens"]

        # Model
        self.model = FusionModel(
            bert_name        = self.cfg["model"]["bert_name"],
            node_feat_dim    = self.cfg["model"]["node_feat_dim"],
            gnn_hidden       = self.cfg["model"]["gnn_hidden_dim"],
            gnn_layers       = self.cfg["model"]["gnn_layers"],
            fusion_heads     = self.cfg["model"]["fusion_heads"],
            dropout          = self.cfg["model"]["dropout"],
            freeze_bert_layers = self.cfg["model"]["freeze_bert_layers"],
        )
        ckpt_path = Path(self.cfg["paths"]["checkpoints"]) / "best_v2.pt"
        ckpt = torch.load(ckpt_path, map_location="cpu")
        self.model.load_state_dict(ckpt["model"])
        self.model.eval()

        # Graph tensors
        self.node_feats, self.edge_index, self.edge_weight = build_graph_tensors(
            self.cfg["paths"]["code_graph"], feat_dim=self.cfg["model"]["node_feat_dim"]
        )

    def available_repos(self):
        return sorted(self.repo_to_node_ids.keys())

    def repo_files(self, repo):
        return sorted(self.id_to_node[nid].split("::", 1)[1]
                      for nid in self.repo_to_node_ids[repo])

    @torch.no_grad()
    def predict(self, dialogue_text, repo, top_k=10, alpha=DEFAULT_ALPHA):
        if repo not in self.repo_to_node_ids:
            raise ValueError(f"Unknown repo: {repo}")

        cand_ids = self.repo_to_node_ids[repo]
        cand_paths = [self.id_to_node[c].split("::", 1)[1] for c in cand_ids]

        # ---- Neural score ----
        cand_text = " ; ".join(cand_paths[:40])
        enc = self.tokenizer(
            dialogue_text,
            "candidates: " + cand_text,
            truncation=True, max_length=self.max_len,
            padding="max_length", return_tensors="pt",
        )
        candidate_mask = torch.zeros(1, self.num_nodes, dtype=torch.bool)
        for c in cand_ids: candidate_mask[0, c] = True

        scores = self.model(
            enc["input_ids"], enc["attention_mask"],
            self.node_feats, self.edge_index, self.edge_weight,
            candidate_mask,
        ).squeeze(0).cpu().numpy()

        neural = np.full(self.num_nodes, -np.inf, dtype=np.float32)
        for c in cand_ids:
            neural[c] = scores[c]

        # ---- BM25 score ----
        bm25_full = self.bm25.get_scores(TOKEN_RE.findall(dialogue_text.lower()))

        # ---- Hybrid ----
        neural_n = _normalize(neural)
        bm25_n   = _normalize(bm25_full)
        combined = alpha * neural_n + (1 - alpha) * bm25_n

        results = []
        for c in cand_ids:
            results.append({
                "file":   cand_paths[cand_ids.index(c)],
                "score":  float(combined[c]),
                "neural": float(neural_n[c]),
                "bm25":   float(bm25_n[c]),
            })
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]


@lru_cache(maxsize=1)
def get_localizer():
    return BugLocalizer()
