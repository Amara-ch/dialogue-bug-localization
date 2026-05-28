"""PyTorch Dataset + collate for bug-localization training."""
import json
import pickle
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class BugLocalizationDataset(Dataset):
    def __init__(self, processed_path, graph_path, split):
        with open(processed_path, "r", encoding="utf-8") as f:
            ds = json.load(f)
        with open(graph_path, "rb") as f:
            g = pickle.load(f)

        self.node_to_id = g["node_to_id"]
        self.repo_of_node = g["repo_of_node"]
        self.num_nodes = len(self.node_to_id)

        # Per-repo candidate masks (which node IDs belong to that repo)
        self.repo_to_node_ids = {}
        for node, nid in self.node_to_id.items():
            self.repo_to_node_ids.setdefault(self.repo_of_node[node], []).append(nid)

        # Filter samples to those in this split
        wanted = set(ds["splits"][split])
        self.samples = [s for s in ds["samples"] if s["id"] in wanted]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        repo = s["repo"]

        # Candidate mask: True for files in this bug's repo
        candidate_mask = torch.zeros(self.num_nodes, dtype=torch.bool)
        candidate_mask[self.repo_to_node_ids[repo]] = True

        # Positive label: 1 for fix_files, 0 otherwise (multi-label)
        labels = torch.zeros(self.num_nodes, dtype=torch.float32)
        for ff in s["fix_files"]:
            key = f"{repo}::{ff}"
            if key in self.node_to_id:
                labels[self.node_to_id[key]] = 1.0

        return {
            "id":             s["id"],
            "repo":           repo,
            "input_ids":      torch.tensor(s["input_ids"],      dtype=torch.long),
            "attention_mask": torch.tensor(s["attention_mask"], dtype=torch.long),
            "candidate_mask": candidate_mask,
            "labels":         labels,
        }


def collate(batch):
    return {
        "ids":            [b["id"] for b in batch],
        "repos":          [b["repo"] for b in batch],
        "input_ids":      torch.stack([b["input_ids"]      for b in batch]),
        "attention_mask": torch.stack([b["attention_mask"] for b in batch]),
        "candidate_mask": torch.stack([b["candidate_mask"] for b in batch]),
        "labels":         torch.stack([b["labels"]         for b in batch]),
    }


def build_graph_tensors(graph_path, feat_dim=128):
    """Returns node_features, edge_index, edge_weight as tensors."""
    from src.training.node_features import file_features
    with open(graph_path, "rb") as f:
        g = pickle.load(f)
    G = g["graph"]
    id_to_node = g["id_to_node"]
    N = len(id_to_node)

    # Node features from path
    feats = np.zeros((N, feat_dim), dtype=np.float32)
    for nid in range(N):
        node = id_to_node[nid]
        path = node.split("::", 1)[1]
        feats[nid] = file_features(path, feat_dim)

    # Edges
    node_to_id = g["node_to_id"]
    src, dst, w = [], [], []
    for u, v, data in G.edges(data=True):
        ui, vi = node_to_id[u], node_to_id[v]
        weight = float(data.get("weight", 1.0))
        # undirected -> add both directions
        src += [ui, vi]
        dst += [vi, ui]
        w   += [weight, weight]

    edge_index  = torch.tensor([src, dst], dtype=torch.long)
    edge_weight = torch.tensor(w, dtype=torch.float32)
    node_feats  = torch.tensor(feats, dtype=torch.float32)
    return node_feats, edge_index, edge_weight
