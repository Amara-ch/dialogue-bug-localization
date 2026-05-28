"""
Augmented dataset: dialogue [SEP] candidate file list, tokenized on the fly.

The model now SEES which files it must rank, in the same input as the
dialogue, allowing BERT self-attention to align bug-symptom tokens with
file-name tokens directly.
"""
import json
import pickle
from pathlib import PurePosixPath

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer


def _candidates_text(candidate_paths, max_files=40):
    """Convert candidate file list to a single text string."""
    parts = []
    for p in candidate_paths[:max_files]:
        # use both full path and stem to give the tokenizer two signals
        parts.append(p)
    return " ; ".join(parts)


class BugLocalizationDatasetV2(Dataset):
    def __init__(self, processed_path, graph_path, split,
                 tokenizer_name="microsoft/codebert-base",
                 max_len=512):
        with open(processed_path, "r", encoding="utf-8") as f:
            ds = json.load(f)
        with open(graph_path, "rb") as f:
            g = pickle.load(f)

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_len = max_len

        self.node_to_id   = g["node_to_id"]
        self.repo_of_node = g["repo_of_node"]
        self.num_nodes    = len(self.node_to_id)

        self.repo_to_node_ids = {}
        for node, nid in self.node_to_id.items():
            self.repo_to_node_ids.setdefault(self.repo_of_node[node], []).append(nid)

        wanted = set(ds["splits"][split])
        self.samples = [s for s in ds["samples"] if s["id"] in wanted]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        repo = s["repo"]

        dialogue = s["dialogue_text"]
        cand_text = _candidates_text(s["candidate_files"])
        enc = self.tokenizer(
            dialogue,
            "candidates: " + cand_text,
            truncation=True, max_length=self.max_len,
            padding="max_length", return_tensors="pt",
        )

        candidate_mask = torch.zeros(self.num_nodes, dtype=torch.bool)
        candidate_mask[self.repo_to_node_ids[repo]] = True

        labels = torch.zeros(self.num_nodes, dtype=torch.float32)
        for ff in s["fix_files"]:
            key = f"{repo}::{ff}"
            if key in self.node_to_id:
                labels[self.node_to_id[key]] = 1.0

        return {
            "id":             s["id"],
            "repo":           repo,
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "candidate_mask": candidate_mask,
            "labels":         labels,
            "dialogue_text":  dialogue,
        }


def collate_v2(batch):
    return {
        "ids":            [b["id"] for b in batch],
        "repos":          [b["repo"] for b in batch],
        "input_ids":      torch.stack([b["input_ids"]      for b in batch]),
        "attention_mask": torch.stack([b["attention_mask"] for b in batch]),
        "candidate_mask": torch.stack([b["candidate_mask"] for b in batch]),
        "labels":         torch.stack([b["labels"]         for b in batch]),
        "dialogue_text":  [b["dialogue_text"] for b in batch],
    }
