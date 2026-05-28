"""Evaluate best checkpoint on the test split."""
import json
from pathlib import Path

import yaml
import torch
from torch.utils.data import DataLoader

from src.models.fusion_model import FusionModel
from src.training.dataset import (
    BugLocalizationDataset, collate, build_graph_tensors
)
from src.training.metrics import topk_accuracy


def main():
    with open("configs/config.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    test_ds = BugLocalizationDataset(
        cfg["paths"]["processed"], cfg["paths"]["code_graph"], "test"
    )
    print(f"Test samples: {len(test_ds)}")

    loader = DataLoader(test_ds, batch_size=cfg["training"]["batch_size"],
                        shuffle=False, collate_fn=collate)

    node_feats, edge_index, edge_weight = build_graph_tensors(
        cfg["paths"]["code_graph"], feat_dim=cfg["model"]["node_feat_dim"]
    )

    model = FusionModel(
        bert_name        = cfg["model"]["bert_name"],
        node_feat_dim    = cfg["model"]["node_feat_dim"],
        gnn_hidden       = cfg["model"]["gnn_hidden_dim"],
        gnn_layers       = cfg["model"]["gnn_layers"],
        fusion_heads     = cfg["model"]["fusion_heads"],
        dropout          = cfg["model"]["dropout"],
        freeze_bert_layers = cfg["model"]["freeze_bert_layers"],
    )
    ckpt = torch.load(Path(cfg["paths"]["checkpoints"]) / "best.pt",
                      map_location="cpu")
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"Loaded best checkpoint from epoch {ckpt['epoch']}")

    agg = {"top1": 0, "top3": 0, "top5": 0, "top10": 0, "mrr": 0.0}
    per_sample = []
    n = 0

    with torch.no_grad():
        for batch in loader:
            scores = model(
                batch["input_ids"], batch["attention_mask"],
                node_feats, edge_index, edge_weight,
                batch["candidate_mask"],
            )
            m = topk_accuracy(scores, batch["labels"], batch["candidate_mask"])
            B = scores.size(0)
            for k, v in m.items():
                agg[k] += v * B
            n += B

            # Per-sample for paper appendix
            for i in range(B):
                s = scores[i]
                order = [int(x) for x in torch.argsort(s, descending=True)
                         if batch["candidate_mask"][i, x]]
                gt = (batch["labels"][i] > 0.5).nonzero(as_tuple=True)[0].tolist()
                rank = next((r + 1 for r, nid in enumerate(order) if nid in gt), -1)
                per_sample.append({
                    "id": batch["ids"][i], "repo": batch["repos"][i],
                    "gt_rank": rank, "num_candidates": int(batch["candidate_mask"][i].sum()),
                })

    for k in agg:
        agg[k] /= max(1, n)
    print("\n===== TEST RESULTS =====")
    for k in ["top1", "top3", "top5", "top10", "mrr"]:
        print(f"  {k:6s} : {agg[k]:.4f}")

    out = Path(cfg["paths"]["checkpoints"]) / "test_results.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"aggregate": agg, "per_sample": per_sample,
                   "n_test": n, "epoch": ckpt["epoch"]}, f, indent=2)
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()
