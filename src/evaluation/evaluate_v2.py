"""Evaluate best_v2 on test split AND save per-sample scores for ensembling."""
import json
from pathlib import Path

import yaml
import torch
from torch.utils.data import DataLoader

from src.models.fusion_model import FusionModel
from src.training.dataset_v2 import BugLocalizationDatasetV2, collate_v2
from src.training.dataset    import build_graph_tensors
from src.training.metrics    import topk_accuracy


def main():
    with open("configs/config.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    test_ds = BugLocalizationDatasetV2(
        cfg["paths"]["processed"], cfg["paths"]["code_graph"], "test",
        cfg["model"]["bert_name"], cfg["model"]["max_dialogue_tokens"],
    )
    print(f"Test samples: {len(test_ds)}")
    loader = DataLoader(test_ds, batch_size=cfg["training"]["batch_size"],
                        shuffle=False, collate_fn=collate_v2)

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
    ckpt = torch.load(Path(cfg["paths"]["checkpoints"]) / "best_v2.pt", map_location="cpu")
    model.load_state_dict(ckpt["model"]); model.eval()
    print(f"Loaded best_v2 epoch {ckpt['epoch']}")

    agg = {"top1": 0, "top3": 0, "top5": 0, "top10": 0, "mrr": 0.0}
    saved_scores = {}   # sample_id -> {node_id: score} for candidate nodes only
    n = 0
    with torch.no_grad():
        for batch in loader:
            scores = model(batch["input_ids"], batch["attention_mask"],
                           node_feats, edge_index, edge_weight,
                           batch["candidate_mask"])
            m = topk_accuracy(scores, batch["labels"], batch["candidate_mask"])
            B = scores.size(0)
            for k, v in m.items():
                agg[k] += v * B
            n += B
            for i in range(B):
                sid = batch["ids"][i]
                cand_ids = batch["candidate_mask"][i].nonzero(as_tuple=True)[0].tolist()
                saved_scores[sid] = {int(c): float(scores[i, c].item()) for c in cand_ids}

    for k in agg: agg[k] /= max(1, n)
    print("\n===== TEST RESULTS (v2) =====")
    for k in ["top1", "top3", "top5", "top10", "mrr"]:
        print(f"  {k:6s} : {agg[k]:.4f}")

    out_dir = Path(cfg["paths"]["checkpoints"])
    with open(out_dir / "test_results_v2.json", "w", encoding="utf-8") as f:
        json.dump({"aggregate": agg, "n_test": n, "epoch": ckpt["epoch"]}, f, indent=2)
    with open(out_dir / "neural_scores_v2.json", "w", encoding="utf-8") as f:
        json.dump(saved_scores, f)
    print(f"\nSaved -> test_results_v2.json + neural_scores_v2.json")


if __name__ == "__main__":
    main()
