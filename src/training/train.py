"""Phase 6: Training loop (CPU-friendly)."""
import json
import time
from pathlib import Path

import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.models.fusion_model import FusionModel
from src.training.dataset import (
    BugLocalizationDataset, collate, build_graph_tensors
)
from src.training.metrics import topk_accuracy


def load_cfg():
    with open("configs/config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def masked_bce_loss(scores, labels, candidate_mask):
    """BCE only over candidate files (non-candidates are -inf -> skip)."""
    # Replace -inf with 0 logits (they will be masked out in loss anyway)
    logits = scores.masked_fill(~candidate_mask, 0.0)
    loss_per = F.binary_cross_entropy_with_logits(
        logits, labels, reduction="none"
    )
    loss_per = loss_per * candidate_mask.float()
    denom = candidate_mask.float().sum().clamp(min=1.0)
    return loss_per.sum() / denom


def run_epoch(model, loader, graph, optimizer=None):
    is_train = optimizer is not None
    model.train(is_train)

    node_feats, edge_index, edge_weight = graph
    total_loss = 0.0
    n_batches  = 0
    agg_metrics = {"top1": 0, "top3": 0, "top5": 0, "top10": 0, "mrr": 0.0}

    for batch in loader:
        scores = model(
            batch["input_ids"], batch["attention_mask"],
            node_feats, edge_index, edge_weight,
            batch["candidate_mask"],
        )
        loss = masked_bce_loss(scores, batch["labels"], batch["candidate_mask"])

        if is_train:
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        total_loss += loss.item()
        n_batches  += 1

        # Metrics
        m = topk_accuracy(scores.detach(), batch["labels"], batch["candidate_mask"])
        for k, v in m.items():
            agg_metrics[k] += v

    n = max(1, n_batches)
    for k in agg_metrics:
        agg_metrics[k] /= n
    return total_loss / n, agg_metrics


def main():
    cfg = load_cfg()
    torch.manual_seed(cfg["training"]["seed"])

    processed = cfg["paths"]["processed"]
    graph_path = cfg["paths"]["code_graph"]
    ckpt_dir = Path(cfg["paths"]["checkpoints"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # 1. Datasets
    train_ds = BugLocalizationDataset(processed, graph_path, "train")
    val_ds   = BugLocalizationDataset(processed, graph_path, "val")
    print(f"Train: {len(train_ds)}  Val: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=cfg["training"]["batch_size"],
                              shuffle=True, collate_fn=collate)
    val_loader   = DataLoader(val_ds,   batch_size=cfg["training"]["batch_size"],
                              shuffle=False, collate_fn=collate)

    # 2. Graph tensors (shared across batch)
    node_feats, edge_index, edge_weight = build_graph_tensors(
        graph_path, feat_dim=cfg["model"]["node_feat_dim"]
    )
    print(f"Graph: nodes={node_feats.shape[0]} edges={edge_index.shape[1]} "
          f"feat_dim={node_feats.shape[1]}")
    graph = (node_feats, edge_index, edge_weight)

    # 3. Model
    model = FusionModel(
        bert_name        = cfg["model"]["bert_name"],
        node_feat_dim    = cfg["model"]["node_feat_dim"],
        gnn_hidden       = cfg["model"]["gnn_hidden_dim"],
        gnn_layers       = cfg["model"]["gnn_layers"],
        fusion_heads     = cfg["model"]["fusion_heads"],
        dropout          = cfg["model"]["dropout"],
        freeze_bert_layers = cfg["model"]["freeze_bert_layers"],
    )
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable params: {n_trainable/1e6:.2f}M")

    # 4. Optimizer
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg["training"]["lr"],
        weight_decay=cfg["training"]["weight_decay"],
    )

    # 5. Training loop
    best_val_top5 = -1.0
    history = []
    epochs = cfg["training"]["epochs"]

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        tr_loss, tr_m = run_epoch(model, train_loader, graph, optimizer)
        vl_loss, vl_m = run_epoch(model, val_loader,   graph, optimizer=None)
        dt = time.time() - t0

        print(f"Epoch {epoch:02d}/{epochs} | "
              f"train loss {tr_loss:.4f} top5 {tr_m['top5']:.3f} | "
              f"val loss {vl_loss:.4f} top1 {vl_m['top1']:.3f} "
              f"top5 {vl_m['top5']:.3f} mrr {vl_m['mrr']:.3f} | "
              f"{dt:.1f}s")

        history.append({
            "epoch": epoch, "train_loss": tr_loss, "val_loss": vl_loss,
            "train": tr_m, "val": vl_m, "time_s": dt,
        })

        if vl_m["top5"] > best_val_top5:
            best_val_top5 = vl_m["top5"]
            torch.save({"model": model.state_dict(), "epoch": epoch, "val": vl_m},
                       ckpt_dir / "best.pt")
            print(f"  -> Saved best (val top5 = {best_val_top5:.3f})")

    # Save history
    with open(ckpt_dir / "history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    print(f"\nDone. Best val top5: {best_val_top5:.3f}")
    print(f"Checkpoints + history: {ckpt_dir}")


if __name__ == "__main__":
    main()
