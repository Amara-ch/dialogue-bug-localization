"""Top-K accuracy and Mean Reciprocal Rank for bug localization."""
import torch


@torch.no_grad()
def topk_accuracy(scores, labels, candidate_mask, ks=(1, 3, 5, 10)):
    """
    scores         [B, N]   (-inf at non-candidates)
    labels         [B, N]   1 at fix files, 0 elsewhere
    candidate_mask [B, N]
    Returns dict {k: hit_rate}, plus 'mrr'.
    """
    B = scores.size(0)
    results = {f"top{k}": 0.0 for k in ks}
    mrr_sum = 0.0

    for i in range(B):
        s = scores[i]
        order = torch.argsort(s, descending=True)
        # Keep only candidates (others are -inf, will sort last)
        cand_order = [int(x) for x in order if candidate_mask[i, x]]
        gt = set((labels[i] > 0.5).nonzero(as_tuple=True)[0].tolist())
        if not gt:
            continue
        # Top-K
        for k in ks:
            topk = set(cand_order[:k])
            if gt & topk:
                results[f"top{k}"] += 1.0
        # MRR (rank of first relevant)
        for rank, nid in enumerate(cand_order, start=1):
            if nid in gt:
                mrr_sum += 1.0 / rank
                break

    for k in ks:
        results[f"top{k}"] /= max(1, B)
    results["mrr"] = mrr_sum / max(1, B)
    return results
