"""Sanity check: model builds & forward pass works."""
import torch
from src.models.fusion_model import FusionModel


def main():
    torch.manual_seed(0)
    B, L, N = 2, 512, 85
    node_feat_dim = 128

    model = FusionModel(node_feat_dim=node_feat_dim)
    model.eval()

    input_ids       = torch.randint(0, 1000, (B, L))
    attention_mask  = torch.ones(B, L, dtype=torch.long)
    node_features   = torch.randn(N, node_feat_dim)
    edge_index      = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long)
    edge_weight     = torch.tensor([1.0, 0.5, 0.7, 1.0])
    candidate_mask  = torch.zeros(B, N, dtype=torch.bool)
    candidate_mask[0, :10]   = True
    candidate_mask[1, 10:25] = True

    with torch.no_grad():
        scores = model(input_ids, attention_mask,
                       node_features, edge_index, edge_weight,
                       candidate_mask)

    print("Output shape:", scores.shape)
    print("Sample 0 top score:", scores[0].max().item())
    print("Non-candidate masked to -inf?", torch.isinf(scores[0, 50]).item())
    total = sum(p.numel() for p in model.parameters())
    train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params:     {total/1e6:.2f}M")
    print(f"Trainable params: {train/1e6:.2f}M")
    print("Model built and forward pass OK.")


if __name__ == "__main__":
    main()
