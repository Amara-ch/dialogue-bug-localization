"""Fusion: Dialogue + Code-graph -> per-file scores."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .dialogue_encoder import DialogueEncoder
from .graph_encoder import GraphEncoder


class FusionModel(nn.Module):
    def __init__(self, bert_name="microsoft/codebert-base",
                 node_feat_dim=128, gnn_hidden=256, gnn_layers=3,
                 fusion_heads=4, dropout=0.2, freeze_bert_layers=8):
        super().__init__()
        self.dialogue_encoder = DialogueEncoder(bert_name, freeze_layers=freeze_bert_layers)
        self.graph_encoder    = GraphEncoder(node_feat_dim, gnn_hidden, gnn_layers, dropout)

        d_text  = self.dialogue_encoder.hidden_dim
        d_graph = self.graph_encoder.out_dim

        self.text_proj = nn.Sequential(
            nn.Linear(d_text, d_graph),
            nn.LayerNorm(d_graph),
            nn.Dropout(dropout),
        )
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_graph, num_heads=fusion_heads,
            dropout=dropout, batch_first=True,
        )
        self.attn_norm  = nn.LayerNorm(d_graph)
        self.score_proj = nn.Linear(d_graph, d_graph)

    def forward(self, input_ids, attention_mask,
                node_features, edge_index, edge_weight, candidate_mask):
        d = self.dialogue_encoder(input_ids, attention_mask)
        d = self.text_proj(d)
        F_nodes = self.graph_encoder(node_features, edge_index, edge_weight)

        q = d.unsqueeze(1)
        k = F_nodes.unsqueeze(0).expand(d.size(0), -1, -1)
        v = k
        key_padding_mask = ~candidate_mask
        attn_out, _ = self.cross_attn(q, k, v, key_padding_mask=key_padding_mask)
        d_ctx = self.attn_norm(d + attn_out.squeeze(1))
        d_ctx = self.score_proj(d_ctx)

        scores = d_ctx @ F_nodes.t()
        scores = scores.masked_fill(~candidate_mask, float("-inf"))
        return scores
