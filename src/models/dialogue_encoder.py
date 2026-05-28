"""Dialogue Encoder: CodeBERT wrapper."""
import torch
import torch.nn as nn
from transformers import AutoModel


class DialogueEncoder(nn.Module):
    def __init__(self, bert_name="microsoft/codebert-base", freeze_layers=8):
        super().__init__()
        self.bert = AutoModel.from_pretrained(bert_name)
        self.hidden_dim = self.bert.config.hidden_size

        for p in self.bert.embeddings.parameters():
            p.requires_grad = False
        for layer in self.bert.encoder.layer[:freeze_layers]:
            for p in layer.parameters():
                p.requires_grad = False

    def forward(self, input_ids, attention_mask):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        return out.last_hidden_state[:, 0, :]
