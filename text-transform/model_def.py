import torch
import torch.nn as nn
import math

DROPOUT = 0.2
FF_SIZE = 8
max_length = 1000  # أو غيرها حسب اللي انت استخدمته في التدريب

class CustomTransformerModel(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_encoder_layers, num_classes):
        super(CustomTransformerModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_embedding = nn.Embedding(max_length, d_model)
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, batch_first=True, dropout=DROPOUT, dim_feedforward=FF_SIZE)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_encoder_layers)
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, src):
        src_positions = torch.arange(0, src.size(1), device=src.device).unsqueeze(0).expand(src.size(0), -1)
        src = self.embedding(src) + self.positional_embedding(src_positions)
        output = self.transformer_encoder(src)
        output = output.mean(dim=1)  # Global average pooling
        output = self.fc(output)
        return output
