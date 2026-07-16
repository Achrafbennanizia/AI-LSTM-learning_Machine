"""M2 — Übungstyp-Klassifikator, MLP (5 Klassen)."""
import torch
import torch.nn as nn


class UebungstypClassifier(nn.Module):
    def __init__(self, input_dim=69, n_klassen=5, hidden_dims=(128, 72, 48), dropout=0.2):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, n_klassen))
        self.net = nn.Sequential(*layers)

    def forward(self, vorhersage, hidden):
        return self.net(torch.cat([vorhersage, hidden], dim=-1))
