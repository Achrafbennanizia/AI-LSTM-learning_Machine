"""
GuitarAI — MLP-Architektur (kein sequenzielles Gedächtnis)
Verarbeitet nur die aktuelle Session-Feature-Vektor.
Baseline: zeigt was ohne zeitliches Gedächtnis möglich ist.
"""
import torch
import torch.nn as nn


class VerhaltensMLPHead(nn.Module):
    """Einfacher MLP — kein LSTM, keine Zeitreihe.
    Input: ein einzelner Feature-Vektor (n_features,)
    Output: Fehlerraten-Vorhersage + hidden-repr für Classifier
    """

    def __init__(self, input_dim=17, hidden_dims=(128, 64), out_dim=5, dropout=0.2):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        self.backbone = nn.Sequential(*layers)
        self.hidden_dim = prev           # für Classifier-Kompatibilität
        self.fc_out = nn.Linear(prev, out_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        x: (batch, n_features)  — kein seq_len für MLP
        Gibt (vorhersage, hidden) zurück — gleiche API wie LSTM/GRU
        """
        h = self.backbone(x)
        vorhersage = self.sigmoid(self.fc_out(h))
        return vorhersage, h             # h als "hidden" für M2-Kompatibilität
