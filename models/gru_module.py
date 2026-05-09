"""
GuitarAI — GRU-Architektur
Gated Recurrent Unit: leichter als LSTM (keine Cell-State),
oft vergleichbare Performance bei schnellerer Konvergenz.
Gleiche API wie VerhaltensLSTM für direkten Vergleich.
"""
import torch
import torch.nn as nn


class VerhaltensGRU(nn.Module):
    """GRU-Variante des Verhaltensmodells.
    Unterschied zu LSTM:
      - Nur 2 Gates (reset + update) statt 3 (input, forget, output)
      - Kein separater Cell-State → weniger Parameter
      - Oft schneller konvergent, gut für kurze Sequenzen
    """

    def __init__(
        self,
        input_dim=17,
        hidden_dim=64,
        n_layers=1,
        dropout=0.2,
        head_hidden=64,
        out_dim=5,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        self.gru = nn.GRU(
            input_dim,
            hidden_dim,
            n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )
        self.drop = nn.Dropout(dropout)

        if head_hidden and head_hidden > 0:
            self.fc_head = nn.Sequential(
                nn.Linear(hidden_dim, head_hidden),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(head_hidden, out_dim),
            )
        else:
            self.fc_head = nn.Linear(hidden_dim, out_dim)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        x: (batch, seq_len, input_dim)
        Gibt (vorhersage, hidden) zurück — gleiche API wie VerhaltensLSTM
        """
        out, h_n = self.gru(x)           # h_n: (n_layers, batch, hidden)
        last = self.drop(out[:, -1, :])
        vorhersage = self.sigmoid(self.fc_head(last))
        return vorhersage, h_n[-1]       # h_n[-1]: letzter Layer
