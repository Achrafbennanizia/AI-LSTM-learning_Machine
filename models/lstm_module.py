import torch, torch.nn as nn


class VerhaltensLSTM(nn.Module):
    """Mehrschicht-LSTM mit breiterem FC-Kopf für Fehlerratenvorhersage."""

    # input_dim MUSS FEATURE_COLS in train.py entsprechen (aktuell 17 Sessions-Features).
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
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )
        self.drop = nn.Dropout(dropout)

        # Lesekopf: entweder ein Linear (head_hidden=0, Kompatibilität) oder MLP
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
        out, (h_n, _) = self.lstm(x)
        last = self.drop(out[:, -1, :])
        vorhersage = self.sigmoid(self.fc_head(last))
        return vorhersage, h_n[-1]
