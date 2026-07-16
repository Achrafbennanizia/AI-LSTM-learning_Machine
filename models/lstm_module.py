"""LSTM — liest die letzten Sessions und sagt Fehlerraten voraus."""
import torch.nn as nn


class VerhaltensLSTM(nn.Module):
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
        self.lstm = nn.LSTM(input_dim, hidden_dim, n_layers, batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.fc_head = nn.Sequential(
            nn.Linear(hidden_dim, head_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden, out_dim),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out, (h_n, _) = self.lstm(x)
        last = self.drop(out[:, -1, :])
        return self.sigmoid(self.fc_head(last)), h_n[-1]
