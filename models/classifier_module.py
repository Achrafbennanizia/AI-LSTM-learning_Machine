import torch, torch.nn as nn


class UebungstypClassifier(nn.Module):
    """Tiefere MLP-Kette (variabel über hidden_dims)."""

    def __init__(
        self,
        input_dim=69,
        n_klassen=5,
        hidden_dims=(128, 72, 48),
        dropout=0.2,
    ):
        super().__init__()
        dims = list(hidden_dims)
        if len(dims) == 0:
            raise ValueError("hidden_dims braucht mindestens einen Eintrag.")

        layers: list = []
        prev = input_dim
        for h in dims:
            layers.extend(
                [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            )
            prev = h
        layers.append(nn.Linear(prev, n_klassen))
        self.net = nn.Sequential(*layers)

    def forward(self, vorhersage, hidden):
        return self.net(torch.cat([vorhersage, hidden], dim=-1))
