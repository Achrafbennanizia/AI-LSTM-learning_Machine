"""Katalog-Matcher — wählt die passende Übung aus 15 Einträgen."""
import torch
import torch.nn as nn

KATALOG_DATEN = [
    ["em_einzeln_40bpm", 0, 0.10, 0.20, 0.80, 0.30, 0.27],
    ["em_einzeln_metronom", 0, 0.12, 0.50, 0.70, 0.30, 0.27],
    ["am_einzeln_40bpm", 0, 0.15, 0.20, 0.85, 0.35, 0.27],
    ["em_am_35bpm", 1, 0.20, 0.70, 0.60, 0.50, 0.23],
    ["em_am_50bpm", 1, 0.25, 0.80, 0.55, 0.50, 0.33],
    ["horse_no_name_50bpm", 3, 0.20, 0.60, 0.40, 0.40, 0.33],
    ["eleanor_rigby_50bpm", 3, 0.25, 0.75, 0.35, 0.40, 0.33],
    ["eleanor_rigby_60bpm", 3, 0.28, 0.80, 0.35, 0.45, 0.40],
    ["eleanor_rigby_70bpm", 3, 0.32, 0.85, 0.40, 0.50, 0.47],
    ["em_rhythmus_metronom", 2, 0.15, 0.90, 0.20, 0.30, 0.27],
    ["knockin_strophe_emcg", 3, 0.38, 0.65, 0.65, 0.55, 0.40],
    ["knockin_voll_60bpm", 3, 0.42, 0.60, 0.70, 0.55, 0.40],
    ["g_einzeln_technik", 4, 0.30, 0.10, 0.90, 0.70, 0.20],
    ["stand_by_me_60bpm", 3, 0.40, 0.70, 0.60, 0.55, 0.40],
    ["wish_you_were_here_intro", 3, 0.48, 0.55, 0.75, 0.70, 0.47],
]
KATALOG_IDS = [row[0] for row in KATALOG_DATEN]


def get_katalog_matrix():
    rows = [[row[1], row[2], row[3], row[4], row[5], row[6]] for row in KATALOG_DATEN]
    return torch.tensor(rows, dtype=torch.float32)


# [AI-assisted ~25%] tool=Cursor Composer | prompt=docs/ai_prompts/katalog_matcher.md
def _encoder(in_dim, hidden_dim, out_dim, blocks, dropout):
    layers = [nn.Linear(in_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout)]
    for _ in range(blocks - 1):
        layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout)]
    layers.append(nn.Linear(hidden_dim, out_dim))
    return nn.Sequential(*layers)


class KatalogMatcher(nn.Module):
    def __init__(
        self,
        query_dim=10,
        katalog_dim=6,
        embed_dim=128,
        hidden_dim=128,
        mlp_blocks=2,
        dropout=0.2,
    ):
        super().__init__()
        self.query_enc = _encoder(query_dim, hidden_dim, embed_dim, mlp_blocks, dropout)
        self.katalog_enc = _encoder(katalog_dim, hidden_dim, embed_dim, mlp_blocks, dropout)

    def forward(self, vorhersage, klasse_logits, katalog_matrix):
        q = self.query_enc(torch.cat([vorhersage, torch.softmax(klasse_logits, dim=-1)], dim=-1))
        k = self.katalog_enc(katalog_matrix)
        return torch.matmul(q, k.T)
# --- end AI-assisted (katalog_matcher) ---
