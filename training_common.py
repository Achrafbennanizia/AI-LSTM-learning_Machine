"""Gemeinsame Dataset-, Modell- und Eval-Hilfen für research/."""
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset

from models.classifier_module import UebungstypClassifier
from models.rnn_module import VerhaltensRNN
from models.gru_module import VerhaltensGRU
from models.katalog_matcher import KATALOG_IDS, KatalogMatcher
from models.lstm_module import VerhaltensLSTM

FEATURE_COLS = [
    "e_griff", "e_druck", "e_timing", "e_technik", "e_muting",
    "v_lern", "k_konsistenz", "d_limit", "p_plateau", "r_repeat",
    "n_sessions", "t_session", "s_level",
    "pause_norm", "akkord_fokus", "wochentag_norm", "tageszeit_norm",
]
KAT_INDEX = {kid: i for i, kid in enumerate(KATALOG_IDS)}

MSE_WEIGHT = 0.5
KATALOG_WEIGHT = 0.8
TB_SAMPLES = 10_000_000

# [AI-assisted ~25%] tool=Cursor Composer | prompt=docs/ai_prompts/training_common.md
class SessionDataset(Dataset):
    def __init__(self, df, seq_len=5):
        self.samples = []
        for _, group in df.groupby("nutzer_id"):
            group = group.sort_values("session")
            feats = group[FEATURE_COLS].values.astype(np.float32)
            labels = group["uebungstyp_label"].values.astype(np.int64)
            kats = group["empfehlung_id"].values
            for i in range(seq_len, len(group)):
                self.samples.append((
                    feats[i - seq_len:i],
                    feats[i, :5],
                    labels[i],
                    KAT_INDEX[kats[i]],
                ))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x, yf, yk, ykat = self.samples[idx]
        return torch.tensor(x), torch.tensor(yf), torch.tensor(yk), torch.tensor(ykat)
# --- end AI-assisted (SessionDataset) ---


def split_pretrain_inc_test(df, seed=42):
    """60/20/20 nach Nutzer — für research/."""
    users = df["nutzer_id"].unique().tolist()
    rng = np.random.RandomState(seed)
    rng.shuffle(users)
    n = len(users)
    n_pre = int(n * 0.6)
    n_test = int(n * 0.2)
    pre = users[:n_pre] 
    inc = users[n_pre:n_pre + (n - n_pre - n_test)]
    test = users[n_pre + (n - n_pre - n_test):]
    return (
        df[df["nutzer_id"].isin(pre)],
        df[df["nutzer_id"].isin(inc)],
        df[df["nutzer_id"].isin(test)],
    )


def build_models(arch, device, hidden=64):
    if arch == "rnn":
        m1 = VerhaltensRNN(
            input_dim=len(FEATURE_COLS), hidden_dim=hidden, n_layers=1, head_hidden=hidden,
        ).to(device)
    elif arch == "lstm":
        m1 = VerhaltensLSTM(
            input_dim=len(FEATURE_COLS), hidden_dim=hidden, n_layers=1, head_hidden=hidden,
        ).to(device)
    else:
        m1 = VerhaltensGRU(
            input_dim=len(FEATURE_COLS), hidden_dim=hidden, n_layers=1, head_hidden=hidden,
        ).to(device)

    m2 = UebungstypClassifier(input_dim=5 + hidden, hidden_dims=(128, 72, 48)).to(device)
    m3 = KatalogMatcher(embed_dim=128, hidden_dim=128, mlp_blocks=2).to(device)
    return {"m1": m1, "m2": m2, "m3": m3}


# [AI-assisted] combined multi-head loss — prompt=docs/ai_prompts/training_common.md
def make_loss_fn(models, katalog_matrix):
    mse = nn.MSELoss()
    ce = nn.CrossEntropyLoss()
    m1, m2, m3 = models["m1"], models["m2"], models["m3"]

    def loss_fn(x, yf, yk, ykat):
        pred, hidden = m1(x)
        logits = m2(pred, hidden)
        scores = m3(pred, logits, katalog_matrix)
        return MSE_WEIGHT * mse(pred, yf) + ce(logits, yk) + KATALOG_WEIGHT * ce(scores, ykat)

    return loss_fn
# --- end AI-assisted (make_loss_fn) ---


@torch.no_grad()
def evaluate(models, loader, katalog_matrix, device):
    for m in models.values():
        m.eval()

    mse = nn.MSELoss()
    ce = nn.CrossEntropyLoss()
    m1, m2, m3 = models["m1"], models["m2"], models["m3"]

    total_loss = 0.0
    correct_k = 0
    correct_kat = 0
    n = 0

    for x, yf, yk, ykat in loader:
        x, yf, yk, ykat = x.to(device), yf.to(device), yk.to(device), ykat.to(device)
        pred, hidden = m1(x)
        logits = m2(pred, hidden)
        scores = m3(pred, logits, katalog_matrix)
        total_loss += (
            MSE_WEIGHT * mse(pred, yf) + ce(logits, yk) + KATALOG_WEIGHT * ce(scores, ykat)
        ).item()
        correct_k += (logits.argmax(-1) == yk).sum().item()
        correct_kat += (scores.argmax(-1) == ykat).sum().item()
        n += yk.size(0)

    for m in models.values():
        m.train()

    return {
        "loss": total_loss / max(len(loader), 1),
        "acc_klasse": correct_k / n * 100 if n else 0.0,
        "acc_katalog": correct_kat / n * 100 if n else 0.0,
    }


def repo_path(*parts):
    root = os.path.dirname(os.path.abspath(__file__))
    return os.path.normpath(os.path.join(root, *parts))


def tb_cmd(log_dir):
    return (
        f'tensorboard --logdir "{os.path.abspath(log_dir)}" '
        f"--samples_per_plugin=scalars={TB_SAMPLES}"
    )
