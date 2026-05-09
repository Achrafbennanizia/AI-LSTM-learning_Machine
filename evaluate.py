"""
GuitarAI — Evaluation nach dem Training
Confusion Matrix + Forgetting Rate + Empfehlungs-Demo
"""
import os, sys
import pandas as pd
import numpy as np
import torch
from sklearn.metrics import confusion_matrix, classification_report

sys.path.append(os.path.dirname(__file__))
from models.lstm_module       import VerhaltensLSTM
from models.classifier_module import UebungstypClassifier
from models.katalog_matcher   import (
    KatalogMatcher,
    KatalogMatcherLegacy,
    get_katalog_matrix,
    KATALOG_IDS,
)
from train import LeoDataset, FEATURE_COLS
from paths import resolve_training_csv

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_models_from_checkpoint(ckpt):
    """Lädt gleiche Netzwerkform wie beim Training aus ckpt[\"arch\"] (oder kleines Legacy-Netz)."""
    arch = ckpt.get("arch")
    if arch is None:
        print("Hinweis: Checkpoint ohne 'arch' → Legacy-Architektur wie train.py-Defaults (64×1, MLP 128–72–48).")
        m1 = VerhaltensLSTM(hidden_dim=64, n_layers=1, dropout=0.2, head_hidden=64)
        m2 = UebungstypClassifier(input_dim=69, hidden_dims=(128, 72, 48), dropout=0.2)
        m3 = KatalogMatcherLegacy()
        return m1, m2, m3

    m1 = VerhaltensLSTM(
        input_dim=arch.get("n_features", 13),
        hidden_dim=arch["hidden"],
        n_layers=arch["lstm_layers"],
        dropout=arch["lstm_dropout"],
        head_hidden=arch["head_hidden"],
    )
    m2 = UebungstypClassifier(
        input_dim=5 + arch["hidden"],
        hidden_dims=tuple(arch["cls_dims"]),
        dropout=arch["lstm_dropout"],
    )
    m3 = KatalogMatcher(
        embed_dim=arch["katalog_embed"],
        hidden_dim=arch["katalog_hidden"],
        mlp_blocks=arch["katalog_mlp_blocks"],
        dropout=arch["katalog_dropout"],
    )
    return m1, m2, m3


# Modell laden
try:
    ckpt = torch.load("checkpoints/best_model.pt", map_location=device, weights_only=False)
except TypeError:
    ckpt = torch.load("checkpoints/best_model.pt", map_location=device)
m1, m2, m3 = build_models_from_checkpoint(ckpt)
m1 = m1.to(device)
m2 = m2.to(device)
m3 = m3.to(device)
m1.load_state_dict(ckpt["m1"])
m2.load_state_dict(ckpt["m2"])
m3.load_state_dict(ckpt["m3"])
m1.eval(); m2.eval(); m3.eval()
katalog_matrix = get_katalog_matrix().to(device)

print(f"Modell geladen (Epoch {ckpt['epoch']}, Test Loss {ckpt['test_loss']:.4f})")

_arch = ckpt.get("arch")
if _arch:
    print(
        f"Checkpoint-Architektur: n_features={_arch.get('n_features', '?')}, "
        f"hidden={_arch.get('hidden')}, lstm_layers={_arch.get('lstm_layers')}, "
        f"head_hidden={_arch.get('head_hidden')}"
    )
_trained = ckpt.get("data_path")
if _trained:
    print(f"Training-CSV (im Checkpoint): {_trained}")

data_csv = resolve_training_csv()
if _trained:
    if os.path.normpath(os.path.abspath(_trained)) != os.path.normpath(os.path.abspath(data_csv)):
        print(
            "WARNUNG: Eval-CSV unterscheidet sich von der beim Training — "
            "`evaluate.py`/`GUITARAI_DATA` auf dieselbe CSV wie `train.py` setzen."
        )
print(f"Eval-CSV: {data_csv}")

df = pd.read_csv(data_csv)
ds  = LeoDataset(df, seq_len=5)

# ── Confusion Matrix (Übungstyp) ──────────────────────────────────────────────
y_true, y_pred_k, y_pred_kat = [], [], []

with torch.no_grad():
    for x, yf, yk, ykat in torch.utils.data.DataLoader(ds, batch_size=32):
        x = x.to(device)
        v, h  = m1(x)
        kl    = m2(v, h)
        ks    = m3(v, kl, katalog_matrix)
        y_true.append(yk.numpy())
        y_pred_k.append(kl.argmax(-1).cpu().numpy())
        y_pred_kat.append(ks.argmax(-1).cpu().numpy())

y_true    = np.concatenate(y_true)
y_pred_k  = np.concatenate(y_pred_k)
y_pred_kat= np.concatenate(y_pred_kat)

TYPEN = {0:"Einzelakkord",1:"Zwei-Akkord",2:"Rhythmus",3:"Vollstück",4:"Technik"}

print("\n=== CONFUSION MATRIX — Übungstyp-Classifier ===")
cm = confusion_matrix(y_true, y_pred_k)
print("         " + " ".join(f"{v:>12}" for v in TYPEN.values()))
for i, row in enumerate(cm):
    print(f"{TYPEN[i]:>12} " + " ".join(f"{v:>12}" for v in row))

print("\n=== CLASSIFICATION REPORT ===")
print(classification_report(
    y_true, y_pred_k,
    labels=list(TYPEN.keys()),
    target_names=list(TYPEN.values()),
    zero_division=0,
))

acc_k = (y_true == y_pred_k).mean() * 100
print(f"Accuracy Übungstyp: {acc_k:.1f}%")

# ── Empfehlungs-Demo mit letzten 5 Sessions eines Nutzers ───────────────────
print("\n=== DEMO — was würde das System jetzt empfehlen? ===")
_demo_uid = df["nutzer_id"].iloc[0]
leo_sessions = df[df["nutzer_id"] == _demo_uid].sort_values("session").tail(5)
x_demo = torch.tensor(leo_sessions[FEATURE_COLS].values, dtype=torch.float32)
x_demo = x_demo.unsqueeze(0).to(device)  # (1, 5, n_features)

with torch.no_grad():
    v, h    = m1(x_demo)
    kl      = m2(v, h)
    ks      = m3(v, kl, katalog_matrix)

    best_kat_idx = ks.argmax(-1).item()
    best_typ_idx = kl.argmax(-1).item()

print(f"Empfohlene Übung:  {KATALOG_IDS[best_kat_idx]}")
print(f"Übungstyp:         {TYPEN[best_typ_idx]}")
print(f"Vorhergesagte Fehlerraten nächste Session:")
v_np = v.cpu().numpy()[0]
for name, val in zip(["e_griff","e_druck","e_timing","e_technik","e_muting"], v_np):
    bar = "█" * int(val * 20)
    print(f"  {name:<12} {val:.3f}  {bar}")
