"""
GuitarAI — Architektur-Vergleich (Forschungsskript)
====================================================
Trainiert und vergleicht alle 5 Architekturen in einem Lauf:

  1. MLP  (ohne EWC) — Baseline
  2. MLP  (mit EWC)  — zeigt EWC-Nutzen ohne Sequenz
  3. LSTM (mit EWC)  — Original-Architektur, jetzt korrekt inkrementell
  4. GRU  (mit EWC)  — leichtere Sequenz-Alternative
  5. DQN             — Reinforcement-Learning-Ansatz

Alle Architekturen werden auf den GLEICHEN Daten trainiert und
mit den GLEICHEN Metriken evaluiert → fairer Vergleich.

TensorBoard zeigt alle 5 Kurven übereinander:
  tensorboard --logdir runs_compare

Forschungsfragen die dieser Vergleich beantwortet:
  - Hilft zeitliches Gedächtnis (LSTM/GRU) gegenüber MLP?
  - Wie groß ist der EWC-Effekt auf Catastrophic Forgetting?
  - Ist RL (DQN) ohne vorgegebene Labels praktisch nutzbar?
  - Welche Architektur vergisst am wenigsten nach neuen Sessions?

Usage:
  python research/train_compare.py --pretrain_epochs 50 \\
      --n_incremental_tasks 10 --incremental_epochs 5
"""
import os, sys, json, argparse
from datetime import datetime
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

_RESEARCH_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_RESEARCH_DIR)
for _p in (_REPO_ROOT, _RESEARCH_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from paths import resolve_training_csv
from io_layout import (
    TB_SCALAR_SAMPLES,
    artifacts_compare,
    export_results_table_csv,
    exports_dir,
    ensure_dirs,
    get_research_root,
    tb_launch_all,
    tb_launch_single,
    tensorboard_compare,
    write_manifest,
)
from models.lstm_module       import VerhaltensLSTM
from models.gru_module        import VerhaltensGRU
from models.mlp_module        import VerhaltensMLPHead
from models.classifier_module import UebungstypClassifier
from models.katalog_matcher   import KatalogMatcher, get_katalog_matrix, KATALOG_IDS
from models.dqn_module        import DQNAgent
from incremental_engine       import IncrementalTrainer

# ── Gemeinsame Feature-Spalten ────────────────────────────────────────────────
FEATURE_COLS = [
    "e_griff", "e_druck", "e_timing", "e_technik", "e_muting",
    "v_lern", "k_konsistenz", "d_limit", "p_plateau", "r_repeat",
    "n_sessions", "t_session", "s_level",
    "pause_norm", "akkord_fokus", "wochentag_norm", "tageszeit_norm",
]
N_FEATURES   = len(FEATURE_COLS)   # 17
N_KLASSEN    = 5
N_KATALOG    = 15


# ── Datasets ──────────────────────────────────────────────────────────────────

class SessionDataset(Dataset):
    """Sequenzielles Dataset (seq_len > 1) für LSTM/GRU."""
    def __init__(self, df, seq_len=5):
        self.samples = []
        for _, gruppe in df.groupby("nutzer_id"):
            gruppe = gruppe.sort_values("session").reset_index(drop=True)
            feats        = gruppe[FEATURE_COLS].values.astype(np.float32)
            labels_clf   = gruppe["uebungstyp_label"].values.astype(np.int64)
            labels_kat   = gruppe["empfehlung_id"].values
            for i in range(seq_len, len(gruppe)):
                x       = feats[i - seq_len:i]
                y_fehler = feats[i, :5]
                y_klasse = labels_clf[i]
                kat_id   = labels_kat[i]
                y_katalog = KATALOG_IDS.index(kat_id) if kat_id in KATALOG_IDS else 0
                self.samples.append((x, y_fehler, y_klasse, y_katalog))

    def __len__(self):  return len(self.samples)
    def __getitem__(self, idx):
        x, yf, yk, ykat = self.samples[idx]
        return torch.tensor(x), torch.tensor(yf), torch.tensor(yk), torch.tensor(ykat)


class FlatDataset(Dataset):
    """Flaches Dataset (kein seq_len) für MLP."""
    def __init__(self, df):
        self.samples = []
        for _, gruppe in df.groupby("nutzer_id"):
            gruppe = gruppe.sort_values("session").reset_index(drop=True)
            feats       = gruppe[FEATURE_COLS].values.astype(np.float32)
            labels_clf  = gruppe["uebungstyp_label"].values.astype(np.int64)
            labels_kat  = gruppe["empfehlung_id"].values
            for i in range(1, len(gruppe)):
                x        = feats[i - 1]     # nur vorherige Session, kein Fenster
                y_fehler = feats[i, :5]
                y_klasse = labels_clf[i]
                kat_id   = labels_kat[i]
                y_katalog = KATALOG_IDS.index(kat_id) if kat_id in KATALOG_IDS else 0
                self.samples.append((x, y_fehler, y_klasse, y_katalog))

    def __len__(self):  return len(self.samples)
    def __getitem__(self, idx):
        x, yf, yk, ykat = self.samples[idx]
        return torch.tensor(x), torch.tensor(yf), torch.tensor(yk), torch.tensor(ykat)


# ── Datensplit auf Nutzer-Ebene ───────────────────────────────────────────────

def split_by_user(df, pretrain_ratio=0.6, test_ratio=0.2, seed=42):
    """
    Teilt Nutzer in 3 Gruppen:
      - Pretrain: Vortraining (Basis-Wissen aufbauen)
      - Incremental: sequenziell eingespeist (eine Nutzergruppe nach der anderen)
      - Test: nie gesehen, für finale Evaluation

    Warum Nutzer-Split statt Zeilen-Split:
      - Verhindert Datenleckage (Bug 1 aus der alten Version)
      - Simuliert realistische Situation: neues Gerät für neuen Nutzer
    """
    rng = np.random.RandomState(seed)
    users = df["nutzer_id"].unique()
    rng.shuffle(users)

    n = len(users)
    n_pretrain     = max(1, int(n * pretrain_ratio))
    n_test         = max(1, int(n * test_ratio))
    n_incremental  = max(0, n - n_pretrain - n_test)

    pretrain_users     = users[:n_pretrain]
    incremental_users  = users[n_pretrain:n_pretrain + n_incremental]
    test_users         = users[n_pretrain + n_incremental:]

    return (
        df[df["nutzer_id"].isin(pretrain_users)],
        df[df["nutzer_id"].isin(incremental_users)],
        df[df["nutzer_id"].isin(test_users)],
    )


# ── Modell-Builder ───────────────────────────────────────────────────────────

def build_seq_models(arch, device, hidden=64, dropout=0.2,
                     katalog_embed=128, katalog_hidden=128):
    """Baut M1 (LSTM oder GRU) + M2 + M3."""
    if arch == "lstm":
        m1 = VerhaltensLSTM(
            input_dim=N_FEATURES, hidden_dim=hidden,
            n_layers=1, dropout=dropout, head_hidden=hidden,
        ).to(device)
    elif arch == "gru":
        m1 = VerhaltensGRU(
            input_dim=N_FEATURES, hidden_dim=hidden,
            n_layers=1, dropout=dropout, head_hidden=hidden,
        ).to(device)
    else:
        raise ValueError(f"Unbekannte Architektur: {arch}")

    m2 = UebungstypClassifier(
        input_dim=5 + hidden, hidden_dims=(128, 72, 48), dropout=dropout
    ).to(device)
    m3 = KatalogMatcher(
        embed_dim=katalog_embed, hidden_dim=katalog_hidden,
        mlp_blocks=2, dropout=dropout,
    ).to(device)
    return {"m1": m1, "m2": m2, "m3": m3}


def build_mlp_models(device, hidden=64, dropout=0.2,
                     katalog_embed=128, katalog_hidden=128):
    """Baut MLP-M1 + M2 + M3."""
    m1 = VerhaltensMLPHead(
        input_dim=N_FEATURES, hidden_dims=(128, hidden), dropout=dropout
    ).to(device)
    m2 = UebungstypClassifier(
        input_dim=5 + hidden, hidden_dims=(128, 72, 48), dropout=dropout
    ).to(device)
    m3 = KatalogMatcher(
        embed_dim=katalog_embed, hidden_dim=katalog_hidden,
        mlp_blocks=2, dropout=dropout,
    ).to(device)
    return {"m1": m1, "m2": m2, "m3": m3}


# ── Kombinierter Loss ─────────────────────────────────────────────────────────

def make_loss_fn(models, katalog_matrix, device, is_mlp=False):
    """Gibt eine Loss-Funktion zurück die auf einem Batch operiert."""
    mse  = nn.MSELoss()
    ce   = nn.CrossEntropyLoss()
    ce_k = nn.CrossEntropyLoss()
    m1, m2, m3 = models["m1"], models["m2"], models["m3"]

    def loss_fn(x, yf, yk, ykat):
        if is_mlp:
            # MLP: x hat Form (batch, n_features) — kein seq_len
            vorhersage, hidden = m1(x)
        else:
            # LSTM/GRU: x hat Form (batch, seq_len, n_features)
            vorhersage, hidden = m1(x)

        klasse_logits = m2(vorhersage, hidden)
        kat_scores    = m3(vorhersage, klasse_logits, katalog_matrix)

        loss = (0.5 * mse(vorhersage, yf) +
                ce(klasse_logits, yk) +
                0.8 * ce_k(kat_scores, ykat))
        return loss

    return loss_fn


# ── Evaluation ───────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(models, dataloader, katalog_matrix, device, is_mlp=False):
    """Berechnet Test-Loss, Übungstyp-Acc und Katalog-Acc."""
    for m in models.values():
        m.eval()

    mse  = nn.MSELoss()
    ce   = nn.CrossEntropyLoss()
    ce_k = nn.CrossEntropyLoss()
    m1, m2, m3 = models["m1"], models["m2"], models["m3"]

    total_loss = correct_k = correct_kat = total = 0

    for x, yf, yk, ykat in dataloader:
        x, yf, yk, ykat = x.to(device), yf.to(device), yk.to(device), ykat.to(device)
        vorhersage, hidden = m1(x)
        klasse_logits      = m2(vorhersage, hidden)
        kat_scores         = m3(vorhersage, klasse_logits, katalog_matrix)

        loss = (0.5 * mse(vorhersage, yf) +
                ce(klasse_logits, yk) +
                0.8 * ce_k(kat_scores, ykat))
        total_loss += loss.item()

        correct_k   += (klasse_logits.float().argmax(-1) == yk.long()).sum().item()
        correct_kat += (kat_scores.float().argmax(-1)   == ykat.long()).sum().item()
        total += yk.size(0)

    for m in models.values():
        m.train()

    n = max(len(dataloader), 1)
    return {
        "loss":       total_loss / n,
        "acc_klasse": correct_k   / max(total, 1) * 100,
        "acc_katalog": correct_kat / max(total, 1) * 100,
    }


# ── DQN Training Loop ────────────────────────────────────────────────────────

def train_dqn(df_pretrain, df_incremental, df_test, device, args, tb_writer):
    """Separater Training-Loop für DQN (andere Update-Logik als supervised).

    Konvertiert das supervised-Learning-Problem in ein RL-Problem:
    - State  = Feature-Vektor der aktuellen Session
    - Action = Katalog-Empfehlung (0-14)
    - Reward = Verbesserung der Fehlerraten in der nächsten Session
    """
    print("\n" + "="*60)
    print("DQN — Deep Q-Learning")
    print("="*60)

    agent = DQNAgent(
        state_dim=N_FEATURES,
        n_actions=N_KATALOG,
        hidden_dims=(128, 64),
        dropout=0.2,
        lr=args.lr,
        gamma=0.95,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay=0.997,
        buffer_capacity=10_000,
        batch_size=args.batch_size,
        target_update_freq=20,
        device=device,
    )
    print(f"  Parameter: {agent.n_params:,}")

    # Alle Daten für RL-Simulation
    all_df = pd.concat([df_pretrain, df_incremental]).sort_values(
        ["nutzer_id", "session"]
    ).reset_index(drop=True)

    total_steps = 0
    rewards_per_episode = []

    # RL-Simulation: gehe durch alle Nutzer und Sessions
    for nutzer_id, gruppe in all_df.groupby("nutzer_id"):
        gruppe = gruppe.sort_values("session").reset_index(drop=True)
        feats  = gruppe[FEATURE_COLS].values.astype(np.float32)
        labels_kat = gruppe["empfehlung_id"].values

        episode_reward = 0.0
        for i in range(len(gruppe) - 1):
            state      = feats[i]
            next_state = feats[i + 1]

            # Aktion wählen (Katalog-Empfehlung)
            action = agent.select_action(state)

            # Reward: wie stark haben sich Fehlerraten verbessert?
            reward = DQNAgent.compute_reward(
                prev_errors=feats[i, :5],
                next_errors=feats[i + 1, :5],
            )
            episode_reward += reward

            done = (i == len(gruppe) - 2)
            agent.push(state, action, reward, next_state, done)

            # Lernen
            loss = agent.learn()

            if tb_writer and loss is not None:
                tb_writer.add_scalar("dqn/loss_batch", loss, total_steps)
                tb_writer.add_scalar("dqn/epsilon",   agent.epsilon, total_steps)
                tb_writer.add_scalar("dqn/reward",    reward, total_steps)
            total_steps += 1

        rewards_per_episode.append(episode_reward)
        if tb_writer:
            tb_writer.add_scalar("dqn/episode_reward", episode_reward, len(rewards_per_episode))

    # DQN Evaluation: Katalog-Acc auf Test-Set
    test_df = df_test.sort_values(["nutzer_id", "session"]).reset_index(drop=True)
    correct_kat = 0
    total_kat   = 0
    for _, gruppe in test_df.groupby("nutzer_id"):
        gruppe  = gruppe.sort_values("session").reset_index(drop=True)
        feats   = gruppe[FEATURE_COLS].values.astype(np.float32)
        labels  = gruppe["empfehlung_id"].values
        for i in range(len(gruppe) - 1):
            action = agent.select_action(feats[i])
            true_kat = KATALOG_IDS.index(labels[i]) if labels[i] in KATALOG_IDS else 0
            if action == true_kat:
                correct_kat += 1
            total_kat += 1

    acc_katalog = correct_kat / max(total_kat, 1) * 100
    avg_reward  = sum(rewards_per_episode) / max(len(rewards_per_episode), 1)

    print(f"  Katalog-Acc (Test): {acc_katalog:.1f}%")
    print(f"  Avg Episode Reward: {avg_reward:.4f}")
    print(f"  Final Epsilon:      {agent.epsilon:.4f}")
    print(f"  Buffer size:        {len(agent.buffer)}")

    if tb_writer:
        tb_writer.add_scalar("compare/dqn/acc_katalog", acc_katalog, 0)
        tb_writer.add_scalar("compare/dqn/avg_reward",  avg_reward, 0)
        tb_writer.flush()

    return {
        "arch": "DQN",
        "acc_klasse":  "N/A (RL hat keine Klassen-Labels)",
        "acc_katalog": acc_katalog,
        "avg_reward":  avg_reward,
        "n_params":    agent.n_params,
        "forgetting":  "N/A",
    }


# ── Supervised Architektur trainieren ────────────────────────────────────────

def train_supervised_arch(
    arch_name, models, df_pretrain, df_incremental, df_test,
    katalog_matrix, device, args, use_ewc, tb_writer, is_mlp=False,
):
    """Trainiert eine supervised Architektur (MLP/LSTM/GRU) inkrementell."""
    print(f"\n{'='*60}")
    ewc_tag = "mit EWC" if use_ewc else "ohne EWC"
    print(f"{arch_name} — {ewc_tag}")
    print("="*60)

    n_params = sum(p.numel() for m in models.values() for p in m.parameters())
    print(f"  Parameter: {n_params:,}")

    # Dataset + DataLoader
    DatasetClass = FlatDataset if is_mlp else SessionDataset
    ds_kw = {} if is_mlp else {"seq_len": args.seq_len}

    ds_pretrain = DatasetClass(df_pretrain, **ds_kw)
    ds_test     = DatasetClass(df_test, **ds_kw)

    if len(ds_pretrain) == 0:
        print("  Warnung: Leeres Pretrain-Dataset, überspringe.")
        return None

    dl_pretrain = DataLoader(ds_pretrain, batch_size=args.batch_size, shuffle=True)
    dl_test     = DataLoader(ds_test,     batch_size=args.batch_size, shuffle=False)

    all_params = [p for m in models.values() for p in m.parameters()]
    optimizer  = torch.optim.Adam(all_params, lr=args.lr)
    scheduler  = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=10, factor=0.5
    )

    loss_fn = make_loss_fn(models, katalog_matrix, device, is_mlp=is_mlp)

    run_tag = f"{arch_name.lower()}_{'ewc' if use_ewc else 'noewc'}"
    writer = None
    if tb_writer is not None:
        # Eigener Sub-Writer pro Architektur
        sub_dir = os.path.join(args.log_dir, args.run_name, run_tag)
        os.makedirs(sub_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=sub_dir, flush_secs=5, max_queue=50)

    trainer = IncrementalTrainer(
        models_dict=models,
        optimizer=optimizer,
        loss_fn=loss_fn,
        device=device,
        use_ewc=use_ewc,
        ewc_lambda=args.ewc_lambda,
        use_replay=True,
        replay_capacity=2000,
        replay_ratio=0.3,
        incremental_epochs=args.incremental_epochs,
        tb_writer=writer,
    )

    # ── Phase A: Vortraining ──────────────────────────────────────────────────
    print(f"  Phase A: Vortraining ({args.pretrain_epochs} Epochen)...")
    trainer.pretrain(dl_pretrain, epochs=args.pretrain_epochs,
                     scheduler=scheduler, tag=f"{run_tag}/pretrain")

    metrics_pretrain = evaluate(models, dl_test, katalog_matrix, device, is_mlp=is_mlp)
    print(f"  Nach Vortraining → Loss: {metrics_pretrain['loss']:.4f} | "
          f"Acc Klasse: {metrics_pretrain['acc_klasse']:.1f}% | "
          f"Acc Katalog: {metrics_pretrain['acc_katalog']:.1f}%")

    if writer:
        writer.add_scalar("phase/pretrain_acc_klasse",  metrics_pretrain["acc_klasse"],  0)
        writer.add_scalar("phase/pretrain_acc_katalog", metrics_pretrain["acc_katalog"], 0)

    # ── Phase B: Inkrementelle Updates ───────────────────────────────────────
    incremental_users = df_incremental["nutzer_id"].unique()
    # Teile inkrementelle Nutzer in Tasks auf
    task_size = max(1, len(incremental_users) // max(args.n_incremental_tasks, 1))
    tasks = [
        incremental_users[i:i + task_size]
        for i in range(0, len(incremental_users), task_size)
    ]

    print(f"  Phase B: {len(tasks)} inkrementelle Tasks...")
    acc_after_each_task = []

    for task_idx, task_users in enumerate(tasks):
        df_task = df_incremental[df_incremental["nutzer_id"].isin(task_users)]
        ds_task = DatasetClass(df_task, **ds_kw)
        if len(ds_task) == 0:
            continue
        dl_task = DataLoader(ds_task, batch_size=args.batch_size, shuffle=True)

        trainer.incremental_update(dl_task, task_name=f"task_{task_idx+1:02d}")

        # Evaluation nach jeder Task → zeigt Forgetting-Kurve
        m = evaluate(models, dl_test, katalog_matrix, device, is_mlp=is_mlp)
        acc_after_each_task.append(m["acc_katalog"])
        print(f"    Task {task_idx+1:2d}/{len(tasks)} → "
              f"Katalog-Acc: {m['acc_katalog']:.1f}% | "
              f"EWC Tasks gesehen: {trainer.ewc.n_tasks_seen if trainer.ewc else 0}")

        if writer:
            writer.add_scalar("incremental/acc_klasse",  m["acc_klasse"],  task_idx + 1)
            writer.add_scalar("incremental/acc_katalog", m["acc_katalog"], task_idx + 1)
            writer.add_scalar("incremental/loss",        m["loss"],        task_idx + 1)

    # ── Finale Evaluation ─────────────────────────────────────────────────────
    metrics_final = evaluate(models, dl_test, katalog_matrix, device, is_mlp=is_mlp)
    print(f"\n  Finale Evaluation:")
    print(f"    Loss:        {metrics_final['loss']:.4f}")
    print(f"    Acc Klasse:  {metrics_final['acc_klasse']:.1f}%")
    print(f"    Acc Katalog: {metrics_final['acc_katalog']:.1f}%")

    # Forgetting = Genauigkeitsabfall nach allen inkrementellen Tasks
    if len(acc_after_each_task) >= 2:
        forgetting = acc_after_each_task[0] - acc_after_each_task[-1]
        print(f"    Forgetting:  {forgetting:+.1f}% (negativ = Vergessen)")
    else:
        forgetting = 0.0

    if writer:
        writer.add_scalar("phase/final_acc_klasse",  metrics_final["acc_klasse"],  0)
        writer.add_scalar("phase/final_acc_katalog", metrics_final["acc_katalog"], 0)
        writer.add_scalar("phase/forgetting", forgetting, 0)

        # Vergleichsscalar für alle Architekturen im gleichen Plot
        if tb_writer:
            tb_writer.add_scalar(f"compare/acc_katalog/{run_tag}", metrics_final["acc_katalog"], 0)
            tb_writer.add_scalar(f"compare/forgetting/{run_tag}",  forgetting, 0)
            tb_writer.flush()
        writer.close()

    return {
        "arch":        f"{arch_name} ({'mit' if use_ewc else 'ohne'} EWC)",
        "acc_klasse":  metrics_final["acc_klasse"],
        "acc_katalog": metrics_final["acc_katalog"],
        "loss":        metrics_final["loss"],
        "forgetting":  forgetting,
        "n_params":    n_params,
        "acc_history": acc_after_each_task,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="GuitarAI — Vollständiger Architektur-Vergleich"
    )
    parser.add_argument(
        "--data",
        default=None,
        help="CSV (Default: wie train.py — paths.resolve_training_csv)",
    )
    parser.add_argument("--pretrain_epochs",    type=int,   default=50)
    parser.add_argument("--n_incremental_tasks",type=int,   default=10)
    parser.add_argument("--incremental_epochs", type=int,   default=5)
    parser.add_argument("--batch_size",         type=int,   default=32)
    parser.add_argument("--lr",                 type=float, default=5e-4)
    parser.add_argument("--seq_len",            type=int,   default=5)
    parser.add_argument("--hidden",             type=int,   default=64)
    parser.add_argument("--ewc_lambda",         type=float, default=400.0)
    parser.add_argument("--pretrain_ratio",     type=float, default=0.6)
    parser.add_argument("--test_ratio",         type=float, default=0.2)
    parser.add_argument("--device",             default="auto")
    parser.add_argument(
        "--research_root",
        default=None,
        help=(
            "Elternverzeichnis für research_outputs/ (Standard: <Repo>/research_outputs). "
            "Alternativ: Umgebungsvariable GUITARAI_RESEARCH_ROOT."
        ),
    )
    parser.add_argument(
        "--log_dir",
        default=None,
        help="TensorBoard-Vergleich (Standard: <research_outputs>/tensorboard/compare)",
    )
    parser.add_argument("--run_name", default=None)
    parser.add_argument(
        "--save_dir",
        default=None,
        help="JSON-Artefakte Vergleich (Standard: <research_outputs>/artifacts/compare)",
    )
    parser.add_argument("--skip_dqn",           action="store_true")
    args = parser.parse_args()

    if args.research_root:
        research_base = os.path.normpath(
            args.research_root
            if os.path.isabs(args.research_root)
            else os.path.join(_REPO_ROOT, args.research_root)
        )
    else:
        research_base = get_research_root(_REPO_ROOT)

    if args.log_dir is None:
        args.log_dir = tensorboard_compare(research_base)
    elif not os.path.isabs(args.log_dir):
        args.log_dir = os.path.normpath(os.path.join(_REPO_ROOT, args.log_dir))

    if args.save_dir is None:
        args.save_dir = artifacts_compare(research_base)
    elif not os.path.isabs(args.save_dir):
        args.save_dir = os.path.normpath(os.path.join(_REPO_ROOT, args.save_dir))

    args.exports_dir = exports_dir(research_base)
    ensure_dirs(args.log_dir, args.save_dir, args.exports_dir)

    if args.data is None:
        args.data = resolve_training_csv()

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    ) if args.device == "auto" else torch.device(args.device)
    print(f"Device: {device}")

    # Daten laden und splitten
    df = pd.read_csv(args.data)
    print(f"Datensatz: {len(df)} Zeilen, {df['nutzer_id'].nunique()} Nutzer")

    df_pretrain, df_incremental, df_test = split_by_user(
        df,
        pretrain_ratio=args.pretrain_ratio,
        test_ratio=args.test_ratio,
    )
    print(f"  Pretrain:     {df_pretrain['nutzer_id'].nunique()} Nutzer")
    print(f"  Inkrementell: {df_incremental['nutzer_id'].nunique()} Nutzer")
    print(f"  Test:         {df_test['nutzer_id'].nunique()} Nutzer")

    run_id = args.run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    args.run_name = run_id
    main_tb_dir = os.path.join(args.log_dir, run_id)
    ensure_dirs(main_tb_dir)
    main_writer = SummaryWriter(
        log_dir=main_tb_dir, flush_secs=5, max_queue=100
    )

    print(f"\nResearch-Output-Basis: {research_base}")
    print(f"TensorBoard (alle Compare-Läufe): {tb_launch_all(args.log_dir)}")
    print(f"TensorBoard (nur dieser Run):      {tb_launch_single(main_tb_dir)}")

    _manifest_pre = {
        "run_id": run_id,
        "research_base": os.path.abspath(research_base),
        "data_csv": os.path.abspath(str(args.data)),
        "tensorboard_root": os.path.abspath(args.log_dir),
        "tensorboard_this_run": os.path.abspath(main_tb_dir),
        "artifacts_dir": os.path.abspath(args.save_dir),
        "exports_dir": os.path.abspath(args.exports_dir),
        "samples_per_plugin_scalars": TB_SCALAR_SAMPLES,
        "device": str(device),
        "rows_total": int(len(df)),
        "users_total": int(df["nutzer_id"].nunique()),
        "users_pretrain": int(df_pretrain["nutzer_id"].nunique()),
        "users_incremental": int(df_incremental["nutzer_id"].nunique()),
        "users_test": int(df_test["nutzer_id"].nunique()),
    }
    main_writer.add_text(
        "run/summary",
        json.dumps(_manifest_pre, indent=2, default=str),
        0,
    )
    main_writer.add_text(
        "run/hparams",
        json.dumps(
            {
                "pretrain_epochs": args.pretrain_epochs,
                "n_incremental_tasks": args.n_incremental_tasks,
                "incremental_epochs": args.incremental_epochs,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "seq_len": args.seq_len,
                "hidden": args.hidden,
                "ewc_lambda": args.ewc_lambda,
                "pretrain_ratio": args.pretrain_ratio,
                "test_ratio": args.test_ratio,
                "skip_dqn": args.skip_dqn,
            },
            indent=2,
        ),
        0,
    )
    main_writer.add_scalar("data/n_rows", float(len(df)), 0)
    main_writer.add_scalar("data/n_users", float(df["nutzer_id"].nunique()), 0)
    main_writer.add_scalar("split/users_pretrain", float(df_pretrain["nutzer_id"].nunique()), 0)
    main_writer.add_scalar("split/users_incremental", float(df_incremental["nutzer_id"].nunique()), 0)
    main_writer.add_scalar("split/users_test", float(df_test["nutzer_id"].nunique()), 0)
    main_writer.flush()

    katalog_matrix = get_katalog_matrix().to(device)

    all_results = []

    # ── 1. MLP ohne EWC (Baseline) ───────────────────────────────────────────
    models = build_mlp_models(device, hidden=args.hidden)
    result = train_supervised_arch(
        "MLP", models, df_pretrain, df_incremental, df_test,
        katalog_matrix, device, args,
        use_ewc=False, tb_writer=main_writer, is_mlp=True,
    )
    if result: all_results.append(result)

    # ── 2. MLP mit EWC ───────────────────────────────────────────────────────
    models = build_mlp_models(device, hidden=args.hidden)
    result = train_supervised_arch(
        "MLP", models, df_pretrain, df_incremental, df_test,
        katalog_matrix, device, args,
        use_ewc=True, tb_writer=main_writer, is_mlp=True,
    )
    if result: all_results.append(result)

    # ── 3. LSTM mit EWC ──────────────────────────────────────────────────────
    models = build_seq_models("lstm", device, hidden=args.hidden)
    result = train_supervised_arch(
        "LSTM", models, df_pretrain, df_incremental, df_test,
        katalog_matrix, device, args,
        use_ewc=True, tb_writer=main_writer, is_mlp=False,
    )
    if result: all_results.append(result)

    # ── 4. GRU mit EWC ───────────────────────────────────────────────────────
    models = build_seq_models("gru", device, hidden=args.hidden)
    result = train_supervised_arch(
        "GRU", models, df_pretrain, df_incremental, df_test,
        katalog_matrix, device, args,
        use_ewc=True, tb_writer=main_writer, is_mlp=False,
    )
    if result: all_results.append(result)

    # ── 5. DQN ───────────────────────────────────────────────────────────────
    if not args.skip_dqn:
        result = train_dqn(
            df_pretrain, df_incremental, df_test,
            device, args, main_writer,
        )
        if result: all_results.append(result)

    # ── Zusammenfassung ───────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("ERGEBNISSE — ARCHITEKTUR-VERGLEICH")
    print("="*60)
    print(f"{'Architektur':<25} {'Klasse-Acc':>10} {'Katalog-Acc':>12} "
          f"{'Forgetting':>11} {'Parameter':>10}")
    print("-"*70)
    for r in all_results:
        acc_k  = f"{r['acc_klasse']:.1f}%" if isinstance(r['acc_klasse'],  float) else r['acc_klasse']
        acc_kat = f"{r['acc_katalog']:.1f}%" if isinstance(r['acc_katalog'], float) else r['acc_katalog']
        forget  = f"{r['forgetting']:+.1f}%"  if isinstance(r['forgetting'],  float) else r['forgetting']
        n_p     = f"{r['n_params']:,}"         if isinstance(r['n_params'],    int)   else r['n_params']
        print(f"{r['arch']:<25} {acc_k:>10} {acc_kat:>12} {forget:>11} {n_p:>10}")

    for i, r in enumerate(all_results):
        if isinstance(r.get("acc_katalog"), (int, float)):
            main_writer.add_scalar("summary/final_acc_katalog_by_order", float(r["acc_katalog"]), i)
        if isinstance(r.get("forgetting"), (int, float)):
            main_writer.add_scalar("summary/forgetting_by_order", float(r["forgetting"]), i)
    main_writer.flush()

    # Als JSON + CSV + Manifest
    os.makedirs(args.save_dir, exist_ok=True)
    out_path = os.path.join(args.save_dir, f"comparison_{run_id}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nErgebnisse gespeichert: {out_path}")

    csv_path = os.path.join(args.exports_dir, f"compare_summary_{run_id}.csv")
    export_results_table_csv(csv_path, all_results)
    print(f"CSV (für R/Excel):    {csv_path}")

    write_manifest(
        os.path.join(args.save_dir, f"run_manifest_{run_id}.json"),
        {
            **_manifest_pre,
            "comparison_json": os.path.abspath(out_path),
            "compare_summary_csv": os.path.abspath(csv_path),
            "n_architectures_ran": len(all_results),
        },
    )

    main_writer.close()
    print(f"\nTensorBoard erneut starten:\n  {tb_launch_single(main_tb_dir)}")


if __name__ == "__main__":
    main()
