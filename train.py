"""
GuitarAI — Haupt-Training
Läuft lokal UND auf dem HPC (SLURM)
"""
import os, sys, json, argparse
from datetime import datetime
from contextlib import nullcontext
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
# Module importieren
sys.path.append(os.path.dirname(__file__))
from models.lstm_module      import VerhaltensLSTM
from models.classifier_module import UebungstypClassifier
from models.katalog_matcher  import KatalogMatcher, get_katalog_matrix, KATALOG_IDS
from paths import resolve_training_csv

# ─── Shared dataset config (von evaluate.py importierbar ohne Training zu starten) ─
FEATURE_COLS = [
    "e_griff", "e_druck", "e_timing", "e_technik", "e_muting",
    "v_lern", "k_konsistenz", "d_limit", "p_plateau", "r_repeat",
    "n_sessions", "t_session", "s_level",
    "pause_norm", "akkord_fokus", "wochentag_norm", "tageszeit_norm",
]

# TensorBoard UI downsamples scalars per tag by default (~1k–10k); raise for full-length curves.
TENSORBOARD_SCALAR_SAMPLES = 10_000_000


def tensorboard_launch_cmd(log_dir: str) -> str:
    root = os.path.abspath(log_dir)
    return (
        f'tensorboard --logdir "{root}" '
        f"--samples_per_plugin=scalars={TENSORBOARD_SCALAR_SAMPLES}"
    )


def tensorboard_single_run_cmd(run_subdir: str) -> str:
    """TensorBoard nur für einen Unterordner (keine Überlagerung älterer Läufe)."""
    return (
        f'tensorboard --logdir "{os.path.abspath(run_subdir)}" '
        f"--samples_per_plugin=scalars={TENSORBOARD_SCALAR_SAMPLES}"
    )


class LeoDataset(Dataset):
    def __init__(self, df, seq_len=5):
        self.seq_len = seq_len
        self.samples = []

        # Pro Nutzer: gleitendes Fenster über Sessions
        for nutzer_id, gruppe in df.groupby("nutzer_id"):
            gruppe = gruppe.sort_values("session").reset_index(drop=True)
            feats  = gruppe[FEATURE_COLS].values.astype(np.float32)
            labels_clf  = gruppe["uebungstyp_label"].values.astype(np.int64)
            labels_kat  = gruppe["empfehlung_id"].values

            for i in range(seq_len, len(gruppe)):
                # Input: letzte seq_len Sessions
                x = feats[i-seq_len:i]           # (seq_len, n_features)

                # Target 1: vorhergesagte Fehlerraten (nächste Session)
                y_fehler = feats[i, :5]           # (5,) = e_griff..e_muting

                # Target 2: Übungstyp-Klasse
                y_klasse = labels_clf[i]          # int

                # Target 3: Katalog-Index
                katalog_id = labels_kat[i]
                y_katalog  = KATALOG_IDS.index(katalog_id) if katalog_id in KATALOG_IDS else 0

                self.samples.append((x, y_fehler, y_klasse, y_katalog))

    def __len__(self):  return len(self.samples)
    def __getitem__(self, idx):
        x, yf, yk, ykat = self.samples[idx]
        return (torch.tensor(x),
                torch.tensor(yf),
                torch.tensor(yk),
                torch.tensor(ykat))


def main():
    # ─── Config ───────────────────────────────────────────────────────────────
    parser = argparse.ArgumentParser(
        description=(
            "GuitarAI: Deep-Learning-Training — Verhaltens-LSTM (sequenziell) + "
            "Übungstyp-MLP + Katalog-Matcher (Dual-Encoder + Similarity)."
        ),
        epilog=(
            "Ohne Flags: kompaktes Standardnetz — LSTM hidden=64 × lstm_layers=1, batch_size=32, lr=5e-4, "
            "cls MLP 128–72–48 (~116k trainierbare Parameter für m1+m2+m3 bei Default-Katalogkopf)."
        ),
    )
    parser.add_argument(
        "--data",
        default=resolve_training_csv(),
        help="Pfad CSV ($GUITARAI_DATA/_DIR oder data/leo_50_sessions.csv im Repo)",
    )
    parser.add_argument("--epochs",     type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr",         type=float, default=0.0005)
    parser.add_argument("--seq_len",    type=int, default=5)   # letzte N Sessions als Input
    parser.add_argument("--hidden",     type=int, default=64, help="LSTM hidden units je Schicht")
    parser.add_argument("--lstm_layers", type=int, default=1, help="Gestapelte LSTM-Schichten")
    parser.add_argument("--lstm_dropout", type=float, default=0.2, help="Dropout zwischen LSTM & FC-Kopf")
    parser.add_argument(
        "--head_hidden",
        type=int,
        default=64,
        help="Zwischen-Linearschicht LSTM→Output (0 = ein Layer wie klassisch)",
    )
    parser.add_argument(
        "--cls_hidden",
        type=str,
        default="128,72,48",
        help="Komma-getrennte Hidden-Größen für Übungstyp-MLP",
    )
    parser.add_argument("--katalog_embed", type=int, default=128)
    parser.add_argument("--katalog_hidden", type=int, default=128)
    parser.add_argument("--katalog_mlp_blocks", type=int, default=2, help=">=1 ReLU-Blöcke pro Encoder")
    parser.add_argument("--katalog_dropout", type=float, default=0.2)
    parser.add_argument("--device",     default="auto")
    parser.add_argument("--save_dir",   default="checkpoints")
    parser.add_argument(
        "--num_workers",
        type=int,
        default=-1,
        help="DataLoader workers (-1 = min(4, CPUs) unter Linux)",
    )
    parser.add_argument(
        "--no_amp",
        action="store_true",
        help="Mixed Precision auf CUDA abschalten",
    )
    parser.add_argument(
        "--log_dir",
        default="runs",
        help="TensorBoard Basisordner (pro Lauf Unterordner, siehe --run_name)",
    )
    parser.add_argument(
        "--run_name",
        default=None,
        help="Name des TensorBoard-Unterordners unter log_dir (Default: Zeitstempel)",
    )
    parser.add_argument(
        "--no_tensorboard",
        action="store_true",
        help="Keine TensorBoard-Events schreiben",
    )
    parser.add_argument(
        "--no_tb_batch_scalars",
        action="store_true",
        help="TensorBoard: keine Scalars pro Minibatch (Xs-Achse fällt auf ~Anzahl Epochen zurück)",
    )
    args = parser.parse_args()

    if args.katalog_mlp_blocks < 1:
        raise SystemExit("--katalog_mlp_blocks muss >= 1 sein")

    cls_dims = [int(x.strip()) for x in args.cls_hidden.split(",") if x.strip()]
    if len(cls_dims) < 1:
        raise SystemExit("--cls_hidden braucht mindestens einen Wert z.B. 128,72,48")

    # Device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    use_amp = device.type == "cuda" and not args.no_amp
    print(f"Device: {device}" + (" (AMP float16 aktiv)" if use_amp else ""))
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # ─── Daten laden ──────────────────────────────────────────────────────────
    df = pd.read_csv(args.data)
    print(f"CSV:       {args.data}")
    print(f"Datensatz: {len(df)} Sessions, {df['nutzer_id'].nunique()} Nutzer")

    # Train/Test Split auf Session-Ebene (bei wenigen Nutzern)
    n = len(df); split = int(n * 0.8)
    df_train = df.iloc[:split]; df_test = df.iloc[split:]

    ds_train = LeoDataset(df_train, args.seq_len)
    ds_test  = LeoDataset(df_test,  args.seq_len)
    if len(ds_train) == 0:
        raise SystemExit(
            f"Keine Trainingsfenster (seq_len={args.seq_len}): "
            "pro Nutzer mindestens seq_len+1 Sessions nötig, CSV prüfen."
        )
    if len(ds_test) == 0:
        print(
            "Warnung: 0 Test-Samples (Zeilen-Split klein oder Nutzer ohne Fenster im Testteil). "
            "Validierung nutzt gleiche Fenster wie Training — für echtes Testset Nutzer-/Session-Split erwägt."
        )
        ds_test = ds_train  # Crash vermeiden; Metriken entsprechend vorsichtig interpretieren

    if args.num_workers < 0:
        try:
            ncpu = os.cpu_count() or 1
            num_workers = min(4, max(0, ncpu // 4))
            if sys.platform == "darwin":
                num_workers = 0
        except Exception:
            num_workers = 0
    else:
        num_workers = args.num_workers
    dl_kw = dict(
        batch_size=args.batch_size,
        pin_memory=(device.type == "cuda"),
        persistent_workers=num_workers > 0,
        num_workers=num_workers,
    )
    dl_train = DataLoader(ds_train, shuffle=True, **dl_kw)
    dl_test  = DataLoader(ds_test, shuffle=False, **dl_kw)
    print(f"Train: {len(ds_train)} Samples | Test: {len(ds_test)} Samples")
    arch = {
        "n_features": len(FEATURE_COLS),
        "hidden": args.hidden,
        "lstm_layers": args.lstm_layers,
        "lstm_dropout": args.lstm_dropout,
        "head_hidden": args.head_hidden,
        "cls_dims": cls_dims,
        "katalog_embed": args.katalog_embed,
        "katalog_hidden": args.katalog_hidden,
        "katalog_mlp_blocks": args.katalog_mlp_blocks,
        "katalog_dropout": args.katalog_dropout,
    }
    print(
        "Architektur:",
        f"LSTM×{args.lstm_layers} hidden={args.hidden}, head_fc={args.head_hidden}; ",
        f"Classifier MLP={cls_dims}; Katalog embed={args.katalog_embed}",
    )
    print(
        f"  → Sequenz: seq_len={args.seq_len} × {len(FEATURE_COLS)} Merkmale; "
        f"Köpfe: (1) Fehlerraten σ, (2) Übungstyp 5 Klassen, (3) Katalog 15 Weights (Matmul-Scores)."
    )
    print(f"  → Training: batch_size={args.batch_size}, lr={args.lr}")

    # ─── Modelle ──────────────────────────────────────────────────────────────
    m1 = VerhaltensLSTM(
        input_dim=len(FEATURE_COLS),
        hidden_dim=args.hidden,
        n_layers=args.lstm_layers,
        dropout=args.lstm_dropout,
        head_hidden=args.head_hidden,
    ).to(device)
    m2 = UebungstypClassifier(
        input_dim=5 + args.hidden,
        hidden_dims=tuple(cls_dims),
        dropout=args.lstm_dropout,
    ).to(device)
    m3 = KatalogMatcher(
        embed_dim=args.katalog_embed,
        hidden_dim=args.katalog_hidden,
        mlp_blocks=args.katalog_mlp_blocks,
        dropout=args.katalog_dropout,
    ).to(device)
    katalog_matrix = get_katalog_matrix().to(device)

    n_params = sum(p.numel() for m in (m1, m2, m3) for p in m.parameters())
    print(f"  → Parameter (m1+m2+m3): {n_params:,}")

    # Alle Parameter zusammen optimieren
    all_params = (list(m1.parameters()) +
                  list(m2.parameters()) +
                  list(m3.parameters()))
    optimizer  = torch.optim.Adam(all_params, lr=args.lr)
    scheduler  = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)

    tb_writer = None
    tb_abs = None
    if not args.no_tensorboard:
        run_id = args.run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
        tb_root = os.path.join(args.log_dir, run_id)
        os.makedirs(tb_root, exist_ok=True)
        tb_writer = SummaryWriter(log_dir=tb_root)
        tb_abs = os.path.abspath(tb_root)
        print(f"TensorBoard logdir: {tb_root}")
        print(f"  → {tensorboard_launch_cmd(args.log_dir)}")
        # Sofort erste Events schreiben (sonst leeres runs/ bis Epoche 1 fertig / bei Abbruch keine Datei)
        tb_writer.add_text(
            "hparams/overview",
            f"data={args.data}, epochs={args.epochs}, batch={args.batch_size}, lr={args.lr}, "
            f"seq_len={args.seq_len}, arch={json.dumps(arch)}, device={device}, "
            f"tensorboard_scalar_step=batch_index (cumul. Minibatches)",
            0,
        )
        tb_writer.add_scalar("meta/training_started", 1.0, 0)
        tb_writer.flush()

    if use_amp:
        scaler = torch.amp.GradScaler("cuda")
    else:
        scaler = None

    def fwd_ctx():
        return torch.amp.autocast("cuda") if use_amp else nullcontext()

    # Loss Functions
    mse  = nn.MSELoss()          # Modul 1: Fehlerraten vorhersagen
    ce   = nn.CrossEntropyLoss() # Modul 2: Übungstyp-Klassifikation
    ce_k = nn.CrossEntropyLoss() # Modul 3: Katalog-Auswahl

    # ─── EWC (Elastic Weight Consolidation) ───────────────────────────────────
    class EWC:
        """Schützt wichtige Gewichte gegen Catastrophic Forgetting"""
        def __init__(self, models, dataloader, device, lambda_ewc=0.4):
            self.lambda_ewc = lambda_ewc
            self.device     = device
            self.fisher     = {}
            self.params_old = {}
            self._compute_fisher(models, dataloader)

        def _compute_fisher(self, models, dataloader):
            all_params = {}
            for name, m in models.items():
                for pname, p in m.named_parameters():
                    all_params[f"{name}.{pname}"] = p

            # Fisher Information: Gradient² über Trainingsdaten
            for key, p in all_params.items():
                self.fisher[key] = torch.zeros_like(p)

            for x, yf, yk, ykat in dataloader:
                x, yf = x.to(device), yf.to(device)
                yk, ykat = yk.to(device), ykat.to(device)
                vorhersage, hidden = models["m1"](x)
                klasse_logits      = models["m2"](vorhersage, hidden)
                kat_scores         = models["m3"](vorhersage, klasse_logits, katalog_matrix)
                loss = (mse(vorhersage, yf) +
                        ce(klasse_logits, yk) +
                        ce_k(kat_scores, ykat))
                loss.backward()
                for key, p in all_params.items():
                    if p.grad is not None:
                        self.fisher[key] += p.grad.data ** 2

            n = len(dataloader)
            for key in self.fisher:
                self.fisher[key] /= n
                self.params_old[key] = all_params[key].data.clone()

        def penalty(self, models):
            loss = 0.0
            all_params = {}
            for name, m in models.items():
                for pname, p in m.named_parameters():
                    all_params[f"{name}.{pname}"] = p

            for key, p in all_params.items():
                if key in self.fisher:
                    loss += (self.fisher[key] * (p - self.params_old[key]) ** 2).sum()
            return self.lambda_ewc * loss

    # ─── Training Loop ────────────────────────────────────────────────────────
    history = {"train_loss":[], "test_loss":[], "test_acc_klasse":[], "test_acc_katalog":[]}
    models_dict = {"m1": m1, "m2": m2, "m3": m3}
    ewc = None
    best_test_loss = float("inf")

    print("\n=== TRAINING STARTET ===")
    # TensorBoard-Schritte auf kumulierter Minibatch-Nummer (= feine Kurven, nicht nur Epochen)
    tb_global_step = 0
    for epoch in range(1, args.epochs + 1):

        # ── TRAIN ──────────────────────────────────────────────────────────────
        m1.train(); m2.train(); m3.train()
        train_loss = 0.0

        for x, yf, yk, ykat in dl_train:
            x, yf = x.to(device, non_blocking=True), yf.to(device, non_blocking=True)
            yk, ykat = yk.to(device, non_blocking=True), ykat.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with fwd_ctx():
                vorhersage, hidden = m1(x)
                klasse_logits      = m2(vorhersage, hidden)
                kat_scores         = m3(vorhersage, klasse_logits, katalog_matrix)

                loss_lstm      = mse(vorhersage, yf)        # LSTM Vorhersage
                loss_klasse    = ce(klasse_logits, yk)       # Übungstyp
                loss_katalog   = ce_k(kat_scores, ykat)      # Katalog-Auswahl

                loss = loss_lstm + loss_klasse + 0.8 * loss_katalog

                # EWC Penalty nach Epoche 20
                if ewc is not None:
                    loss = loss + ewc.penalty(models_dict)

            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(all_params, max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                nn.utils.clip_grad_norm_(all_params, max_norm=1.0)
                optimizer.step()
            train_loss += loss.item()

            if tb_writer is not None and not args.no_tb_batch_scalars:
                tb_writer.add_scalar("loss/train_batch", loss.detach().item(), tb_global_step)
                tb_global_step += 1

        train_loss /= len(dl_train)

        # EWC initialisieren nach Epoche 20
        if epoch == 20 and ewc is None:
            print("  → EWC initialisiert (Catastrophic Forgetting Schutz aktiv)")
            ewc = EWC(models_dict, dl_train, device)

        # ── TEST ───────────────────────────────────────────────────────────────
        m1.eval(); m2.eval(); m3.eval()
        test_loss   = 0.0
        correct_k   = 0
        correct_kat = 0
        total       = 0

        with torch.no_grad():
            for x, yf, yk, ykat in dl_test:
                x, yf = x.to(device, non_blocking=True), yf.to(device, non_blocking=True)
                yk, ykat = yk.to(device, non_blocking=True), ykat.to(device, non_blocking=True)

                with fwd_ctx():
                    vorhersage, hidden = m1(x)
                    klasse_logits      = m2(vorhersage, hidden)
                    kat_scores         = m3(vorhersage, klasse_logits, katalog_matrix)

                    loss = (mse(vorhersage, yf) +
                            ce(klasse_logits, yk) +
                            ce_k(kat_scores, ykat))
                    test_loss += loss.item()

                # FP32 für Metriken: unter CUDA+AMP können FP16-Logits bei argmax/== keine Treffer zeigen → TB bleibt fälschlich bei 0
                lk = klasse_logits.float()
                zk = kat_scores.float()
                yk_i = yk.long().reshape(-1)
                ykat_i = ykat.long().reshape(-1)
                correct_k += (lk.argmax(dim=-1) == yk_i).sum().item()
                correct_kat += (zk.argmax(dim=-1) == ykat_i).sum().item()
                total += yk_i.numel()

        test_loss /= len(dl_test)
        acc_k   = correct_k   / total * 100 if total > 0 else 0
        acc_kat = correct_kat / total * 100 if total > 0 else 0

        scheduler.step(test_loss)

        history["train_loss"].append(train_loss)
        history["test_loss"].append(test_loss)
        history["test_acc_klasse"].append(acc_k)
        history["test_acc_katalog"].append(acc_kat)

        # tb_step = kumulierte Minibatches (kurven haben Epochen×batches Punkte); mit --no_tb_batch_scalars: Epoche als X-Achse
        if tb_writer is not None:
            tb_step = tb_global_step if not args.no_tb_batch_scalars else epoch
            tb_writer.add_scalar("loss/train_total", train_loss, tb_step)
            tb_writer.add_scalar("loss/test_total", test_loss, tb_step)
            tb_writer.add_scalar("accuracy/test_uebungstyp", acc_k / 100.0, tb_step)
            tb_writer.add_scalar("accuracy/test_katalog", acc_kat / 100.0, tb_step)
            # Gleiche Kennzahlen mit Epoche als X-Achse + Prozent 0–100 (TensorBoard-Verwechslungen vermeiden)
            tb_writer.add_scalar("epoch/acc_test_uebungstyp_pct", acc_k, epoch)
            tb_writer.add_scalar("epoch/acc_test_katalog_pct", acc_kat, epoch)
            lr = optimizer.param_groups[0]["lr"]
            tb_writer.add_scalar("optim/lr", lr, tb_step)
            tb_writer.flush()

        # Checkpoint speichern wenn bestes Modell
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            os.makedirs(args.save_dir, exist_ok=True)
            torch.save({
                "epoch": epoch,
                "arch": arch,
                "data_path": os.path.abspath(str(args.data)),
                "m1": m1.state_dict(),
                "m2": m2.state_dict(),
                "m3": m3.state_dict(),
                "test_loss": test_loss,
                "acc_klasse": acc_k,
                "acc_katalog": acc_kat,
            }, f"{args.save_dir}/best_model.pt")

        # Log alle 10 Epochen
        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:>4}/{args.epochs} | "
                  f"Train Loss: {train_loss:.4f} | "
                  f"Test Loss: {test_loss:.4f} | "
                  f"Acc Klasse: {acc_k:.1f}% | "
                  f"Acc Katalog: {acc_kat:.1f}%")

    # Verlauf speichern
    with open(f"{args.save_dir}/history.json", "w") as f:
        json.dump(history, f, indent=2)

    had_tensorboard = tb_writer is not None
    if had_tensorboard:
        tb_writer.close()

    print(f"\n=== FERTIG ===")
    print(f"Bestes Modell: Test Loss {best_test_loss:.4f}")
    print(f"Gespeichert in: {args.save_dir}/best_model.pt")
    if had_tensorboard and tb_abs is not None:
        print(
            f"\nTensorBoard — nur dieser Lauf (empfohlen, wenn mehrere Runs in {args.log_dir}):"
            f"\n  {tensorboard_single_run_cmd(tb_abs)}"
        )
        print(
            f"\nTensorBoard — alle Läufe unter `{args.log_dir}` im gleichen Graphen:"
            f"\n  {tensorboard_launch_cmd(args.log_dir)}"
            f"\n(Links in der Runs-Sidebar kannst du einzelne Läufe abwählen.)"
            f"\nTensorBoard immer auf der Maschine starten, auf der diese Ordner liegen;"
            f"\nauf dem HPC: Login-Node, ssh -L 6006:127.0.0.1:6006 ..., dann dort tensorboard."
        )


if __name__ == "__main__":
    main()
