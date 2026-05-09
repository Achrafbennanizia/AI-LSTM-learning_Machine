# GuitarAI — Forschung (Architekturvergleich & Forgetting)

## Ausgabe-Verzeichnisse (`research_outputs/`)

Alles Landet **neben** dem Haupttraining (`runs/`, `checkpoints/`), damit Produktion und Forschung getrennt bleiben.

| Pfad | Inhalt |
|------|--------|
| `research_outputs/tensorboard/compare/` | TB-Events je **Architekturvergleich**-Lauf (`train_compare.py`) |
| `research_outputs/tensorboard/forgetting/` | TB-Events **Forgetting-Benchmark** |
| `research_outputs/artifacts/compare/` | `comparison_*.json`, `run_manifest_*.json` |
| `research_outputs/artifacts/forgetting/` | `forgetting_*.json`, Manifeste |
| `research_outputs/exports/` | **CSV-Tabellen** (`compare_summary_*.csv`, `forgetting_summary_*.csv`) für Excel / R / Papers |

Überschreiben der Basis:

- **`--research_root pfad`** — Elternordner statt `<Repo>/research_outputs`
- **`GUITARAI_RESEARCH_ROOT`** — globale Umgebungsvariable (höchste Priorität in `get_research_root`)

Detaillierte Tipps: **`research_outputs/README.txt`**.

## Befehle (vom Projektroot)

```bash
cd /pfad/zu/AI-Gui

# Architekturvergleich (MLP, LSTM, GRU, DQN optional)
python research/train_compare.py --pretrain_epochs 50 --skip_dqn

# Forgetting-Benchmark
python research/benchmark_forgetting.py --n_tasks 8
```

**TensorBoard** (volle Skalen-Länge wie bei `train.py`):

```bash
tensorboard --logdir research_outputs/tensorboard/compare \
  --samples_per_plugin=scalars=10000000

tensorboard --logdir research_outputs/tensorboard/forgetting \
  --samples_per_plugin=scalars=10000000
```

Die Skripte **drucken** am Ende die exakten `--logdir`-Zeilen für *einen* Lauf oder für alle Läufe unter dem jeweiligen Tensorboard-Root.

## Was in TensorBoard landet

- **Vergleich:** `run/summary`, `run/hparams`, `data/*`, `split/*`, pro Architektur Unterordner mit `pretrain/`, `incremental/`, `phase/`, plus **`compare/*`** auf dem Haupt-Writer.
- **Forgetting:** `run/summary`, Kurven `forgetting/*`, `acc_task1/*`, **`summary/*`** Aggregat am Ende.
- **CSV:** flache Zeilen pro Architektur (inkl. JSON-serialisierte Listen in Zellen wo nötig).

## Module

- `research/io_layout.py` — Pfade, TB-Kommandos, Export-Helfer
- `research/incremental_engine.py` — EWC, Replay, `IncrementalTrainer`
- `models/gru_module.py`, `mlp_module.py`, `dqn_module.py` — weitere Köpfe / RL
