# Forschung — Architekturvergleich & Forgetting

## Ausgabe

| Pfad | Inhalt |
|------|--------|
| `research_outputs/tensorboard/compare/` | Architekturvergleich |
| `research_outputs/tensorboard/forgetting/` | Task-1-Retention (`acc_task/`) |
| `research_outputs/artifacts/compare/` | `comparison_*.json` |
| `research_outputs/artifacts/forgetting/` | `task1_*.json` |

## Befehle

```bash
python research/train_compare.py --data data/leo_sessions_v3.csv
python research/benchmark_forgetting.py --data data/leo_sessions_v3.csv
```

## TensorBoard

```bash
tensorboard --logdir research_outputs/tensorboard/compare --samples_per_plugin=scalars=10000000
tensorboard --logdir research_outputs/tensorboard/forgetting --samples_per_plugin=scalars=10000000
```

**Daten** (`generate_data_v3.py`):
- 20.000 Sessions, 400 Nutzer (5 Typen × 80 × 50 Sessions)
- Split: 60/20/20 (Pretrain / Inkrementell / Test)

**Architekturvergleich** — TensorBoard pro Modell:
- `incremental/acc_katalog` (step 0 = nach Vortraining; Abfall = Vergessen)
- `incremental/loss`
- `incremental/training_loss`

**Task-1-Retention** (`benchmark_forgetting.py`):
- `acc_task/<arch>` (step 0 = Baseline; Abfall = Vergessen auf Task 1)

Zusammenfassungen in `research_outputs/artifacts/*.json`.

**AI-Provenance:** `docs/AI_CODE_ATTRIBUTION.md`

## Dateien

- `train_compare.py` — RNN, RNN+EWC, LSTM+EWC, GRU+EWC
- `benchmark_forgetting.py` — Acc auf Task 1 nach jedem neuen Task
- `incremental_engine.py` — EWC und Experience Replay
- `io_layout.py` — Ausgabe-Pfade
