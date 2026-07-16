# GuitarAI

Deep-Learning-Projekt zur **Auswertung von Übungssessions am E-Gitarrentrainer Leo**: Aus einer kurzen **Historie aufeinanderfolgender Sessions** werden (1) die erwarteten **Fehlerraten** der nächsten Session vorhergesagt, (2) der **Übungstyp** klassifiziert und (3) eine **Konkretübung aus dem Übungs-Katalog** ausgewählt.

Experimente laufen **lokal** (CPU/GPU) und auf einem **Linux-HPC unter Slurm**.

---

## Architektur (Überblick)

| Komponente | Rolle |
|------------|--------|
| **VerhaltensRNN** (`models/rnn_module.py`) | **M1**-Variante: Vanilla RNN + MLP-Kopf |
| **VerhaltensLSTM / GRU** (`models/lstm_module.py`, `gru_module.py`) | **M1**-Varianten mit LSTM/GRU-Gedächtnis |
| **UebungstypClassifier** (`models/classifier_module.py`) | **M2** — MLP auf `[Vorhersage (5), Hidden]` → **5 Übungstypen** |
| **KatalogMatcher** (`models/katalog_matcher.py`) | **M3** — Auswahl unter **15** Katalog-Einträgen |

Verluste: MSE auf Fehlerratenziele, Cross-Entropy für Übungstyp und Katalog. **EWC** und **Experience Replay** gegen katastrophales Vergessen (`research/incremental_engine.py`).

---

## Daten

- **17 Features** in `FEATURE_COLS` (`training_common.py`), plus `uebungstyp_label`, `empfehlung_id`
- Generator: `data/generate_data_v3.py` → z. B. `data/leo_sessions_v3.csv`
- Split für Forschung: 60/20/20 (Pretrain / Inkrementell / Test)

```bash
python data/generate_data_v3.py --users-per-type 80,80,80,80,80 --sessions 50
```

---

## Lokales Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

```bash
python research/train_compare.py --data data/leo_sessions_v3.csv
python research/benchmark_forgetting.py --data data/leo_sessions_v3.csv
```

TensorBoard: `research_outputs/tensorboard/` — siehe `research_outputs/README.md`.

**AI-assisted code:** `docs/AI_CODE_ATTRIBUTION.md` (prompts under `docs/ai_prompts/`).

---

## Hochleistungsrechner (Slurm)

1. `.env.example` → `.env` (Partition, Account, GPU-`gres`, Mail)
2. `python3 scripts/render_job_slurm.py` → `job_research.slurm`
3. Auf dem Login-Node: `sbatch job_research.slurm`

---

## Projektstruktur

```
training_common.py    # FEATURE_COLS, Dataset, Modelle, Loss, Eval
models/               # RNN, LSTM, GRU, Classifier, Katalogmatcher
data/generate_data_v3.py
research/             # Architekturvergleich, Forgetting-Benchmark
  incremental_engine.py
  train_compare.py
  benchmark_forgetting.py
scripts/render_job_slurm.py
job_research.slurm.template
research_outputs/     # (gitignored) TensorBoard + JSON-Artefakte
```

Details: `research/README_research.md`
