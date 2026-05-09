# GuitarAI

Deep-Learning-Projekt zur **Auswertung von Übungssessions am E-Gitarrentrainer Leo**: Aus einer kurzen **Historie aufeinanderfolgender Sessions** werden (1) die erwarteten **Fehlerraten** der nächsten Session vorhergesagt, (2) der **Übungstyp** klassifiziert und (3) eine **Konkretübung aus dem Übungs-Katalog** ausgewählt.

Das Training läuft **lokal** (CPU/GPU) und auf einem **Linux-HPC unter Slurm** (z. B. HiPerGOS/OS).

---

## Architektur (Überblick)

| Komponente | Rolle |
|------------|--------|
| **VerhaltensLSTM** (`models/lstm_module.py`) | Sequenzmodell über die letzten `seq_len` Sessions (aktuell Standard: **5**) mit **17** Merkmalen pro Session; gibt Fehlerratenvorschau (**σ**) und den letzten versteckten Zustand aus. |
| **UebungstypClassifier** (`models/classifier_module.py`) | MLP auf `[Vorhersage (5), Hidden]` → **5 Übungstypen**. |
| **KatalogMatcher** (`models/katalog_matcher.py`) | Zwei Encoder (Query aus Fehlerratenvorschage + Übungstyp-Wahrscheinlichkeiten vs. feste **Katalogmerkmale**); Auswahl unter **15** Katalog-Einträgen über Ähnlichkeit / Scores. |

Verluste: MSE auf Fehlerratenziele, Cross-Entropy für Übungstyp und Katalog. Optional wirkt **EWC** (Elastic Weight Consolidation) gegen katastrophisches Vergessen mit.

Checkpoints unter `checkpoints/best_model.pt` speichern u. a. `arch`, Gewichte (`m1`–`m3`) und den **Pfad zur Trainings-CSV** (`data_path`), damit `evaluate.py` dieselbe Datenbasis referenzieren kann.

---

## Daten

- Erwartete Spalten u. a.: `nutzer_id`, `session`, die **17** Merkmale in `FEATURE_COLS` (definiert in `train.py`), `uebungstyp_label`, `empfehlung_id` (Katalog-ID wie in `katalog_matcher.KATALOG_IDS`).
- **Große CSV/JSON-Dateien** werden per `.gitignore` nicht versioniert. Stattdessen liegt ein **Generator-Skript** in `data/leo_story_generator.py` — damit lässt sich z. B. ein mehrnutzeriges Set wie `full_dataset_20k.csv` erzeugen (siehe Kommentare in `job.slurm.template`).
- Datenpfad:
  - Umgebungsvariable **`GUITARAI_DATA`** (Datei oder Verzeichnis mit `leo_50_sessions.csv`) bzw. **`GUITARAI_DATA_DIR`**, sonst Fallback siehe `paths.py` und `./data/`.

---

## Lokales Setup

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Training (Standardeinstellungen siehe Hilfe):

```bash
python train.py --help
python train.py --data pfad/zur.csv
```

Evaluation (lädt standardmäßig `checkpoints/best_model.pt`):

```bash
python evaluate.py
```

**TensorBoard:** Events liegen unter `runs/<Zeitstempel>/`. Hinweise zu `--samples_per_plugin` und IPv4-Binding stehen in `runs/README.txt`.

---

## Hochleistungsrechner (Slurm)

1. `.env.example` nach **`.env`** kopieren und **partition, account, Mail, GPU-`gres`**, Limits etc. eintragen (`.env` ist gitignored).
2. Job-Script erzeugen:

   ```bash
   python3 scripts/render_job_slurm.py
   ```

   Ausgabe: **`job.slurm`** im Projektroot (ebenfalls gitignored, falls persönlich).
3. Auf dem Login-Node: Repo/venv bereitstellen, ggf. `full_dataset_20k.csv` generieren, dann:

   ```bash
   sbatch job.slurm
   ```

Das Template enthält konkrete `train.py`-Argumente (u. a. Epochen, LR, Hidden-Größen); siehe **`job.slurm.template`**.

---

## Projektstruktur (Kern)

```
train.py           # Training, Dataset, Checkpointing, TensorBoard
evaluate.py        # Confusion Matrix, Reports, kleine Demo
paths.py           # Auflösung CSV-Pfad über Env/Defaults
models/            # LSTM, Classifier, Katalogmatcher
scripts/render_job_slurm.py   # `.env` + `job.slurm.template` → `job.slurm`
data/leo_story_generator.py   # Synthetische / skalierbare Datensatz-Erzeugung
```

---

## Referenz-Konventionen

- **Architektur-Defaults** in `train.py` (u. a. `hidden=64`, `lstm_layers=1`, `batch_size=32`, `lr=5e-4`) sind auf ein kompaktes Modell (**~116k** trainierbare Parameter für `m1`+`m2`+`m3` bei Default-Katalogkopf; exakte Zahl wird beim Trainingsstart ausgegeben).
- Schlüsselinvariante: Änderungen an **`FEATURE_COLS`** / Ein-/Ausgabedimensionen erfordern abgestimmte Anpassungen in Modellen und ggf. neues Training; alte Checkpoints sind dann nicht ohne Weiteres kompatibel.
