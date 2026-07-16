# AI-assisted code — provenance (report appendix)

This document records where generative AI was used in the GuitarAI codebase, which tool/version applied, and the prompts that shaped the code. All AI-generated parts were **reviewed and edited by the author** before use.

## Tool

| Field | Value |
|-------|--------|
| **IDE / agent** | [Cursor](https://cursor.com) — Composer agent |
| **Session period** | May–July 2026 |
| **Model family** | Claude (via Cursor; exact snapshot varies by session) |
| **Human role** | Problem definition, literature mapping, review, integration, HPC runs |

## Coverage policy

| File / area | AI share | Notes |
|-------------|----------|--------|
| `data/generate_data_v3.py` | **~100%** (draft + iterations) | Full attribution in file header; see `docs/ai_prompts/generate_data_v3.md` |
| `research/incremental_engine.py` | **~25%** | EWC Fisher penalty + replay batch merge (hardest logic) |
| `models/katalog_matcher.py` | **~25%** | Dual-encoder query–catalog scoring |
| `training_common.py` | **~25%** | Session windowing dataset + multi-head loss |
| Other modules (`rnn_module`, `lstm`, `gru`, `classifier`, research scripts) | **Manual / light AI** | Standard PyTorch patterns; no full-file AI draft |

## How to read in-code markers

Blocks tagged with a comment like:

```text
# [AI-assisted] tool=Cursor Composer | prompt=docs/ai_prompts/<file>.md#<section>
```

mark sections where the initial draft came from AI. Untagged code is primarily human-written or adapted from PyTorch / project conventions.

## Prompt log (summary)

Detailed prompts are in `docs/ai_prompts/`. Representative requests:

1. **Synthetic data generator** — Implement literature-grounded learner simulation (Fitts, Ebbinghaus, spacing, ZPD, deliberate practice, correlated hand noise, 17 features, 5 archetypes).
2. **Incremental training** — EWC diagonal Fisher + experience replay buffer interleaved with new-task batches.
3. **Catalog matcher** — Embed predicted errors + exercise-type logits; score 15 catalog rows via dot product.
4. **Research pipeline** — Fair architecture comparison (RNN/LSTM/GRU), TensorBoard logging, forgetting benchmark.

## Experiment artifacts

Aggregated metrics (no host or account metadata) may be stored under `research_outputs/artifacts/`. SLURM logs stay outside the repository.

## Declaration (for report)

> Parts of `generate_data_v3.py` were drafted with Cursor Composer from natural-language specifications grounded in motor-learning literature. Approximately 25% of the training and matching stack (EWC, replay, catalog matcher, dataset windowing) received AI-assisted first drafts. The author verified behaviour, ran experiments on a Linux HPC cluster, and revised all code before submission.
