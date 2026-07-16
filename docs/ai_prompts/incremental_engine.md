# Prompts — `research/incremental_engine.py` (AI-assisted sections only)

**Tool:** Cursor Composer  
**Sections:** `EWC` class, `incremental_update` replay + penalty merge (~25% of file)

---

## Prompt — EWC + replay for continual learning

> Implement incremental training for a three-module PyTorch pipeline: pretrain loop, then incremental updates per user chunk. After each task, estimate diagonal Fisher information from squared gradients and store parameter snapshots. During incremental training add EWC penalty λ=400. Keep a replay buffer (capacity 2000); each batch should concatenate ~1/3 replay samples with new-task data. Clip gradients at norm 1.0. Update Fisher after pretrain and after each incremental task.

---

**Human edits:** `use_ewc` flag, return mean task loss, replay add during pretrain.
