# Prompts — `training_common.py` (AI-assisted sections only)

**Tool:** Cursor Composer  
**Sections:** `SessionDataset`, `make_loss_fn` (~25% of file)

---

## Prompt — Shared research training utilities

> Create shared helpers for architecture comparison: FEATURE_COLS (17 inputs), SessionDataset that groups by user, uses last seq_len=5 sessions as input and next session's first 5 error rates as M1 target, plus exercise-type and catalog labels. 60/20/20 user split (pretrain / incremental / test). Combined loss: 0.5×MSE on errors + CE on exercise type + 0.8×CE on catalog. build_models switches M1 between RNN/LSTM/GRU only; M2/M3 identical.

---

**Human edits:** `evaluate`, `tb_cmd`, import paths, RNN replacing CNN.
