# Prompts — `models/katalog_matcher.py` (AI-assisted sections only)

**Tool:** Cursor Composer  
**Sections:** `_encoder`, `KatalogMatcher.forward` (~25% of file)

---

## Prompt — Catalog matching head (M3)

> Module 3 should score 15 fixed catalog exercises. Build query embedding from concatenated predicted error rates (5) and softmax exercise-type probabilities (5). Encode each catalog row (6 features: type, difficulty, transfer stats) with a shared MLP. Return batch × 15 scores via matrix multiply query_emb @ catalog_emb.T for cross-entropy training against `empfehlung_id`.

---

**Human edits:** Catalog table data, hyperparameters (embed_dim=128, 2 MLP blocks), integration in `training_common.make_loss_fn`.
