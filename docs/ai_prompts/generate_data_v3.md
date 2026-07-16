# Prompts — `data/generate_data_v3.py`

**Tool:** Cursor Composer  
**Target file:** `data/generate_data_v3.py` (full file, multiple iterations)

---

## Prompt 1 — Initial generator architecture

> Build a research-grade synthetic guitar-learner data generator (v3) that encodes motor-learning laws: Fitts & Posner stages, power-law practice (Newell & Rosenbloom), Ebbinghaus forgetting with motor floor, Bjork spacing, Vygotsky ZPD for exercise difficulty, Ericsson deliberate practice (weak-skill targeting), Singley–Anderson transfer vectors per catalog exercise. Output 17 session features + labels (`empfehlung_id`, `uebungstyp_label`). Five user archetypes with individual learning rate, motivation, ceilings, problem chord. Three layers: latent competence, session events, correlated observation noise (left hand / right hand / motor coordination).

## Prompt 2 — Correlated error noise

> Add `observe_errors()` so grip, pressure, muting share a left-hand latent factor; timing and technique share a right-hand factor; all five errors share session-wide motor coordination noise. Noise scale must depend on Fitts stage and event multiplier.

## Prompt 3 — Event system

> Implement weighted session events (discovery, flow, plateau breakthrough, bad day, overreach, regression after pause, muscle fatigue, cold start) that modify learning gain, stability bonus, and observation noise. Probabilities depend on stage, session quality, plateau streak, gap days, repetition count.

## Prompt 4 — Derived learning signals

> Derive `v_lern`, `k_konsistenz`, `d_limit`, `p_plateau`, `r_repeat` from actual session history (rolling variance, ceiling hits, plateau streak) — not random columns.

## Prompt 5 — Cohort export

> Generate 400 users (5 types × 80), 50 sessions each, CSV + JSON export, reproducible seeds, debug columns prefixed with `_` excluded from model CSV.

---

**Human edits after AI:** Skill dataclass fix, CSV field lists, streaming shard options, typo fixes in comments, calibration constants for ~50-session learning curves.
