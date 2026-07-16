"""
GuitarAI — Kognitiver Datengenerator v3 (Forschungsversion)
============================================================

AI-assisted code (full file — report provenance)
------------------------------------------------
  Tool:     Cursor IDE, Composer agent
  Period:   May–July 2026
  Prompts:  docs/ai_prompts/generate_data_v3.md
  Index:    docs/AI_CODE_ATTRIBUTION.md
  Review:   Human-edited after generation (literature mapping, calibration, export)

Implementiert ALLE relevanten kognitionswissenschaftlichen Gesetze für
musikalisches Motorlernen. Basiert auf Literatur:

  • Fitts & Posner (1967)        — 3 Phasen des motorischen Lernens
                                   (cognitive → associative → autonomous)
  • Newell & Rosenbloom (1981)   — Potenzgesetz des Lernens
  • Ebbinghaus (1885)            — Vergessenskurve, exponentieller Zerfall
  • Bjork (1994), Dunlosky (2013)— Spacing-Effekt, verteiltes Üben
  • Vygotsky                     — Zone der nächsten Entwicklung
  • Ericsson et al. (1993)       — Deliberate Practice, gezielte Schwächenarbeit
  • Singley & Anderson (1989)    — Skill Transfer
  • Schmidt & Lee                — Performance-Variabilität pro Stage
  • Music-Lit                    — Hohe Anfangsfehler, Plateau ~ Session 5,
                                   Sprünge bei Wiederholung

ARCHITEKTUR — DREI EBENEN:
  1) Latente Kompetenz       (deterministisch per Seed, Lernkurve)
  2) Events als Guide+Noise  (strukturierte Varianz: discovery, plateau, regression)
  3) Beobachtungsrauschen    (Fitts-Stage-abhängig, korreliert über Hände)

ZWEI ZEN:
  • Zufall NUR in: Nutzer-Profil (einmalig), Kontext (pro Session),
    Event-Wahl (gewichtet), Beobachtungsrauschen.
  • Lernkurve selbst ist GLATT und reproduzierbar.

17 DIMENSIONEN MIT REALISTISCHEN KORRELATIONEN:
  e_griff   ←─┐ (linke Hand, Hauptgriff)
  e_druck   ←─┼── shared "left_hand_noise" (Korrelation 0.5–0.6)
  e_muting  ←─┘
  e_timing  ←─┐ (rechte Hand, Rhythmus)
  e_technik ←─┴── shared "right_hand_noise" (Korrelation 0.3–0.4)
  + session-weite "motor_coordination_noise" auf alle 5

  v_lern        = abgeleitet aus tatsächlicher Kompetenzänderung
  k_konsistenz  = invers zu rollender Varianz der Fehlerraten
  d_limit       = Häufigkeit von Ceiling-Hits (echtes Plateau-Signal)
  p_plateau     = abgeleitet aus v_lern < threshold über N Sessions
  r_repeat      = Wiederholung derselben Übung (echter Counter)
  n_sessions    = log-normiert
  t_session     = Sitzungsqualität (Tageszeit + Wochentag + Motivation - Müdigkeit)
  s_level       = Fitts-Phase (0-0.20=cog, 0.20-0.55=assoc, 0.55+=auto)
  pause_norm    = gap_days / 30
  akkord_fokus  = Problem-Akkord-Schwierigkeit (konstant pro Nutzer)
  wochentag_norm, tageszeit_norm = Kontext

Usage:
  python generate_data_v3.py --users-per-type 80,80,80,80,80 --sessions 50
  python generate_data_v3.py --single --typ 3 --sessions 100
"""
from __future__ import annotations
import argparse, csv, json, math, random
from pathlib import Path
from dataclasses import dataclass
from collections import deque, Counter

# ═══════════════════════════════════════════════════════════════════════════════
# KATALOG (15 Übungen mit Transfer-Vektor)
# Transfer: [grip, timing, pressure, technique, muting]
# ═══════════════════════════════════════════════════════════════════════════════
KATALOG = [
    {"id":"em_einzeln_40bpm",         "typ":0,"diff":0.10,"tr":[0.80,0.20,0.55,0.30,0.20]},
    {"id":"em_einzeln_metronom",      "typ":0,"diff":0.14,"tr":[0.70,0.55,0.50,0.30,0.22]},
    {"id":"am_einzeln_40bpm",         "typ":0,"diff":0.18,"tr":[0.85,0.20,0.58,0.35,0.28]},
    {"id":"em_am_35bpm",              "typ":1,"diff":0.22,"tr":[0.60,0.70,0.42,0.32,0.38]},
    {"id":"em_am_50bpm",              "typ":1,"diff":0.28,"tr":[0.55,0.80,0.42,0.32,0.38]},
    {"id":"horse_no_name_50bpm",      "typ":3,"diff":0.22,"tr":[0.40,0.60,0.38,0.28,0.48]},
    {"id":"eleanor_rigby_50bpm",      "typ":3,"diff":0.27,"tr":[0.35,0.75,0.35,0.32,0.48]},
    {"id":"eleanor_rigby_60bpm",      "typ":3,"diff":0.30,"tr":[0.35,0.80,0.35,0.32,0.48]},
    {"id":"eleanor_rigby_70bpm",      "typ":3,"diff":0.34,"tr":[0.40,0.85,0.38,0.38,0.48]},
    {"id":"em_rhythmus_metronom",     "typ":2,"diff":0.16,"tr":[0.22,0.90,0.32,0.22,0.38]},
    {"id":"knockin_strophe_emcg",     "typ":3,"diff":0.40,"tr":[0.65,0.65,0.48,0.48,0.58]},
    {"id":"knockin_voll_60bpm",       "typ":3,"diff":0.44,"tr":[0.70,0.60,0.48,0.52,0.62]},
    {"id":"g_einzeln_technik",        "typ":4,"diff":0.30,"tr":[0.90,0.12,0.62,0.90,0.42]},
    {"id":"stand_by_me_60bpm",        "typ":3,"diff":0.42,"tr":[0.60,0.70,0.48,0.52,0.60]},
    {"id":"wish_you_were_here_intro", "typ":3,"diff":0.50,"tr":[0.75,0.55,0.52,0.62,0.60]},
]
KAT  = {k["id"]: k for k in KATALOG}
KIDS = [k["id"] for k in KATALOG]
N    = 5  # grip, timing, pressure, technique, muting

# ═══════════════════════════════════════════════════════════════════════════════
# NUTZERTYPEN — Profile mit individuellen Schwächen
# ═══════════════════════════════════════════════════════════════════════════════
TYPEN = {
    0:{"name":"Akkord-Fokus",     "ic":[0.48,0.36,0.52,0.52,0.58], "weak":1},
    1:{"name":"Rhythmus-Kämpfer", "ic":[0.33,0.57,0.48,0.50,0.58], "weak":0},
    2:{"name":"Timing-Profi",     "ic":[0.48,0.67,0.48,0.33,0.58], "weak":3},
    3:{"name":"Ausgewogen",       "ic":[0.45,0.45,0.50,0.43,0.58], "weak":None},
    4:{"name":"Technik-Fokus",    "ic":[0.57,0.45,0.38,0.48,0.58], "weak":2},
}
AKKORDE  = ["G","F","D","Bm","keiner"]
AK_LEVEL = {"G":0.22,"F":0.38,"D":0.52,"Bm":0.72,"keiner":0.05}
AK_W     = [0.30,0.25,0.20,0.15,0.10]
GAPS     = [1,2,3,7,14,30]
GAP_W    = [0.40,0.26,0.18,0.10,0.04,0.02]


def cl(v, lo=0.0, hi=1.0):
    return max(lo, min(hi, float(v)))


# ═══════════════════════════════════════════════════════════════════════════════
# FITTS & POSNER STAGE-MODELL
# ═══════════════════════════════════════════════════════════════════════════════
def fitts_stage(s_level: float, avg_skill: float) -> str:
    """Bestimmt aktuelle motorische Lernphase nach Fitts & Posner (1967).

    cognitive  : hohe Variabilität, schnelle Anfangsgewinne, viel bewusste Kontrolle
    associative: sinkende Variabilität, langsamere Gewinne, teils automatisch
    autonomous : sehr niedrige Variabilität, kaum Gewinne, vollautomatisch
    """
    composite = (s_level * 0.6) + (avg_skill * 0.4)
    if composite < 0.30:  return "cognitive"
    if composite < 0.65:  return "associative"
    return "autonomous"


def stage_noise(stage: str) -> float:
    """Performance-Variabilität sinkt mit fortschreitender Lernphase.
    Schmidt & Lee (Motor Control & Learning), gut belegt durch Music-Forschung.
    """
    return {"cognitive":   0.045,
            "associative": 0.022,
            "autonomous":  0.010}[stage]


def stage_learning_modifier(stage: str) -> float:
    """Lerngewinn pro Session ist phasenabhängig.
    Cognitive: schnelle Gewinne (steile Anfangsphase der S-Kurve)
    Associative: langsamer (Bend in der Kurve)
    Autonomous: minimal (Plateau, nur noch durch deliberate practice)
    """
    return {"cognitive":   1.4,
            "associative": 1.0,
            "autonomous":  0.55}[stage]


# ═══════════════════════════════════════════════════════════════════════════════
# SKILL DATAKLASSE
# ═══════════════════════════════════════════════════════════════════════════════
@dataclass
class Skill:
    lvl: float
    ceil: float
    stab: float
    n_pract: int = 0


# ═══════════════════════════════════════════════════════════════════════════════
# KOGNITIVE GESETZE
# ═══════════════════════════════════════════════════════════════════════════════
def forget(sk: Skill, days: int) -> Skill:
    """Ebbinghaus für Motorgedächtnis.
    Floor: 75% (Motor-Memory bleibt fast immer)
    Stabilität sinkt NICHT durch Pause (= aktuelles SRS-Forschungskonsens)
    """
    if days <= 0: return sk
    s = sk.stab * (3.5 if sk.lvl > 0.80 else 1.8 if sk.lvl > 0.65 else 1.0)
    ret = math.exp(-days / max(s, 1.0))
    floor = sk.lvl * 0.75
    return Skill(cl(floor + (sk.lvl - floor) * ret), sk.ceil, sk.stab, sk.n_pract)


def practice(sk: Skill, tr: float, diff: float, sq: float,
             lr: float, stage: str, s_idx: int) -> Skill:
    """Potenzgesetz + Spacing-Effekt + Vygotsky Zone der nächsten Entwicklung.

    Lerngewinn = α × stage_modifier × lr × transfer × quality × diff_match × headroom
    Stabilität wächst log-spaced: häufige Wiederholungen am gleichen Tag bringen
    weniger Stabilität als verteilt → Spacing-Effekt (Bjork, 1994).
    """
    if tr < 0.05:
        return sk
    headroom = max(0.0, sk.ceil - sk.lvl)
    ideal = 0.10 + sk.lvl * 0.55                      # Vygotsky ZPD
    match = cl(1.0 - abs(diff - ideal) * 2.0, 0.05, 1.0)
    # Kalibriert für ~36 Sessions → spürbarer Lernfortschritt (Music-Lit)
    base  = 0.095 * stage_learning_modifier(stage) * lr * tr * sq * match * headroom
    gain  = cl(base, 0.0, 0.080)
    sessions_since = max(1, s_idx - sk.n_pract)
    spacing = cl(math.log(sessions_since + 1) / math.log(3), 0.4, 2.5)
    stab_gain = 2.0 * tr * sq * spacing
    return Skill(cl(sk.lvl + gain), sk.ceil,
                 min(sk.stab + stab_gain, 130.0), s_idx)


def pick_exercise(skills: list, s_lvl: float, last_ex: str | None,
                  stage: str, rng: random.Random) -> str:
    """Übungswahl: ZPD + Deliberate Practice (Schwächenfokus) + Variation.

    Ericsson: Schwäche gezielt angehen → schnellere Verbesserung.
    Aber: 100% gleiche Übung → kein Transfer → Variation einbauen.
    """
    weak = min(range(N), key=lambda i: skills[i].lvl)
    target = 0.10 + s_lvl * 0.65
    scored = []
    for k in KATALOG:
        dm = cl(1.0 - abs(k["diff"] - target) * 2.2, 0.0, 1.0)
        wt = k["tr"][weak]
        # Penalty für sofortige Wiederholung in cognitive stage (Variation für Transfer)
        rep_penalty = 0.15 if (k["id"] == last_ex and stage == "cognitive") else 0.0
        scored.append((dm * 0.55 + wt * 0.40 - rep_penalty + rng.uniform(0, 0.05),
                       k["id"]))
    scored.sort(reverse=True)
    return scored[0][1] if rng.random() < 0.78 else scored[min(1, len(scored)-1)][1]


# ═══════════════════════════════════════════════════════════════════════════════
# SITZUNGSQUALITÄT (Tageszeit, Wochentag, Motivation, Müdigkeit)
# ═══════════════════════════════════════════════════════════════════════════════
def session_quality(tz, wt, mot, fatigue, rng):
    tz_score = {0.25:0.82, 0.50:0.74, 0.75:0.92, 1.00:0.52}.get(tz, 0.70)
    we_bonus = 0.06 if wt >= 5 else 0.0
    fatigue_p = max(0, fatigue - 0.7) * 0.3
    daily = cl(mot + rng.gauss(0, 0.05))
    return cl((tz_score + we_bonus - fatigue_p) * daily, 0.35, 1.00)


# ═══════════════════════════════════════════════════════════════════════════════
# EVENTS — als Guide UND Noise im Lernprozess
# ═══════════════════════════════════════════════════════════════════════════════
"""
Events sind strukturierte Abweichungen vom Standard-Verlauf.
Sie WIRKEN auf: learning_mult, forgetting_mult, noise_mult, exercise_choice.

Wahrscheinlichkeit hängt von Phase, Streak und Kontext ab.

Music-Lit: typischer Verlauf zeigt
  - hohe Anfangsfehler (cognitive stage default)
  - Plateau ab ~ Session 5 (associative stage entry)
  - Sprünge nach Wiederholung (repetition_breakthrough)
  - Aha-Momente / discovery in cognitive stage
"""
EVENTS = {
    # Positive Events — bringen Lernfortschritt
    "discovery_moment": {
        "desc": "Aha-Moment: plötzliches Verstehen eines Bewegungsmusters",
        "learning_mult": 2.5,    # 2.5× Lerngewinn diese Session
        "noise_mult":    0.7,
        "stab_bonus":    8.0,
    },
    "flow_state": {
        "desc": "Flow-Zustand: hohe Konzentration und Engagement",
        "learning_mult": 1.6,
        "noise_mult":    0.6,
        "stab_bonus":    4.0,
    },
    "repetition_breakthrough": {
        "desc": "Plateau-Durchbruch durch deliberate practice",
        "learning_mult": 2.0,
        "noise_mult":    0.8,
        "stab_bonus":    6.0,
    },
    "consolidation_session": {
        "desc": "Festigung: alles bekannte verfestigt sich, Stabilität wächst stark",
        "learning_mult": 0.6,
        "noise_mult":    0.5,
        "stab_bonus":    10.0,
    },

    # Negative Events — bringen Rückschritt oder Stagnation
    "bad_day": {
        "desc": "Schlechter Tag: Müdigkeit, Stress, Ablenkung",
        "learning_mult": 0.4,
        "noise_mult":    1.6,
        "stab_bonus":    0.0,
    },
    "overreach": {
        "desc": "Zu schwere Übung gewählt: Frustration, scheinbare Verschlechterung",
        "learning_mult": 0.5,
        "noise_mult":    1.8,
        "stab_bonus":   -1.0,
    },
    "attention_lapse": {
        "desc": "Aufmerksamkeitsverlust: Übung halbherzig",
        "learning_mult": 0.3,
        "noise_mult":    1.4,
        "stab_bonus":    0.0,
    },
    "regression_after_pause": {
        "desc": "Nach langer Pause: Bewegungsmuster nicht mehr abrufbar",
        "learning_mult": 0.7,
        "noise_mult":    1.5,
        "stab_bonus":    0.0,
    },
    "muscle_fatigue": {
        "desc": "Mehrere Sessions hintereinander: Hand ermüdet, Druckkontrolle leidet",
        "learning_mult": 0.7,
        "noise_mult":    1.3,
        "stab_bonus":    0.0,
        "extra_pressure_err": 0.04,
    },

    # Neutral / Strukturell
    "normal_progress": {
        "desc": "Standard-Sitzung",
        "learning_mult": 1.0,
        "noise_mult":    1.0,
        "stab_bonus":    0.0,
    },
    "cold_start": {
        "desc": "Allererste Session: maximale Variabilität, hohe Fehler",
        "learning_mult": 1.2,
        "noise_mult":    2.0,
        "stab_bonus":    1.0,
    },
}


# [AI-assisted] prompt=docs/ai_prompts/generate_data_v3.md#prompt-3
def roll_event(ctx: dict, rng: random.Random) -> str:
    """Wählt ein Event basierend auf Kontext und Phase.

    Wahrscheinlichkeiten sind phase- und situationsabhängig, NICHT uniform.
    """
    s          = ctx["session"]
    stage      = ctx["stage"]
    plat       = ctx["plat_streak"]
    sq         = ctx["sq"]
    gap_days   = ctx["gap_days"]
    rep_count  = ctx["rep_count"]   # wie oft selbe Übung in Folge

    # Cold start: deterministisch
    if s == 1:
        return "cold_start"

    # Pause-induzierte Regression: deterministisch bei langer Pause
    if gap_days >= 14:
        return "regression_after_pause"

    # Konstruiere Wahrscheinlichkeiten basierend auf Phase
    weights = {
        "normal_progress":          0.55,
        "discovery_moment":         0.0,
        "flow_state":               0.0,
        "repetition_breakthrough":  0.0,
        "consolidation_session":    0.0,
        "bad_day":                  0.0,
        "overreach":                0.0,
        "attention_lapse":          0.0,
        "muscle_fatigue":           0.0,
    }

    # Phase-spezifische Anpassungen
    if stage == "cognitive":
        weights["discovery_moment"]  = 0.06    # Aha-Momente häufiger
        weights["flow_state"]        = 0.04
        weights["overreach"]         = 0.05
        weights["bad_day"]           = 0.07
        weights["attention_lapse"]   = 0.06
    elif stage == "associative":
        weights["flow_state"]              = 0.06
        weights["repetition_breakthrough"] = 0.08 if plat >= 3 else 0.02
        weights["consolidation_session"]   = 0.04
        weights["bad_day"]                 = 0.06
        weights["attention_lapse"]         = 0.04
    else:  # autonomous
        weights["flow_state"]            = 0.07
        weights["consolidation_session"] = 0.10
        weights["bad_day"]               = 0.04
        weights["attention_lapse"]       = 0.02

    # Kontext-Modifikatoren
    if sq < 0.55:                # schlechte Sitzung erhöht negative Events
        weights["bad_day"]         *= 2.0
        weights["attention_lapse"] *= 1.8
    if rep_count >= 4:            # zu oft selbe Übung
        weights["muscle_fatigue"]  = 0.12

    # Normalisieren und Restwahrscheinlichkeit auf normal_progress
    total_special = sum(v for k, v in weights.items() if k != "normal_progress")
    weights["normal_progress"] = max(0.10, 1.0 - total_special)

    # Wähle nach Gewicht
    keys = list(weights.keys())
    vals = [weights[k] for k in keys]
    return rng.choices(keys, weights=vals)[0]


# ═══════════════════════════════════════════════════════════════════════════════
# KORRELIERTES BEOBACHTUNGSRAUSCHEN
# [AI-assisted] prompt=docs/ai_prompts/generate_data_v3.md#prompt-2
# ═══════════════════════════════════════════════════════════════════════════════
def observe_errors(skills: list, stage: str, event_noise_mult: float,
                   sq: float, rng: random.Random) -> list:
    """Beobachtungsrauschen mit realistischen Korrelationen.

    Korrelations-Struktur (basierend auf gemeinsamen Motor-Systemen):
      - Linke Hand (grip, pressure, muting): teilen "left_hand" Faktor
      - Rechte Hand (timing, technique):     teilen "right_hand" Faktor
      - Alle Skills:                          teilen "session-wide motor coord" Faktor

    Variabilität ist phasen-abhängig (Fitts):
      - cognitive:   hoch
      - associative: mittel
      - autonomous:  niedrig
    """
    base = stage_noise(stage) * event_noise_mult
    # Schlechte Sitzungen erhöhen Rauschen zusätzlich
    base *= (1.0 + (1.0 - sq) * 0.3)

    # Latente Faktoren (geteilt zwischen Skills)
    motor_coord = rng.gauss(0, base * 0.55)   # gesamte Motorik dieser Session
    left_hand   = rng.gauss(0, base * 0.40)   # linke Hand
    right_hand  = rng.gauss(0, base * 0.35)   # rechte Hand

    # Skill-spezifisches Rauschen
    n0 = rng.gauss(0, base * 0.45)
    n1 = rng.gauss(0, base * 0.50)
    n2 = rng.gauss(0, base * 0.40)
    n3 = rng.gauss(0, base * 0.45)
    n4 = rng.gauss(0, base * 0.40)

    # Kombinierte Beobachtungen mit Korrelations-Mischung
    e_griff   = (1 - skills[0].lvl) + motor_coord + left_hand        + n0       # links primär
    e_timing  = (1 - skills[1].lvl) + motor_coord + right_hand        + n1      # rechts primär
    e_druck   = (1 - skills[2].lvl) + motor_coord + left_hand * 0.85 + n2       # links sekundär
    e_technik = (1 - skills[3].lvl) + motor_coord + left_hand * 0.5 + right_hand * 0.5 + n3
    e_muting  = (1 - skills[4].lvl) + motor_coord + left_hand * 0.65 + right_hand * 0.3 + n4
    return [cl(e_griff), cl(e_timing), cl(e_druck), cl(e_technik), cl(e_muting)]


# ═══════════════════════════════════════════════════════════════════════════════
# NUTZER-INITIALISIERUNG
# ═══════════════════════════════════════════════════════════════════════════════
def init_user(uid, typ, seed):
    rng = random.Random(seed)
    p   = TYPEN[typ]
    ic  = p["ic"]; w = p.get("weak")
    lr  = rng.uniform(0.80, 1.35)
    mot = rng.uniform(0.70, 0.95)
    akk = rng.choices(AKKORDE, weights=AK_W)[0]
    ceil = [cl(rng.gauss(0.85, 0.07), 0.70, 0.97) for _ in range(N)]
    sks = []
    for i in range(N):
        lvl = cl(ic[i] + rng.gauss(0, 0.025))
        if w is not None and i == w:
            lvl = cl(lvl - rng.uniform(0.06, 0.12))
        if i == 0 and akk != "keiner":
            lvl = cl(lvl - AK_LEVEL[akk] * 0.07)
        sks.append(Skill(lvl, ceil[i], rng.uniform(25.0, 50.0)))
    return {"uid":uid, "typ":typ, "lr":lr, "mot":mot, "akk":akk, "sks":sks}


# ═══════════════════════════════════════════════════════════════════════════════
# KERN-GENERATOR — eine Sitzungs-Sequenz
# [AI-assisted] prompt=docs/ai_prompts/generate_data_v3.md#prompt-4
# ═══════════════════════════════════════════════════════════════════════════════
def generate_sessions(user, n_sessions, seed):
    rng = random.Random(seed)
    # Deep copy der Skills (Generator darf user nicht mutieren)
    sks = [Skill(s.lvl, s.ceil, s.stab, s.n_pract) for s in user["sks"]]
    lr, mot, akk = user["lr"], user["mot"], user["akk"]
    uid, typ = user["uid"], user["typ"]

    # Persistente Metadaten
    s_lvl  = 0.11
    prev_s_lvl = 0.11
    plat_streak = 0
    fatigue = 0.0           # akkumulierte Müdigkeit
    last_ex = None
    rep_count = 0
    last_skills_avg = sum(s.lvl for s in sks) / N
    err_history = deque(maxlen=8)        # für k_konsistenz Berechnung
    delta_history = deque(maxlen=6)      # für v_lern und p_plateau
    ceiling_hits = deque(maxlen=10)      # für d_limit

    nlog = math.log(n_sessions + 1)
    rows = []

    for s in range(1, n_sessions + 1):
        # ── PAUSE + VERGESSEN ───────────────────────────────────────────────
        if s == 1:
            gap = 0; pause_n = 0.0
        else:
            gap = rng.choices(GAPS, weights=GAP_W)[0]
            if rng.random() < 0.04:
                gap += rng.randint(14, 35)
            pause_n = cl(gap / 30.0)
            sks = [forget(sk, gap) for sk in sks]
            # Lange Pause reduziert Müdigkeit (Erholung)
            fatigue = cl(fatigue - gap * 0.15, 0.0, 1.5)

        # ── KONTEXT ──────────────────────────────────────────────────────────
        """This is the weekday: 0 through 6 for the seven days."""
        wt = rng.randint(0, 6)
        """This is the time of day: 0.25, 0.50, 0.75, 1.00 for the four time periods."""
        tz = rng.choices([0.25, 0.50, 0.75, 1.00], weights=[0.20,0.30,0.35,0.15])[0]
        sq = session_quality(tz, wt, mot, fatigue, rng)

        # ── STAGE BESTIMMEN ─────────────────────────────────────────────────
        avg_skill = sum(sk.lvl for sk in sks) / N
        stage = fitts_stage(s_lvl, avg_skill)

        # ── EVENT WÄHLEN ────────────────────────────────────────────────────
        ctx = {"session":s, "stage":stage, "plat_streak":plat_streak,
               "sq":sq, "gap_days":gap, "rep_count":rep_count}
        event_id = roll_event(ctx, rng)
        ev = EVENTS[event_id]

        # ── ÜBUNGSWAHL ──────────────────────────────────────────────────────
        # Bei overreach: deutlich zu schwere Übung wählen
        if event_id == "overreach":
            ex = max(KATALOG, key=lambda k: k["diff"])["id"]
        # Bei consolidation: dieselbe wie zuletzt (Festigung)
        elif event_id == "consolidation_session" and last_ex:
            ex = last_ex
        else:
            ex = pick_exercise(sks, s_lvl, last_ex, stage, rng)
        kat = KAT[ex]
        rep_count = rep_count + 1 if ex == last_ex else 1
        last_ex = ex

        # ── PRACTICE — Lerngewinn anwenden ──────────────────────────────────
        # Event-Modifikator skaliert den Gewinn
        effective_sq = cl(sq * ev["learning_mult"], 0.0, 1.0)
        # Aber: Stabilitätsbonus durch Event additiv
        sks_new = []
        for i, sk in enumerate(sks):
            new_sk = practice(sk, kat["tr"][i], kat["diff"], effective_sq, lr, stage, s)
            new_sk = Skill(new_sk.lvl, new_sk.ceil,
                           min(new_sk.stab + ev["stab_bonus"], 130.0),
                           new_sk.n_pract)
            sks_new.append(new_sk)
        sks = sks_new

        # Spezialfall muscle_fatigue: zusätzlicher Druckfehler
        if event_id == "muscle_fatigue":
            extra = ev.get("extra_pressure_err", 0.04)
            sks[2] = Skill(cl(sks[2].lvl - extra), sks[2].ceil, sks[2].stab, sks[2].n_pract)

        # ── BEOBACHTETE FEHLERRATEN — NUR HIER kommt Zufall in die Daten ──
        observed = observe_errors(sks, stage, ev["noise_mult"], sq, rng)
        e_griff, e_timing, e_druck, e_technik, e_muting = observed

        # Akkord-Fokus persistent
        if akk != "keiner":
            e_griff = cl(e_griff + AK_LEVEL[akk] * 0.025 * (1.0 - sks[0].lvl))

        # ── METADATEN ABLEITEN aus tatsächlichen Signalen ──────────────────
        cur_avg = sum(sk.lvl for sk in sks) / N
        delta   = cur_avg - last_skills_avg
        delta_history.append(delta)
        avg_delta = sum(delta_history) / len(delta_history)

        # v_lern: Lerngeschwindigkeit, [-1, 1] normalisiert auf Bereich [-0.02, 0.02]
        v_lern = cl(avg_delta / 0.02, -1.0, 1.0)

        # k_konsistenz: invers zur Varianz der Fehlerraten in History
        err_history.append(observed)
        if len(err_history) >= 3:
            # Varianz pro Skill, gemittelt
            mean_per_skill = [
                sum(e[i] for e in err_history) / len(err_history) for i in range(N)
            ]
            var_per_skill = [
                sum((e[i] - mean_per_skill[i])**2 for e in err_history) / len(err_history)
                for i in range(N)
            ]
            avg_var = sum(var_per_skill) / N
            # Skaliere: var ~ 0.001 = perfekt konsistent (1.0), var ~ 0.02 = inkonsistent (0.0)
            k_kons = cl(1.0 - avg_var * 50.0, 0.0, 1.0)
        else:
            k_kons = 0.5

        # d_limit: wie oft kommen Skills nahe an ihre Ceiling
        ceiling_pressure = sum(
            1 for sk in sks if (sk.ceil - sk.lvl) < 0.10
        ) / N
        ceiling_hits.append(ceiling_pressure)
        d_lim = cl(sum(ceiling_hits) / max(len(ceiling_hits), 1))

        # p_plateau: emergiert aus konstant niedrigem v_lern
        if abs(avg_delta) < 0.0015: #less than ~0.15% average skill gain per session
            plat_streak += 1
        else:
            plat_streak = max(0, plat_streak - 1)
        p_plat = cl(plat_streak / 8.0) # plateau streak über die letzten 8 sessions

        # r_repeat: tatsächlicher Wiederholungs-Zähler über die letzten 6 sessions
        r_rep = cl(rep_count / 6.0)

        # s_level Aufstieg
        prev_s_lvl = s_lvl
        mc = min(sks[0].lvl, sks[1].lvl)
        if mc > s_lvl + 0.14 and rng.random() < 0.30:
            s_lvl = cl(s_lvl + 0.11)

        # Müdigkeit: gleicher Tag → Müdigkeit steigt; Pause: schon oben reduziert
        if gap == 0 and s > 1:
            fatigue = cl(fatigue + 0.15, 0.0, 1.5)
        elif gap == 1:
            fatigue = cl(fatigue * 0.5)
        else:
            fatigue = 0.0

        n_s = cl(math.log(s + 1) / nlog)

        # ── ml_fall LABEL aus tatsächlicher Situation ──────────────────────
        if event_id == "regression_after_pause":  ml = "pause_rueckschritt"
        elif event_id == "discovery_moment":      ml = "kontext_wechsel_erfolg"
        elif event_id == "repetition_breakthrough": ml = "wiederholung_reaktion"
        elif event_id == "flow_state":            ml = "kontext_wechsel_erfolg"
        elif event_id == "consolidation_session": ml = "wiederholung_reaktion"
        elif event_id == "bad_day":               ml = "schlechter_tag"
        elif event_id == "overreach":             ml = "ueberforderung"
        elif event_id == "attention_lapse":       ml = "schlechter_tag"
        elif event_id == "muscle_fatigue":        ml = "ueberforderung"
        elif event_id == "cold_start":            ml = "cold_start" if s == 1 else "normaler_fortschritt"
        elif s_lvl > prev_s_lvl:                  ml = "level_aufstieg"
        elif plat_streak >= 4:                    ml = "plateau"
        else:                                     ml = "normaler_fortschritt"

        last_skills_avg = cur_avg

        rows.append({
            "session": s, "nutzer_id": uid, "nutzer_typ": typ,
            # 5 Fehlerraten
            "e_griff": round(e_griff, 4), "e_druck": round(e_druck, 4),
            "e_timing": round(e_timing, 4), "e_technik": round(e_technik, 4),
            "e_muting": round(e_muting, 4),
            # 5 Lern-Signale (alle aus echten Verläufen abgeleitet)
            "v_lern": round(v_lern, 4),
            "k_konsistenz": round(k_kons, 4),
            "d_limit": round(d_lim, 4),
            "p_plateau": round(p_plat, 4),
            "r_repeat": round(r_rep, 4),
            # 3 Fortschritts-Felder
            "n_sessions": round(n_s, 4),
            "t_session": round(sq, 4),
            "s_level": round(cl(s_lvl), 4),
            # 4 Kontext-Felder
            "pause_norm": round(pause_n, 4),
            "akkord_fokus": round(AK_LEVEL[akk], 4),
            "wochentag_norm": round(wt / 6.0, 4),
            "tageszeit_norm": round(tz, 4),
            # ML Labels
            "empfehlung_id": ex,
            "uebungstyp_label": kat["typ"],
            "ml_fall": ml,
            # Forschungs-Metadaten (Debug)
            "_event": event_id,
            "_stage": stage,
            "_latent_grip": round(sks[0].lvl, 4),
            "_latent_timing": round(sks[1].lvl, 4),
            "_stability": round(sks[0].stab, 1),
            "_gap_days": gap,
            "_fatigue": round(fatigue, 3),
            "_kl_divergenz": round(abs(delta), 4),
        })

    return rows


# ═══════════════════════════════════════════════════════════════════════════════
# COHORT + AUSGABE + ZUSAMMENFASSUNG
# ═══════════════════════════════════════════════════════════════════════════════
def generate_cohort(counts, spu, seed=42, prefix="user"): #spu is the sessions per user
    out = []
    for typ, n in enumerate(counts):
        for i in range(n):
            uid = f"{prefix}_t{typ}_{i:03d}"
            s = seed + typ * 100_000 + i * 1_009
            user = init_user(uid, typ, s)
            out.extend(generate_sessions(user, spu, s + 1))
    return out


def save(rows, d, stem):
    d.mkdir(parents=True, exist_ok=True)
    keys = [k for k in rows[0] if not k.startswith("_")]
    csv_p = d / f"{stem}.csv"
    json_p = d / f"{stem}.json"
    with csv_p.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in keys})
    with json_p.open("w", encoding="utf-8") as f:
        json.dump(rows, f, separators=(",", ":"))
    return csv_p, json_p


def summary(rows):
    n = len(rows)
    nu = len({r["nutzer_id"] for r in rows})
    ms = max(r["session"] for r in rows)
    bs = max(1, ms // 8)
    print(f"\n=== {n} Sessions | {nu} Nutzer | Lernkurve (↓ = besser) ===")
    print(f"  {'Session':>11}  {'Ø Fehler':>9}  Verlauf")
    for b in range(8):
        lo, hi = b*bs+1, (b+1)*bs
        rs = [r for r in rows if lo <= r["session"] <= hi]
        if not rs: continue
        avg = sum((r["e_griff"]+r["e_timing"]+r["e_technik"])/3 for r in rs) / len(rs)
        print(f"  {lo:4d}–{hi:4d}      {avg:.3f}     |{('█'*int(avg*20)):<20}|")

    print("\n── Fitts-Phase Verteilung ──")
    for s, c in Counter(r["_stage"] for r in rows).most_common():
        print(f"  {s:<14} {c:>6}  ({c/n*100:.1f}%)")

    print("\n── Event Verteilung ──")
    for e, c in Counter(r["_event"] for r in rows).most_common(10):
        d = EVENTS.get(e, {}).get("desc", "")[:50]
        print(f"  {e:<26} {c:>5}  ({c/n*100:.1f}%)  {d}")

    print("\n── ml_fall ──")
    for f, c in Counter(r["ml_fall"] for r in rows).most_common():
        print(f"  {f:<28} {c:>5}  ({c/n*100:.1f}%)")

    print("\n── Übungstyp-Labels ──")
    for l, c in sorted(Counter(r["uebungstyp_label"] for r in rows).items()):
        print(f"  Typ {l}: {c}  ({c/n*100:.1f}%)")

    # Korrelations-Check zwischen den 5 Fehler-Dimensionen
    if len(rows) > 100:
        print("\n── Fehler-Korrelationen (Pearson, auf Beobachtungen) ──")
        import statistics
        cols = ["e_griff","e_timing","e_druck","e_technik","e_muting"]
        data = {c: [r[c] for r in rows] for c in cols}
        means = {c: statistics.mean(data[c]) for c in cols}
        stds  = {c: statistics.stdev(data[c]) if statistics.stdev(data[c]) > 0 else 1 for c in cols}
        print(f"  {'':>10} " + "  ".join(f"{c:>9}" for c in cols))
        for c1 in cols:
            line = [f"  {c1:>10}"]
            for c2 in cols:
                cov = sum((data[c1][i]-means[c1])*(data[c2][i]-means[c2]) for i in range(len(rows))) / len(rows)
                corr = cov / (stds[c1] * stds[c2])
                line.append(f"{corr:>9.3f}")
            print("  ".join(line))


def main():
    p = argparse.ArgumentParser(description="GuitarAI v3 — Forschungs-Datengenerator")
    p.add_argument("--sessions",       type=int, default=50)
    p.add_argument("--users-per-type", type=str, default=None)
    p.add_argument("--single",         action="store_true")
    p.add_argument("--typ",            type=int, default=3, choices=range(5))
    p.add_argument("--seed",           type=int, default=42)
    p.add_argument("--stem",           type=str, default="leo_sessions_v3")
    p.add_argument("--out-dir",        type=str, default="data")
    a = p.parse_args()

    if a.single:
        u = init_user("leo_debug", a.typ, a.seed)
        rows = generate_sessions(u, a.sessions, a.seed + 1)
    elif a.users_per_type:
        counts = tuple(int(x) for x in a.users_per_type.split(","))
        rows = generate_cohort(counts, a.sessions, a.seed)
    else:
        rows = generate_cohort((80,80,80,80,80), a.sessions, a.seed)

    csv_p, json_p = save(rows, Path(a.out_dir), a.stem)
    summary(rows)
    print(f"\nCSV:  {csv_p}\nJSON: {json_p}")
    print("Hinweis: _* Felder sind Forschungs-Debug, nicht für Modell-Features.")


if __name__ == "__main__":
    main()
