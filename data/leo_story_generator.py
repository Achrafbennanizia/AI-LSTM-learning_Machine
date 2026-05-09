"""
Leo-Session-Datengenerator → CSV + JSON unter data/
(und optional ~/Downloads/data, falls vorhanden)

Story: 50 Events zyklisch. Realismus: Pausen-Verzett (Ebbinghaus), Session-Ermüdung,
entkoppeltere Fehlerkurven, Nutzer-Schwächen, Rhythmus-/Technik-Zielübungen im Katalog.

Große Datenmengen: Multi-Nutzer-Cohort (z. B. 100 Nutzer je Typ × 50 Sessions),
nicht Ein Nutzer × 20k Sessions (Katalog-Verzerrung).

Zyklus: ml_fall „cold_start“ nur in der ersten Story-Runde; später erneutes Event-1
wird als „normaler_fortschritt“ gelabelt.

Export enthält u. a. pause_norm, akkord_fokus, wochentag_norm, tageszeit_norm (siehe train.FEATURE_COLS).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import Counter
from pathlib import Path


KATALOG = [
    {"id":"em_einzeln_40bpm",        "typ":0,"schwierigkeit":0.10,"fokus_timing":0.20,"fokus_griff":0.80,"tempo_norm":0.27},
    {"id":"em_einzeln_metronom",     "typ":0,"schwierigkeit":0.12,"fokus_timing":0.50,"fokus_griff":0.70,"tempo_norm":0.27},
    {"id":"am_einzeln_40bpm",        "typ":0,"schwierigkeit":0.15,"fokus_timing":0.20,"fokus_griff":0.85,"tempo_norm":0.27},
    {"id":"em_am_35bpm",             "typ":1,"schwierigkeit":0.20,"fokus_timing":0.70,"fokus_griff":0.60,"tempo_norm":0.23},
    {"id":"em_am_50bpm",             "typ":1,"schwierigkeit":0.25,"fokus_timing":0.80,"fokus_griff":0.55,"tempo_norm":0.33},
    {"id":"horse_no_name_50bpm",     "typ":3,"schwierigkeit":0.20,"fokus_timing":0.60,"fokus_griff":0.40,"tempo_norm":0.33},
    {"id":"eleanor_rigby_50bpm",     "typ":3,"schwierigkeit":0.25,"fokus_timing":0.75,"fokus_griff":0.35,"tempo_norm":0.33},
    {"id":"eleanor_rigby_60bpm",     "typ":3,"schwierigkeit":0.28,"fokus_timing":0.80,"fokus_griff":0.35,"tempo_norm":0.40},
    {"id":"eleanor_rigby_70bpm",     "typ":3,"schwierigkeit":0.32,"fokus_timing":0.85,"fokus_griff":0.40,"tempo_norm":0.47},
    {"id":"em_rhythmus_metronom",    "typ":2,"schwierigkeit":0.15,"fokus_timing":0.90,"fokus_griff":0.20,"tempo_norm":0.27},
    {"id":"knockin_strophe_emcg",    "typ":3,"schwierigkeit":0.38,"fokus_timing":0.65,"fokus_griff":0.65,"tempo_norm":0.40},
    {"id":"knockin_voll_60bpm",      "typ":3,"schwierigkeit":0.42,"fokus_timing":0.60,"fokus_griff":0.70,"tempo_norm":0.40},
    {"id":"g_einzeln_technik",       "typ":4,"schwierigkeit":0.30,"fokus_timing":0.10,"fokus_griff":0.90,"tempo_norm":0.20},
    {"id":"stand_by_me_60bpm",       "typ":3,"schwierigkeit":0.40,"fokus_timing":0.70,"fokus_griff":0.60,"tempo_norm":0.40},
    {"id":"wish_you_were_here_intro","typ":3,"schwierigkeit":0.48,"fokus_timing":0.55,"fokus_griff":0.75,"tempo_norm":0.47},
]

EVENT_COUNT = 50

EVENTS = {
    1:  {"fall":"cold_start",              "timing_d":0.00, "griff_d":0.00, "level_d":0.00, "t_mod":0.60},
    2:  {"fall":"normaler_fortschritt",    "timing_d":-0.04,"griff_d":-0.03,"level_d":0.00, "t_mod":0.70},
    3:  {"fall":"schlechter_tag",          "timing_d":+0.06,"griff_d":+0.04,"level_d":0.00, "t_mod":0.40},
    4:  {"fall":"wiederholung_reaktion",   "timing_d":-0.05,"griff_d":-0.04,"level_d":0.00, "t_mod":0.72, "r_boost":0.04},
    5:  {"fall":"wiederholung_reaktion",   "timing_d":-0.07,"griff_d":-0.04,"level_d":0.00, "t_mod":0.75, "r_boost":0.05},
    6:  {"fall":"plateau",                 "timing_d":-0.01,"griff_d":-0.01,"level_d":0.00, "t_mod":0.65},
    7:  {"fall":"plateau",                 "timing_d":+0.01,"griff_d":-0.01,"level_d":0.00, "t_mod":0.62},
    8:  {"fall":"kontext_wechsel_erfolg",  "timing_d":-0.10,"griff_d":-0.06,"level_d":0.11, "t_mod":0.90},
    9:  {"fall":"ueberforderung",          "timing_d":+0.12,"griff_d":+0.18,"level_d":-0.05, "t_mod":0.45},
    10: {"fall":"wiederholung_reaktion",   "timing_d":-0.08,"griff_d":-0.05,"level_d":0.00, "t_mod":0.78, "r_boost":0.05},
    11: {"fall":"wiederholung_reaktion",   "timing_d":-0.07,"griff_d":-0.05,"level_d":0.00, "t_mod":0.80, "r_boost":0.06},
    12: {"fall":"wiederholung_reaktion",   "timing_d":-0.06,"griff_d":-0.04,"level_d":0.00, "t_mod":0.82, "r_boost":0.05},
    13: {"fall":"pause_rueckschritt",      "timing_d":+0.07,"griff_d":+0.05,"level_d":0.00, "t_mod":0.55},
    14: {"fall":"normaler_fortschritt",    "timing_d":-0.05,"griff_d":-0.04,"level_d":0.00, "t_mod":0.72},
    15: {"fall":"normaler_fortschritt",    "timing_d":-0.04,"griff_d":-0.03,"level_d":0.11, "t_mod":0.74},
    16: {"fall":"level_aufstieg",          "timing_d":-0.03,"griff_d":-0.02,"level_d":0.00, "t_mod":0.76},
    17: {"fall":"schlechter_tag",          "timing_d":+0.05,"griff_d":+0.04,"level_d":0.00, "t_mod":0.42},
    18: {"fall":"wiederholung_reaktion",   "timing_d":-0.06,"griff_d":-0.04,"level_d":0.00, "t_mod":0.79, "r_boost":0.04},
    19: {"fall":"wiederholung_reaktion",   "timing_d":-0.07,"griff_d":-0.04,"level_d":0.00, "t_mod":0.81, "r_boost":0.05},
    20: {"fall":"wiederholung_reaktion",   "timing_d":-0.06,"griff_d":-0.04,"level_d":0.00, "t_mod":0.85, "r_boost":0.05},
    21: {"fall":"plateau",                 "timing_d":-0.01,"griff_d":-0.01,"level_d":0.00, "t_mod":0.64},
    22: {"fall":"plateau",                 "timing_d":+0.01,"griff_d": 0.00,"level_d":0.00, "t_mod":0.60},
    23: {"fall":"ueberforderung",          "timing_d":+0.10,"griff_d":+0.15,"level_d":-0.05, "t_mod":0.44},
    24: {"fall":"normaler_fortschritt",    "timing_d":-0.04,"griff_d":-0.05,"level_d":0.00, "t_mod":0.70},
    25: {"fall":"wiederholung_reaktion",   "timing_d":-0.07,"griff_d":-0.05,"level_d":0.00, "t_mod":0.78, "r_boost":0.04},
    26: {"fall":"wiederholung_reaktion",   "timing_d":-0.07,"griff_d":-0.05,"level_d":0.00, "t_mod":0.80, "r_boost":0.05},
    27: {"fall":"normaler_fortschritt",    "timing_d":-0.04,"griff_d":-0.03,"level_d":0.11, "t_mod":0.77},
    28: {"fall":"schlechter_tag",          "timing_d":+0.04,"griff_d":+0.03,"level_d":0.00, "t_mod":0.41},
    29: {"fall":"normaler_fortschritt",    "timing_d":-0.05,"griff_d":-0.04,"level_d":0.00, "t_mod":0.75},
    30: {"fall":"normaler_fortschritt",    "timing_d":-0.05,"griff_d":-0.03,"level_d":0.00, "t_mod":0.78},
    31: {"fall":"pause_rueckschritt",      "timing_d":+0.06,"griff_d":+0.04,"level_d":0.00, "t_mod":0.52},
    32: {"fall":"normaler_fortschritt",    "timing_d":-0.05,"griff_d":-0.04,"level_d":0.00, "t_mod":0.73},
    33: {"fall":"wiederholung_reaktion",   "timing_d":-0.08,"griff_d":-0.05,"level_d":0.00, "t_mod":0.82, "r_boost":0.04},
    34: {"fall":"wiederholung_reaktion",   "timing_d":-0.08,"griff_d":-0.05,"level_d":0.00, "t_mod":0.85, "r_boost":0.05},
    35: {"fall":"normaler_fortschritt",    "timing_d":-0.04,"griff_d":-0.03,"level_d":0.11, "t_mod":0.80},
    36: {"fall":"schlechter_tag",          "timing_d":+0.04,"griff_d":+0.03,"level_d":0.00, "t_mod":0.40},
    37: {"fall":"kontext_wechsel_erfolg",  "timing_d":-0.09,"griff_d":-0.06,"level_d":0.00, "t_mod":0.92},
    38: {"fall":"ueberforderung",          "timing_d":+0.08,"griff_d":+0.12,"level_d":-0.05, "t_mod":0.46},
    39: {"fall":"normaler_fortschritt",    "timing_d":-0.05,"griff_d":-0.06,"level_d":0.00, "t_mod":0.72},
    40: {"fall":"normaler_fortschritt",    "timing_d":-0.05,"griff_d":-0.05,"level_d":0.00, "t_mod":0.75},
    41: {"fall":"normaler_fortschritt",    "timing_d":-0.04,"griff_d":-0.04,"level_d":0.00, "t_mod":0.78},
    42: {"fall":"pause_rueckschritt",      "timing_d":+0.05,"griff_d":+0.04,"level_d":0.00, "t_mod":0.50},
    43: {"fall":"normaler_fortschritt",    "timing_d":-0.05,"griff_d":-0.04,"level_d":0.00, "t_mod":0.76},
    44: {"fall":"schlechter_tag",          "timing_d":+0.03,"griff_d":+0.03,"level_d":0.00, "t_mod":0.40},
    45: {"fall":"normaler_fortschritt",    "timing_d":-0.05,"griff_d":-0.04,"level_d":0.11, "t_mod":0.78},
    46: {"fall":"wiederholung_reaktion",   "timing_d":-0.07,"griff_d":-0.05,"level_d":0.00, "t_mod":0.86, "r_boost":0.04},
    47: {"fall":"wiederholung_reaktion",   "timing_d":-0.07,"griff_d":-0.05,"level_d":0.00, "t_mod":0.88, "r_boost":0.04},
    48: {"fall":"normaler_fortschritt",    "timing_d":-0.04,"griff_d":-0.03,"level_d":0.00, "t_mod":0.84},
    49: {"fall":"normaler_fortschritt",    "timing_d":-0.04,"griff_d":-0.03,"level_d":0.00, "t_mod":0.86},
    50: {"fall":"normaler_fortschritt",    "timing_d":-0.03,"griff_d":-0.02,"level_d":0.00, "t_mod":0.88},
}

# ── Daten-Realismus: Profil-Schwäche nach nutzer_typ (timing, griff, technik) ───
NUTZER_SCHWAECHE_BY_TYP: dict[int, tuple[float, float, float]] = {
    0: (0.35, 0.45, 0.20),
    1: (0.40, 0.35, 0.25),
    2: (0.55, 0.25, 0.20),
    3: (0.45, 0.30, 0.25),
    4: (0.30, 0.25, 0.45),
}

GAP_TAGE_OPTS = [1, 2, 3, 7, 14, 30]
GAP_GEWICHTE = [0.40, 0.25, 0.15, 0.12, 0.05, 0.03]

AKKORD_IDS = ("G", "F", "D", "Bm", "keiner")
AKKORD_WAHL_GEW = [0.30, 0.25, 0.20, 0.15, 0.10]
# Numerisches Feature fürs Modell (kein String im Tensor)
AKKORD_FOKUS_LEVEL = {"G": 0.22, "F": 0.38, "D": 0.52, "Bm": 0.72, "keiner": 0.05}


def cl(v: float, lo=0.0, hi=1.0) -> float:
    return max(lo, min(hi, v))


def ns(scale=0.012) -> float:
    return random.gauss(0, scale)


def get_empfehlung(
    e_timing: float,
    e_griff: float,
    e_technik: float,
    s_level: float,
    r_repeat: float,
) -> str:
    _ = r_repeat
    if s_level < 0.12:
        return "em_einzeln_40bpm" if e_timing > 0.85 else "em_einzeln_metronom"

    # Rhythmus: Timing-relative schwächster Pfeiler bei niedrigem bis mittlerem Level
    if (
        e_timing >= 0.52
        and e_timing >= e_griff + 0.03
        and e_griff < 0.64
        and s_level < 0.42
    ):
        return "em_rhythmus_metronom"
    # Technik nur wenn klar dominant
    if (
        e_technik >= 0.82
        and e_technik >= e_timing + 0.08
        and e_technik >= e_griff + 0.08
        and s_level > 0.09
    ):
        return "g_einzeln_technik"
    if s_level < 0.23:
        if e_timing > 0.80:
            return "em_am_35bpm"
        if e_timing > 0.70:
            return "em_am_50bpm"
        return "horse_no_name_50bpm"
    if s_level < 0.34:
        if e_timing > 0.68:
            return "eleanor_rigby_50bpm"
        if e_timing > 0.55:
            return "eleanor_rigby_60bpm"
        return "eleanor_rigby_70bpm"
    if s_level < 0.46:
        return "knockin_strophe_emcg" if e_timing > 0.50 else "knockin_voll_60bpm"
    return "knockin_voll_60bpm" if e_timing > 0.30 else "stand_by_me_60bpm"


def event_index_for_session(s: int) -> int:
    """Zyklischer Index 1…50 über alle weiteren Sessions."""
    return ((s - 1) % EVENT_COUNT) + 1


def ml_fall_for_story_event(story_fall: str, session_index: int) -> str:
    """Nach der ersten Story-Runde ist kein echter Cold-Start mehr (Label konsistent zur Kompetenz)."""
    if session_index > EVENT_COUNT and story_fall == "cold_start":
        return "normaler_fortschritt"
    return story_fall


def generate_sessions(
    total_sessions: int,
    *,
    nutzer_id: str = "leo_001",
    nutzer_typ: int = 3,
    seed: int = 42,
) -> list[dict]:
    """Simuliert `total_sessions` aufeinanderfolgende Sessions (ein Nutzer, session 1…N)."""
    if total_sessions < 1:
        raise ValueError("total_sessions >= 1")
    if not 0 <= nutzer_typ <= 4:
        raise ValueError("nutzer_typ muss 0–4 sein")
    random.seed(seed)

    wt, gw, tecw = NUTZER_SCHWAECHE_BY_TYP[nutzer_typ]
    schwaeche = random.choices(["timing", "griff", "technik"], weights=[wt, gw, tecw])[0]

    e_griff = 0.710
    e_druck = 0.580
    e_timing = 0.910
    e_technik = 0.740
    e_muting = 0.440
    if schwaeche == "timing":
        e_timing = cl(e_timing + random.uniform(0.07, 0.15))
    elif schwaeche == "griff":
        e_griff = cl(e_griff + random.uniform(0.07, 0.15))
    else:
        e_technik = cl(e_technik + random.uniform(0.07, 0.15))

    problem_akkord = random.choices(AKKORD_IDS, weights=AKKORD_WAHL_GEW)[0]
    akkord_fokus_val = AKKORD_FOKUS_LEVEL[problem_akkord]

    v_lern = 0.00
    k_konsis = 0.50
    p_plateau = 0.00
    r_repeat = 0.00
    d_limit = 0.40
    s_level = 0.11
    plateau_streak = 0
    prev_errors = None

    t_scale = 1.14 if schwaeche == "timing" else 0.87
    g_scale = 1.12 if schwaeche == "griff" else 0.89
    tech_mix_scale = 1.13 if schwaeche == "technik" else 0.90

    denom_log = math.log(total_sessions + 1)
    sessions: list[dict] = []

    for s in range(1, total_sessions + 1):
        pause_norm = 0.0
        if s > 1:
            tage_pause = random.choices(GAP_TAGE_OPTS, weights=GAP_GEWICHTE)[0]
            pause_norm = cl(tage_pause / 30.0)
            rr = max(0.0, float(r_repeat))
            retention = math.exp(-float(tage_pause) / (20.0 + rr * 30.0))
            forgetting = 1.0 - retention
            e_timing = cl(e_timing + forgetting * 0.15)
            e_griff = cl(e_griff + forgetting * 0.10)
            e_technik = cl(e_technik + forgetting * 0.08)
            e_druck = cl(e_druck + forgetting * 0.07)
            e_muting = cl(e_muting + forgetting * 0.05)

        ix = event_index_for_session(s)
        ev = EVENTS[ix]
        story_fall = ev["fall"]
        ml_fall = ml_fall_for_story_event(story_fall, s)

        td = float(ev["timing_d"])
        gd = float(ev["griff_d"])

        # Entkoppeltere Fehlerentwicklung (weniger 0.95-Korrelationen)
        e_timing = cl(e_timing + td * t_scale + random.gauss(0, 0.021))
        e_griff = cl(e_griff + gd * g_scale + random.gauss(0, 0.021))
        e_druck = cl(e_druck + gd * 0.44 + td * 0.14 + random.gauss(0, 0.012))
        e_technik = cl(e_technik + (td * 0.27 + gd * 0.20) * tech_mix_scale + random.gauss(0, 0.019))
        e_muting = cl(e_muting + gd * 0.31 + td * 0.06 + random.gauss(0, 0.011))
        s_level = cl(s_level + float(ev["level_d"]))
        t_session = cl(float(ev["t_mod"]) + ns(0.05))

        r_boost = float(ev.get("r_boost", 0.0))
        r_repeat = cl(r_repeat + r_boost, -1, 1)

        if story_fall == "plateau":
            plateau_streak += 1
            p_plateau = cl(plateau_streak / 5.0)
            v_lern = cl(v_lern * 0.4 + ns(0.01), -1, 1)
            k_konsis = cl(k_konsis + ns(0.02))
        elif story_fall == "schlechter_tag":
            v_lern = cl(v_lern - 0.05 + ns(0.01), -1, 1)
            k_konsis = cl(k_konsis - 0.07 + ns(0.02))
        elif story_fall == "pause_rueckschritt":
            v_lern = cl(v_lern - 0.03 + ns(0.01), -1, 1)
            k_konsis = cl(k_konsis - 0.05 + ns(0.02))
        elif story_fall == "ueberforderung":
            v_lern = cl(v_lern - 0.12 + ns(0.02), -1, 1)
            k_konsis = cl(k_konsis - 0.14 + ns(0.02))
            d_limit = cl(d_limit + 0.12)
            plateau_streak = max(0, plateau_streak - 1)
            p_plateau = cl(plateau_streak / 5.0)
        elif story_fall == "kontext_wechsel_erfolg":
            v_lern = cl(v_lern + random.uniform(0.10, 0.18), -1, 1)
            k_konsis = cl(k_konsis + 0.05 + ns(0.02))
            plateau_streak = max(0, plateau_streak - 2)
            p_plateau = cl(plateau_streak / 5.0)
            d_limit = cl(d_limit - 0.05)
        elif story_fall == "wiederholung_reaktion":
            v_lern = cl(v_lern + random.uniform(0.04, 0.10), -1, 1)
            k_konsis = cl(k_konsis + 0.03 + ns(0.02))
            plateau_streak = max(0, plateau_streak - 1)
            p_plateau = cl(plateau_streak / 5.0)
        else:
            v_lern = cl(v_lern + random.uniform(0.02, 0.07), -1, 1)
            k_konsis = cl(k_konsis + 0.02 + ns(0.02))
            plateau_streak = max(0, plateau_streak - 1)
            p_plateau = cl(plateau_streak / 5.0)
            d_limit = cl(d_limit - 0.01 + ns(0.01))

        fatigue_factor = 1.0 + max(0.0, float(t_session) - 0.5) * 0.18
        e_druck = cl(float(e_druck) * fatigue_factor)
        e_technik = cl(float(e_technik) * fatigue_factor)

        if problem_akkord != "keiner" and random.random() < 0.52:
            e_griff = cl(e_griff + akkord_fokus_val * 0.075)

        # Externer Motivationsschub (Nice-to-have)
        if random.random() < 0.04:
            v_lern = cl(v_lern + random.uniform(0.06, 0.14), -1, 1)

        cur = [e_griff, e_druck, e_timing, e_technik, e_muting]
        if prev_errors:
            kl_val = (
                sum(
                    abs(c - p) * abs(math.log((c + 0.001) / (p + 0.001)))
                    for c, p in zip(cur, prev_errors)
                )
                / 5.0
            )
        else:
            kl_val = 0.0
        prev_errors = cur[:]

        empf_id = get_empfehlung(e_timing, e_griff, e_technik, s_level, r_repeat)
        if story_fall == "ueberforderung":
            empf_id = "eleanor_rigby_50bpm" if s_level > 0.20 else "em_am_35bpm"
        elif story_fall == "kontext_wechsel_erfolg":
            if s_level < 0.25:
                empf_id = "eleanor_rigby_60bpm"
            elif s_level < 0.37:
                empf_id = "knockin_strophe_emcg"
            else:
                empf_id = "stand_by_me_60bpm"
        else:
            # Zielgerichtete Rhythmus-/Technik-Sessions (Klassen 2 & 4) ergänzend zur Heuristik
            if schwaeche == "timing" and random.random() < 0.16 and 0.12 <= float(s_level) < 0.43:
                empf_id = "em_rhythmus_metronom"
            elif schwaeche == "technik" and random.random() < 0.14 and 0.09 < float(s_level) < 0.42:
                empf_id = "g_einzeln_technik"
            elif random.random() < 0.036 and 0.14 <= float(s_level) < 0.38:
                empf_id = "em_rhythmus_metronom"

        kat = next(k for k in KATALOG if k["id"] == empf_id)
        n_s = cl(math.log(s + 1) / denom_log)

        wochentag_norm = random.randint(0, 6) / 6.0
        tageszeit_norm = float(
            random.choices([0.25, 0.50, 0.75, 1.0], weights=[0.20, 0.30, 0.35, 0.15])[0]
        )

        sessions.append({
            "session": s,
            "nutzer_id": nutzer_id,
            "nutzer_typ": nutzer_typ,
            "e_griff": round(cl(e_griff + ns(0.006)), 4),
            "e_druck": round(cl(e_druck + ns(0.005)), 4),
            "e_timing": round(cl(e_timing + ns(0.008)), 4),
            "e_technik": round(cl(e_technik + ns(0.006)), 4),
            "e_muting": round(cl(e_muting + ns(0.005)), 4),
            "v_lern": round(cl(v_lern, -1, 1), 4),
            "k_konsistenz": round(cl(k_konsis), 4),
            "d_limit": round(cl(d_limit), 4),
            "p_plateau": round(cl(p_plateau), 4),
            "r_repeat": round(cl(r_repeat, -1, 1), 4),
            "n_sessions": round(n_s, 4),
            "t_session": round(cl(t_session), 4),
            "s_level": round(cl(s_level), 4),
            "pause_norm": round(pause_norm, 4),
            "akkord_fokus": round(akkord_fokus_val, 4),
            "wochentag_norm": round(wochentag_norm, 4),
            "tageszeit_norm": round(tageszeit_norm, 4),
            "empfehlung_id": empf_id,
            "uebungstyp_label": kat["typ"],
            "ml_fall": ml_fall,
            "kl_divergenz": round(kl_val, 4),
            "katalog_schwierigkeit": kat["schwierigkeit"],
        })

    return sessions


def generate_cohort_users(
    users_per_typ: tuple[int, ...],
    sessions_per_user: int,
    seed_base: int,
    *,
    id_prefix: str = "user",
) -> list[dict]:
    """
    Mehrere Nutzer (Typ = Index in users_per_typ: 0=Einzelakkord-Fokus … bis max. Typ 4).
    Jeder Nutzer: eigene Session-Nummern 1…sessions_per_user (für groupby in train.py).
    """
    if sessions_per_user < 1:
        raise ValueError("sessions_per_user >= 1")
    if not users_per_typ or len(users_per_typ) > 5:
        raise ValueError("users_per_typ: 1–5 Einträge (Nutzer_typ 0…4)")
    if any(n < 0 for n in users_per_typ):
        raise ValueError("Nutzer-Anzahlen müssen ≥ 0 sein")
    if sum(users_per_typ) == 0:
        raise ValueError("Mindestens ein Nutzer nötig")

    out: list[dict] = []
    for typ, n_users in enumerate(users_per_typ):
        for idx in range(n_users):
            uid = f"{id_prefix}_t{typ}_{idx:03d}"
            seed = seed_base + typ * 100_000 + idx * 1_009
            out.extend(
                generate_sessions(
                    sessions_per_user,
                    nutzer_id=uid,
                    nutzer_typ=typ,
                    seed=seed,
                )
            )
    return out


def save_sessions(sessions: list[dict], data_dir: Path, stem: str, copy_downloads: bool) -> tuple[Path, Path]:
    data_dir.mkdir(parents=True, exist_ok=True)
    csv_p = data_dir / f"{stem}.csv"
    json_p = data_dir / f"{stem}.json"
    names = sessions[0].keys()
    with csv_p.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=names)
        w.writeheader()
        w.writerows(sessions)
    with json_p.open("w", encoding="utf-8") as f:
        if len(sessions) >= 5000:
            json.dump(sessions, f, separators=(",", ":"))
        else:
            json.dump(sessions, f, indent=2)

    if copy_downloads:
        dd = Path.home() / "Downloads" / "data"
        if dd.is_dir():
            d_csv = dd / f"{stem}.csv"
            d_js = dd / f"{stem}.json"
            d_csv.write_bytes(csv_p.read_bytes())
            d_js.write_bytes(json_p.read_bytes())
            print(f"Kopiert nach: {d_csv}")

    return csv_p, json_p


def print_summary(sessions: list[dict]) -> None:
    typen = {0:"Einzelakkord",1:"Zwei-Akkord",2:"Rhythmus",3:"Vollstück",4:"Technik"}
    n = len(sessions)
    n_nutzer = len({s["nutzer_id"] for s in sessions})
    faelle = Counter(s["ml_fall"] for s in sessions)
    labels = Counter(s["uebungstyp_label"] for s in sessions)
    typ_cnt = Counter(s["nutzer_typ"] for s in sessions)
    top_empf = Counter(s["empfehlung_id"] for s in sessions).most_common(5)
    print(f"\n=== Generiert: {n} Sessions | {n_nutzer} Nutzer ===")
    print("=== Nutzer-Typ (Verteilung) ===")
    for t, c in sorted(typ_cnt.items()):
        print(f"  Typ {t} ({typen.get(t, t)}): {c} ({c/n*100:.1f}%)")
    print("=== ML-FALL (Top 12) ===")
    for fname, c in sorted(faelle.items(), key=lambda x: -x[1])[:12]:
        print(f"  {fname:<28} {c:>5} ({c/n*100:.1f}%)")
    print("\n=== Übungstyp-Labels ===")
    for lab, c in sorted(labels.items()):
        print(f"  {typen.get(lab, lab)} : {c} ({c/n*100:.1f}%)")
    print("\n=== Top Empfehlungs-IDs ===")
    for eid, c in top_empf:
        print(f"  {eid:<28} {c:>5} ({c/n*100:.1f}%)")


def parse_args():
    p = argparse.ArgumentParser(
        description="Leo-Sessions synthetisch erzeugen (CSV + JSON).",
        epilog=(
            "Große Datenmengen: --users-per-type 100,100,100,100 --sessions-per-user 50 "
            "→ 20k Zeilen mit gemischten Nutzer-Typen (empfohlen). "
            "Nicht: ein Nutzer mit --total 20000 (Katalog/Label kollabiert)."
        ),
    )
    p.add_argument("--total", type=int, default=None, help="Ein-Nutzer: Gesamt-Sessions (z.B. 550)")
    p.add_argument(
        "--additional",
        type=int,
        default=500,
        help="Ein-Nutzer: Zusätzlich zu den ursprünglichen 50 (Default 500 → 550 total)",
    )
    p.add_argument("--stem", type=str, default="leo_50_sessions", help="Dateiname ohne Endung unter data/")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--nutzer-id", type=str, default="leo_001", help="nur Ein-Nutzer-Modus")
    p.add_argument(
        "--nutzer-typ",
        type=int,
        default=3,
        choices=[0, 1, 2, 3, 4],
        help="nur Ein-Nutzer-Modus: Profil-Typ 0…4",
    )
    p.add_argument(
        "--sessions-per-user",
        type=int,
        default=50,
        help="Multi-Nutzer: Sessions pro Nutzer (Default 50 = eine Story-Runde)",
    )
    p.add_argument(
        "--users-per-type",
        type=str,
        default=None,
        help="Multi-Nutzer: z.B. 100,100,100,100 = je 100 Nutzer Typ 0–3 (× sessions-per-user Zeilen)",
    )
    p.add_argument(
        "--nutzer-id-prefix",
        type=str,
        default="user",
        help="Multi-Nutzer: Präfix für IDs {prefix}_t{TYP}_{NNN}",
    )
    p.add_argument("--no-copy-downloads", action="store_true", help="Nicht nach ~/Downloads/data spiegeln")
    return p.parse_args()


def main():
    args = parse_args()
    data_dir = Path(__file__).resolve().parent

    if args.users_per_type is not None:
        parts = [p.strip() for p in args.users_per_type.split(",") if p.strip() != ""]
        try:
            counts = tuple(int(x) for x in parts)
        except ValueError as e:
            raise SystemExit("--users-per-type: nur ganze Zahlen, komma-separiert") from e
        sessions = generate_cohort_users(
            counts,
            args.sessions_per_user,
            args.seed,
            id_prefix=args.nutzer_id_prefix,
        )
    else:
        total = args.total if args.total is not None else 50 + max(0, args.additional)
        sessions = generate_sessions(
            total,
            nutzer_id=args.nutzer_id,
            nutzer_typ=args.nutzer_typ,
            seed=args.seed,
        )

    csv_p, json_p = save_sessions(
        sessions,
        data_dir,
        args.stem,
        copy_downloads=not args.no_copy_downloads,
    )
    print_summary(sessions)
    print(f"\nCSV:  {csv_p}")
    print(f"JSON: {json_p}")


if __name__ == "__main__":
    main()
