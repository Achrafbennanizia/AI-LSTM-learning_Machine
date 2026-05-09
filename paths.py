"""Resolve training CSV: env overrides, then repo data/, then ~/Downloads/data.

Env (höchste Priorität):
  GUITARAI_DATA — voller Pfad zu einer CSV-Datei; falls Verzeichnis: Datei leo_50_sessions.csv darin.
  GUITARAI_DATA_DIR — nur Verzeichnis; gleiche Datei wie oben.
"""

from __future__ import annotations

import os
from pathlib import Path

_LEO_FILENAME = "leo_50_sessions.csv"


def resolve_training_csv() -> str:
    env = (
        os.environ.get("GUITARAI_DATA", "").strip()
        or os.environ.get("GUITARAI_DATA_DIR", "").strip()
    )
    if env:
        ep = Path(env).expanduser()
        if ep.is_file():
            return str(ep.resolve())
        if ep.is_dir():
            p = ep / _LEO_FILENAME
            if p.is_file():
                return str(p.resolve())

    repo = Path(__file__).resolve().parent / "data" / _LEO_FILENAME

    candidates = (
        repo,
        Path.home() / "Downloads" / "data" / _LEO_FILENAME,
        Path("/downloads/data") / _LEO_FILENAME,
    )
    for p in candidates:
        if p.is_file():
            return str(p.resolve())

    return str(repo)
