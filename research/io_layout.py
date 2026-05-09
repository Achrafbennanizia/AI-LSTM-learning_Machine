"""
Research output layout — einheitliche Verzeichnisse für TensorBoard, Artefakte, Exporte.

Umgebung:
  GUITARAI_RESEARCH_ROOT — optionaler absoluter Pfad statt <Repo>/research_outputs/
"""
from __future__ import annotations

import json
import os
from typing import Any

TB_SCALAR_SAMPLES = 10_000_000


def get_research_root(repo_root: str) -> str:
    env = os.environ.get("GUITARAI_RESEARCH_ROOT", "").strip()
    if env:
        return os.path.normpath(os.path.expanduser(env))
    return os.path.normpath(os.path.join(repo_root, "research_outputs"))


def tensorboard_compare(root: str) -> str:
    return os.path.join(root, "tensorboard", "compare")


def tensorboard_forgetting(root: str) -> str:
    return os.path.join(root, "tensorboard", "forgetting")


def artifacts_compare(root: str) -> str:
    return os.path.join(root, "artifacts", "compare")


def artifacts_forgetting(root: str) -> str:
    return os.path.join(root, "artifacts", "forgetting")


def exports_dir(root: str) -> str:
    return os.path.join(root, "exports")


def tb_launch_all(log_dir: str) -> str:
    root = os.path.abspath(log_dir)
    return (
        f'tensorboard --logdir "{root}" '
        f"--samples_per_plugin=scalars={TB_SCALAR_SAMPLES}"
    )


def tb_launch_single(run_subdir: str) -> str:
    return (
        f'tensorboard --logdir "{os.path.abspath(run_subdir)}" '
        f"--samples_per_plugin=scalars={TB_SCALAR_SAMPLES}"
    )


def ensure_dirs(*paths: str) -> None:
    for p in paths:
        os.makedirs(p, exist_ok=True)


def write_manifest(path: str, payload: dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)


def export_results_table_csv(path: str, rows: list[dict[str, Any]]) -> None:
    """Flache Forschungstabelle als CSV (Excel/Numbers/R)."""
    import csv

    if not rows:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    keys: list[str] = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    flat = []
    for r in rows:
        line = {}
        for k in keys:
            v = r.get(k, "")
            if isinstance(v, (list, dict)):
                line[k] = json.dumps(v, default=str)
            else:
                line[k] = v
        flat.append(line)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(flat)
