#!/usr/bin/env python3
"""
Schreibt ./job.slurm aus job.slurm.template + .env.

Keine echte Projekt-Mail/Accounts im Repo: nur Platzhalter in der Template +
persönliche Werte in `.env` (gitignored).

Usage:
  cp .env.example .env   # anpassen
  python3 scripts/render_job_slurm.py [--env path] [--template path] [-o job.slurm]
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path
import shlex
import sys

ROOT = Path(__file__).resolve().parents[1]


def load_dot_env(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    raw = path.read_text(encoding="utf-8")
    for line in raw.splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip()
        val = val.split("#", 1)[0].strip()
        if not key:
            continue
        if (val.startswith('"') and val.endswith('"')) or (
            val.startswith("'") and val.endswith("'")
        ):
            val = val[1:-1]
        out[key] = val
    return out


_KEYS = frozenset(
    {
        "SLURM_TIME_LIMIT",
        "SLURM_PARTITION",
        "SLURM_ACCOUNT",
        "SLURM_GRES",
        "SLURM_MEM",
        "SLURM_CPUS_PER_TASK",
        "SLURM_MAIL_TYPE",
        "SLURM_MAIL_USER",
    }
)


def main() -> int:
    p = argparse.ArgumentParser(description="Render job.slurm from template + .env")
    p.add_argument("--env", type=Path, default=ROOT / ".env")
    p.add_argument("--template", type=Path, default=ROOT / "job.slurm.template")
    p.add_argument("-o", "--output", type=Path, default=ROOT / "job.slurm")
    args = p.parse_args()

    if not args.env.is_file():
        print(f"Missing {args.env} — copy from .env.example", file=sys.stderr)
        return 1
    env = load_dot_env(args.env)
    missing = sorted(k for k in _KEYS if not env.get(k))
    if missing:
        print(f".env missing or empty keys: {', '.join(missing)}", file=sys.stderr)
        return 1

    text = args.template.read_text(encoding="utf-8")
    for k in sorted(_KEYS, key=len, reverse=True):
        needle = "@@%s@@" % k
        if needle not in text:
            print(f"Template missing {needle}", file=sys.stderr)
            return 1
        text = text.replace(needle, env[k])

    leftover = set(re.findall(r"@@([A-Z0-9_]+)@@", text))
    if leftover - _KEYS:
        print(f"Unknown @@placeholders@@ left in template: {leftover}", file=sys.stderr)
        return 1

    acc = env["SLURM_ACCOUNT"]
    fb = env.get("SLURM_JOB_ACCOUNT_FALLBACK", "").strip()

    injections = []
    injections.append("")
    injections.append("# --- aus .env bei `render_job_slurm.py` eingefügt ---")
    injections.append(f"export SLURM_ACCOUNT={shlex.quote(acc)}")
    injections.append(f"export SLURM_JOB_ACCOUNT_FALLBACK={shlex.quote(fb)}")
    injections.append("# ---")
    injections.append("")

    marker = "set -euo pipefail"
    if marker not in text:
        print("Template must contain `set -euo pipefail` for injection site", file=sys.stderr)
        return 1
    text = text.replace(marker, marker + "\n" + "\n".join(injections), 1)

    args.output.write_text(text, encoding="utf-8")
    args.output.chmod(args.output.stat().st_mode | 0o111)
    print(f"Wrote {args.output}")

    research_tpl = ROOT / "job_research.slurm.template"
    research_out = ROOT / "job_research.slurm"
    if research_tpl.is_file():
        rt = research_tpl.read_text(encoding="utf-8")
        for k in sorted(_KEYS, key=len, reverse=True):
            rt = rt.replace("@@%s@@" % k, env[k])
        leftover_r = set(re.findall(r"@@([A-Z0-9_]+)@@", rt))
        if leftover_r - _KEYS:
            print(f"Research template unknown placeholders: {leftover_r}", file=sys.stderr)
            return 1
        inj_r = [
            "",
            "# --- aus .env bei `render_job_slurm.py` eingefügt ---",
            f"export SLURM_ACCOUNT={shlex.quote(acc)}",
            f"export SLURM_JOB_ACCOUNT_FALLBACK={shlex.quote(fb)}",
            "# ---",
            "",
        ]
        rt = rt.replace(marker, marker + "\n" + "\n".join(inj_r), 1)
        research_out.write_text(rt, encoding="utf-8")
        research_out.chmod(research_out.stat().st_mode | 0o111)
        print(f"Wrote {research_out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
