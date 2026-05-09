#!/usr/bin/env bash
# Run on HiPer login node from repo root: bash scripts/hpc_submit_two_jobs.sh
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

sed \
  -e "s/@@SLURM_TIME_LIMIT@@/04:00:00/" \
  -e "s/@@SLURM_PARTITION@@/compute/" \
  -e "s/@@SLURM_ACCOUNT@@/l_iui_hetapken_fki_ss26/" \
  -e "s/@@SLURM_GRES@@/gpu:1g.5gb:1/" \
  -e "s/@@SLURM_MEM@@/16G/" \
  -e "s/@@SLURM_CPUS_PER_TASK@@/4/" \
  -e "s/@@SLURM_MAIL_TYPE@@/END,FAIL/" \
  -e "s/@@SLURM_MAIL_USER@@/a.bennaniziatni@hs-osnabrueck.de/" \
  job_research.slurm.template > job_research.slurm
chmod +x job_research.slurm

python3 <<'PY'
from pathlib import Path
p = Path("job.slurm")
t = p.read_text()
old = """VENV_PY=\"${SLURM_SUBMIT_DIR}/.venv/bin/python\"

# ── Verzeichnisse ──────────────────────────────────────────────────────────────
mkdir -p logs checkpoints runs

# ── Voraussetzungen prüfen ─────────────────────────────────────────────────────
if [[ ! -f \"${DATA_CSV}\" ]]; then"""
new = """# ── Verzeichnisse ──────────────────────────────────────────────────────────────
mkdir -p logs checkpoints runs

# ── Voraussetzungen prüfen ─────────────────────────────────────────────────────
if [[ ! -f \"${DATA_CSV}\" ]]; then"""
if old not in t:
    raise SystemExit("job.slurm pattern not found")
t = t.replace(old, new, 1)
ins = """if [[ -x \"${SLURM_SUBMIT_DIR}/guitarai_env/bin/python\" ]]; then
  VENV_PY=\"${SLURM_SUBMIT_DIR}/guitarai_env/bin/python\"
elif [[ -x \"${SLURM_SUBMIT_DIR}/.venv/bin/python\" ]]; then
  VENV_PY=\"${SLURM_SUBMIT_DIR}/.venv/bin/python\"
else
  echo \"FEHLER: Weder guitarai_env noch .venv mit python gefunden.\" >&2
  exit 1
fi
echo \"Python: ${VENV_PY}\"

"""
marker = "mkdir -p logs checkpoints runs\n\n# ── Voraussetzungen prüfen"
if marker not in t:
    raise SystemExit("marker2 not found")
t = t.replace(marker, "mkdir -p logs checkpoints runs\n\n" + ins + "# ── Voraussetzungen prüfen", 1)
old_check = """if [[ ! -x \"${VENV_PY}\" ]]; then
    echo \"FEHLER: ${VENV_PY} fehlt.\" >&2
    echo \"Venv einrichten: python3 -m venv .venv && .venv/bin/pip install -r requirements.txt\" >&2
    exit 1
fi

"""
t = t.replace(old_check, "", 1)
p.write_text(t)
print("Patched job.slurm venv selection.")
PY

echo "--- sbatch ---"
J1=$(sbatch --parsable job.slurm)
J2=$(sbatch --parsable job_research.slurm)
echo "job.slurm id: $J1"
echo "job_research.slurm id: $J2"
squeue -u "${USER}" | head -15
