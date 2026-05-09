GuitarAI — research_outputs/
==============================

Alle Ergebnisse aus research/train_compare.py und research/benchmark_forgetting.py
landen hier (oder unter GUITARAI_RESEARCH_ROOT), damit Produktions-`runs/` und
Haupt-`train.py` getrennt bleiben.

Struktur
--------

  tensorboard/compare/     ← Architekturvergleich (Unterordner = Run-ID → je Architektur)
  tensorboard/forgetting/  ← Forgetting-Benchmark (Run-ID)
  artifacts/compare/       ← comparison_*.json, run_manifest_*.json
  artifacts/forgetting/    ← forgetting_*.json, run_manifest_*.json
  exports/                 ← *.csv Tabellen für Papers / R / Excel

TensorBoard (lange Skalen, volle Punktzahl wie train.py):

  cd <Repo>
  tensorboard --logdir research_outputs/tensorboard/compare --samples_per_plugin=scalars=10000000

Nur einen Lauf (kein Overlay alter Runs):

  tensorboard --logdir "research_outputs/tensorboard/compare/20260209_153022" \\
      --samples_per_plugin=scalars=10000000

HPC: TensorBoard auf dem Login-Node starten, wo `research_outputs/` liegt; SSH-Tunnel zum Laptop.
