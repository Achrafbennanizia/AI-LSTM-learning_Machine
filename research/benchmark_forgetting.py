"""
GuitarAI — Catastrophic Forgetting Benchmark
=============================================
Misst explizit wie viel jede Architektur von früheren Tasks vergisst.

Methode (Backward Transfer):
  - Trainiere auf Task 1, messe Acc(Task1) = A
  - Trainiere auf Task 2, messe Acc(Task1) erneut = B
  - Forgetting = A - B  (positiv = Vergessen)

Diese Messung ist der wissenschaftliche Standard für
inkrementelles Lernen und sollte in deinem Bericht erscheinen.

Output:
  - forgetting_results.json  — Rohdaten pro Architektur und Task
  - TensorBoard: forgetting/ Kurven für alle Architekturen

Usage:
  python research/benchmark_forgetting.py [--data ...]
"""
import os, sys, json, argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

_RESEARCH_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_RESEARCH_DIR)
for _p in (_REPO_ROOT, _RESEARCH_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from paths import resolve_training_csv
from io_layout import (
    TB_SCALAR_SAMPLES,
    artifacts_forgetting,
    export_results_table_csv,
    exports_dir,
    ensure_dirs,
    get_research_root,
    tb_launch_all,
    tb_launch_single,
    tensorboard_forgetting,
    write_manifest,
)
from train_compare import (
    build_mlp_models, build_seq_models,
    FlatDataset, SessionDataset,
    make_loss_fn, evaluate, split_by_user,
)
from models.katalog_matcher import get_katalog_matrix
from incremental_engine import IncrementalTrainer


def run_forgetting_benchmark(arch_name, use_ewc, df_pretrain, tasks_dfs,
                             df_test, katalog_matrix, device, args, writer):
    """Misst Forgetting: Acc auf Task 1 nach jedem weiteren Training."""
    is_mlp = (arch_name == "MLP")
    if is_mlp:
        models = build_mlp_models(device, hidden=args.hidden)
    elif arch_name == "LSTM":
        models = build_seq_models("lstm", device, hidden=args.hidden)
    elif arch_name == "GRU":
        models = build_seq_models("gru", device, hidden=args.hidden)
    else:
        return None

    DatasetClass = FlatDataset if is_mlp else SessionDataset
    ds_kw = {} if is_mlp else {"seq_len": args.seq_len}

    all_params = [p for m in models.values() for p in m.parameters()]
    optimizer  = torch.optim.Adam(all_params, lr=args.lr)
    loss_fn    = make_loss_fn(models, katalog_matrix, device, is_mlp=is_mlp)

    run_tag = f"{arch_name}_{'ewc' if use_ewc else 'noewc'}"

    trainer = IncrementalTrainer(
        models_dict=models, optimizer=optimizer, loss_fn=loss_fn,
        device=device, use_ewc=use_ewc, ewc_lambda=args.ewc_lambda,
        use_replay=True, incremental_epochs=args.incremental_epochs,
    )

    # Vortraining
    ds_pre = DatasetClass(df_pretrain, **ds_kw)
    dl_pre = DataLoader(ds_pre, batch_size=args.batch_size, shuffle=True)
    trainer.pretrain(dl_pre, epochs=args.pretrain_epochs)

    # Acc auf Task 1 BEVOR inkrementelles Training
    ds_task1 = DatasetClass(tasks_dfs[0], **ds_kw)
    dl_task1 = DataLoader(ds_task1, batch_size=args.batch_size, shuffle=False)
    baseline_acc = evaluate(models, dl_task1, katalog_matrix, device, is_mlp)["acc_katalog"]

    forgetting_curve = [0.0]  # Forgetting nach Task 0 = 0
    acc_on_task1     = [baseline_acc]

    # Inkrementell durch Tasks iterieren und Acc auf Task 1 messen
    for task_idx, df_task in enumerate(tasks_dfs):
        ds_task = DatasetClass(df_task, **ds_kw)
        dl_task = DataLoader(ds_task, batch_size=args.batch_size, shuffle=True)
        trainer.incremental_update(dl_task, task_name=f"task_{task_idx}")

        # Wie gut ist das Modell noch auf Task 1?
        acc_now = evaluate(models, dl_task1, katalog_matrix, device, is_mlp)["acc_katalog"]
        acc_on_task1.append(acc_now)
        forgetting = baseline_acc - acc_now
        forgetting_curve.append(forgetting)

        print(f"  [{run_tag}] Nach Task {task_idx+1}: "
              f"Acc@Task1 = {acc_now:.1f}% | Forgetting = {forgetting:+.1f}%")

        if writer:
            writer.add_scalar(f"forgetting/{run_tag}", forgetting, task_idx + 1)
            writer.add_scalar(f"acc_task1/{run_tag}",  acc_now,    task_idx + 1)

    return {
        "arch":            run_tag,
        "baseline_acc":    baseline_acc,
        "final_acc_task1": acc_on_task1[-1],
        "max_forgetting":  max(forgetting_curve),
        "avg_forgetting":  sum(forgetting_curve) / len(forgetting_curve),
        "forgetting_curve": forgetting_curve,
        "acc_history":     acc_on_task1,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",               default=None, help="CSV (Default: paths.resolve_training_csv)")
    parser.add_argument("--pretrain_epochs",    type=int,   default=30)
    parser.add_argument("--n_tasks",            type=int,   default=8)
    parser.add_argument("--incremental_epochs", type=int,   default=3)
    parser.add_argument("--batch_size",         type=int,   default=32)
    parser.add_argument("--lr",                 type=float, default=5e-4)
    parser.add_argument("--seq_len",            type=int,   default=5)
    parser.add_argument("--hidden",             type=int,   default=64)
    parser.add_argument("--ewc_lambda",         type=float, default=400.0)
    parser.add_argument(
        "--research_root",
        default=None,
        help="Wie train_compare: Basis für research_outputs/ (oder GUITARAI_RESEARCH_ROOT).",
    )
    parser.add_argument(
        "--log_dir",
        default=None,
        help="TensorBoard Forgetting (Standard: …/research_outputs/tensorboard/forgetting)",
    )
    parser.add_argument(
        "--save_dir",
        default=None,
        help="JSON-Artefakte (Standard: …/research_outputs/artifacts/forgetting)",
    )
    args = parser.parse_args()

    if args.research_root:
        research_base = os.path.normpath(
            args.research_root
            if os.path.isabs(args.research_root)
            else os.path.join(_REPO_ROOT, args.research_root)
        )
    else:
        research_base = get_research_root(_REPO_ROOT)

    if args.log_dir is None:
        args.log_dir = tensorboard_forgetting(research_base)
    elif not os.path.isabs(args.log_dir):
        args.log_dir = os.path.normpath(os.path.join(_REPO_ROOT, args.log_dir))

    if args.save_dir is None:
        args.save_dir = artifacts_forgetting(research_base)
    elif not os.path.isabs(args.save_dir):
        args.save_dir = os.path.normpath(os.path.join(_REPO_ROOT, args.save_dir))

    args.exports_dir = exports_dir(research_base)
    ensure_dirs(args.log_dir, args.save_dir, args.exports_dir)

    if args.data is None:
        args.data = resolve_training_csv()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    df = pd.read_csv(args.data)
    df_pretrain, df_incremental, df_test = split_by_user(df, 0.5, 0.15)

    # Tasks aus inkrementellen Nutzern bilden
    inc_users = df_incremental["nutzer_id"].unique()
    task_size = max(1, len(inc_users) // args.n_tasks)
    tasks_dfs = [
        df_incremental[df_incremental["nutzer_id"].isin(inc_users[i:i+task_size])]
        for i in range(0, len(inc_users), task_size)
    ][:args.n_tasks]
    tasks_dfs = [t for t in tasks_dfs if len(t) > 0]
    if not tasks_dfs:
        raise SystemExit(
            "benchmark_forgetting: keine Task-Teilmengen — zu wenige inkrementelle Nutzer "
            "(mehr Daten / kleineres --n_tasks)."
        )

    print(f"Tasks: {len(tasks_dfs)} | Nutzer pro Task: ~{task_size}")

    print(f"Research-Output-Basis: {research_base}")
    print(f"TensorBoard (alle Forgetting-Läufe): {tb_launch_all(args.log_dir)}")

    katalog_matrix = get_katalog_matrix().to(device)
    run_id  = datetime.now().strftime("%Y%m%d_%H%M%S")
    tb_dir  = os.path.join(args.log_dir, run_id)
    ensure_dirs(tb_dir)
    writer  = SummaryWriter(log_dir=tb_dir, flush_secs=5, max_queue=100)
    print(f"TensorBoard (dieser Run): {tb_launch_single(tb_dir)}")

    writer.add_text(
        "run/summary",
        json.dumps(
            {
                "run_id": run_id,
                "research_base": os.path.abspath(research_base),
                "data_csv": os.path.abspath(str(args.data)),
                "tensorboard_root": os.path.abspath(args.log_dir),
                "tb_this_run": os.path.abspath(tb_dir),
                "artifacts": os.path.abspath(args.save_dir),
                "exports": os.path.abspath(args.exports_dir),
                "samples_per_plugin_scalars": TB_SCALAR_SAMPLES,
                "n_tasks": len(tasks_dfs),
                "rows": int(len(df)),
                "users": int(df["nutzer_id"].nunique()),
            },
            indent=2,
            default=str,
        ),
        0,
    )
    writer.flush()

    configs = [
        ("MLP",  False),   # MLP ohne EWC — Baseline
        ("MLP",  True),    # MLP mit EWC
        ("LSTM", True),    # LSTM mit EWC
        ("GRU",  True),    # GRU mit EWC
    ]

    all_results = []
    for arch, ewc in configs:
        print(f"\n{'='*50}")
        print(f"Forgetting Benchmark: {arch} ({'EWC' if ewc else 'kein EWC'})")
        result = run_forgetting_benchmark(
            arch, ewc, df_pretrain, tasks_dfs, df_test,
            katalog_matrix, device, args, writer,
        )
        if result:
            all_results.append(result)

    for i, r in enumerate(all_results):
        writer.add_scalar("summary/max_forgetting", float(r["max_forgetting"]), i)
        writer.add_scalar("summary/avg_forgetting", float(r["avg_forgetting"]), i)
        writer.add_scalar("summary/baseline_acc_task1", float(r["baseline_acc"]), i)
    writer.flush()

    # Ergebnistabelle
    print("\n" + "="*60)
    print("FORGETTING BENCHMARK — ERGEBNISSE")
    print("="*60)
    print(f"{'Architektur':<22} {'Basis-Acc':>10} {'Final-Acc':>10} "
          f"{'Max Forget':>11} {'Avg Forget':>11}")
    print("-"*65)
    for r in all_results:
        print(f"{r['arch']:<22} {r['baseline_acc']:>9.1f}% "
              f"{r['final_acc_task1']:>9.1f}% "
              f"{r['max_forgetting']:>+10.1f}% "
              f"{r['avg_forgetting']:>+10.1f}%")

    os.makedirs(args.save_dir, exist_ok=True)
    out = os.path.join(args.save_dir, f"forgetting_{run_id}.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nGespeichert: {out}")

    csv_path = os.path.join(args.exports_dir, f"forgetting_summary_{run_id}.csv")
    export_results_table_csv(csv_path, all_results)
    print(f"CSV:           {csv_path}")

    write_manifest(
        os.path.join(args.save_dir, f"run_manifest_{run_id}.json"),
        {
            "run_id": run_id,
            "research_base": os.path.abspath(research_base),
            "forgetting_json": os.path.abspath(out),
            "forgetting_csv": os.path.abspath(csv_path),
            "tensorboard_run": os.path.abspath(tb_dir),
        },
    )

    writer.close()
    print(f"\nTensorBoard (dieser Lauf): {tb_launch_single(tb_dir)}")


if __name__ == "__main__":
    main()
