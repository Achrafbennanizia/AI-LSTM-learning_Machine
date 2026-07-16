"""
Task-1-Retention: Katalog-Acc auf Task 1 nach jedem neuen Task.
Vergessen = Abfall der acc_task-Kurve (step 0 → letzter step).

  python research/benchmark_forgetting.py --data data/leo_sessions_v3.csv
"""
import argparse
import json
import os
import sys
from datetime import datetime

import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO_ROOT)
sys.path.insert(0, os.path.join(_REPO_ROOT, "research"))

from io_layout import forgetting_artifacts, forgetting_tb, mkdir
from models.katalog_matcher import get_katalog_matrix
from incremental_engine import IncrementalTrainer
from training_common import (
    SessionDataset,
    build_models,
    evaluate,
    make_loss_fn,
    split_pretrain_inc_test,
)


def run_one(arch_name, use_ewc, df_pre, task_dfs, device, args, writer):
    arch = arch_name.lower()
    tag = f"{arch_name}_{'ewc' if use_ewc else 'noewc'}"
    ds_args = {"seq_len": args.seq_len}

    models = build_models(arch, device, args.hidden)
    optimizer = torch.optim.Adam([p for m in models.values() for p in m.parameters()], lr=args.lr)
    loss_fn = make_loss_fn(models, get_katalog_matrix().to(device))

    trainer = IncrementalTrainer(
        models, optimizer, loss_fn, device,
        ewc_lambda=args.ewc_lambda,
        incremental_epochs=args.incremental_epochs,
    )
    trainer.use_ewc = use_ewc

    dl_pre = DataLoader(SessionDataset(df_pre, **ds_args), batch_size=args.batch_size, shuffle=True)
    trainer.pretrain(dl_pre, args.pretrain_epochs)

    dl_task1 = DataLoader(SessionDataset(task_dfs[0], **ds_args), batch_size=args.batch_size)
    katalog_matrix = get_katalog_matrix().to(device)
    baseline_metrics = evaluate(models, dl_task1, katalog_matrix, device)
    baseline = baseline_metrics["acc_katalog"]
    writer.add_scalar(f"acc_task/{tag}", baseline, 0)

    for i, df_task in enumerate(task_dfs):
        dl_task = DataLoader(SessionDataset(df_task, **ds_args), batch_size=args.batch_size, shuffle=True)
        trainer.incremental_update(dl_task)

        metrics = evaluate(models, dl_task1, katalog_matrix, device)
        acc = metrics["acc_katalog"]

        writer.add_scalar(f"acc_task/{tag}", acc, i + 1)
        writer.flush()

    return {
        "arch": tag,
        "baseline_acc": baseline,
        "final_acc": acc,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--pretrain_epochs", type=int, default=25)
    parser.add_argument("--n_tasks", type=int, default=6)
    parser.add_argument("--incremental_epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--seq_len", type=int, default=5)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--ewc_lambda", type=float, default=400.0)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log_dir = forgetting_tb(_REPO_ROOT)
    save_dir = forgetting_artifacts(_REPO_ROOT)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    mkdir(log_dir, save_dir, os.path.join(log_dir, run_id))

    df = pd.read_csv(args.data)
    df_pre, df_inc, _ = split_pretrain_inc_test(df)

    users = df_inc["nutzer_id"].unique()
    chunk = len(users) // args.n_tasks
    task_dfs = [df_inc[df_inc["nutzer_id"].isin(users[i:i + chunk])] for i in range(0, len(users), chunk)]
    writer = SummaryWriter(os.path.join(log_dir, run_id), flush_secs=5)
    configs = [("RNN", False), ("RNN", True), ("LSTM", True), ("GRU", True)]

    results = []
    for arch, ewc in configs:
        results.append(run_one(arch, ewc, df_pre, task_dfs, device, args, writer))
    writer.close()

    out = os.path.join(save_dir, f"task1_{run_id}.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
