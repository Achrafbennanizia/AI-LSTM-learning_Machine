"""
Architekturvergleich: M1 = RNN / LSTM / GRU, M2+M3 = MLP.

  python research/train_compare.py --data data/leo_sessions_v3.csv
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

from io_layout import compare_artifacts, compare_tb, mkdir
from models.katalog_matcher import get_katalog_matrix
from incremental_engine import IncrementalTrainer
from training_common import (
    SessionDataset,
    build_models,
    evaluate,
    make_loss_fn,
    split_pretrain_inc_test,
)


def train_one(label, arch, use_ewc, df_pre, df_inc, df_test, katalog_matrix, device, args):
    ds_args = {"seq_len": args.seq_len}

    models = build_models(arch, device, args.hidden)
    optimizer = torch.optim.Adam([p for m in models.values() for p in m.parameters()], lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    loss_fn = make_loss_fn(models, katalog_matrix)

    tag = label.lower().replace(" ", "_")
    tb_path = os.path.join(args.log_dir, args.run_name, tag)
    mkdir(tb_path)
    writer = SummaryWriter(tb_path, flush_secs=5)

    trainer = IncrementalTrainer(
        models, optimizer, loss_fn, device,
        ewc_lambda=args.ewc_lambda,
        incremental_epochs=args.incremental_epochs,
    )
    trainer.use_ewc = use_ewc

    dl_pre = DataLoader(SessionDataset(df_pre, **ds_args), batch_size=args.batch_size, shuffle=True)
    dl_test = DataLoader(SessionDataset(df_test, **ds_args), batch_size=args.batch_size)

    def on_pretrain_epoch(epoch, train_loss):
        scheduler.step(train_loss)

    trainer.pretrain(dl_pre, args.pretrain_epochs, on_epoch=on_pretrain_epoch)

    val = evaluate(models, dl_test, katalog_matrix, device)
    writer.add_scalar("incremental/acc_katalog", val["acc_katalog"], 0)
    writer.add_scalar("incremental/loss", val["loss"], 0)

    users = df_inc["nutzer_id"].unique()
    chunk = len(users) // args.n_incremental_tasks
    tasks = [users[i:i + chunk] for i in range(0, len(users), chunk)]

    for i, task_users in enumerate(tasks):
        df_task = df_inc[df_inc["nutzer_id"].isin(task_users)]
        dl_task = DataLoader(SessionDataset(df_task, **ds_args), batch_size=args.batch_size, shuffle=True)
        train_loss = trainer.incremental_update(dl_task)

        val = evaluate(models, dl_test, katalog_matrix, device)

        writer.add_scalar("incremental/training_loss", train_loss, i + 1)
        writer.add_scalar("incremental/acc_katalog", val["acc_katalog"], i + 1)
        writer.add_scalar("incremental/loss", val["loss"], i + 1)
        writer.flush()

    writer.close()
    return {
        "arch": label,
        "acc_katalog": val["acc_katalog"],
        "loss": val["loss"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--pretrain_epochs", type=int, default=35)
    parser.add_argument("--n_incremental_tasks", type=int, default=8)
    parser.add_argument("--incremental_epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--seq_len", type=int, default=5)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--ewc_lambda", type=float, default=400.0)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.log_dir = compare_tb(_REPO_ROOT)
    args.save_dir = compare_artifacts(_REPO_ROOT)
    args.run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    mkdir(args.log_dir, args.save_dir)

    df = pd.read_csv(args.data)
    df_pre, df_inc, df_test = split_pretrain_inc_test(df)

    ds_args = {"seq_len": args.seq_len}
    data_info = {
        "csv_rows": len(df),
        "users_total": int(df["nutzer_id"].nunique()),
        "users_pretrain": int(df_pre["nutzer_id"].nunique()),
        "users_incremental": int(df_inc["nutzer_id"].nunique()),
        "users_test": int(df_test["nutzer_id"].nunique()),
        "training_samples_pretrain": len(SessionDataset(df_pre, **ds_args)),
        "training_samples_test": len(SessionDataset(df_test, **ds_args)),
        "seq_len": args.seq_len,
        "features": 17,
    }
    katalog_matrix = get_katalog_matrix().to(device)
    runs = [
        ("RNN ohne EWC", "rnn", False),
        ("RNN mit EWC", "rnn", True),
        ("LSTM mit EWC", "lstm", True),
        ("GRU mit EWC", "gru", True),
    ]

    results = []
    for label, arch, ewc in runs:
        results.append(train_one(
            label, arch, ewc, df_pre, df_inc, df_test, katalog_matrix, device, args,
        ))

    out = os.path.join(args.save_dir, f"comparison_{args.run_name}.json")
    with open(out, "w") as f:
        json.dump({"data": data_info, "results": results}, f, indent=2)


if __name__ == "__main__":
    main()
