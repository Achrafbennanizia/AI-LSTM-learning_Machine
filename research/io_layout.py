"""Ausgabe-Pfade unter research_outputs/."""
import os

from training_common import tb_cmd as _tb_cmd

TB_SAMPLES = 10_000_000


def research_root(repo_root):
    return os.path.join(repo_root, "research_outputs")


def compare_tb(repo_root):
    return os.path.join(research_root(repo_root), "tensorboard", "compare")


def forgetting_tb(repo_root):
    return os.path.join(research_root(repo_root), "tensorboard", "forgetting")


def compare_artifacts(repo_root):
    return os.path.join(research_root(repo_root), "artifacts", "compare")


def forgetting_artifacts(repo_root):
    return os.path.join(research_root(repo_root), "artifacts", "forgetting")


def tb_cmd(log_dir):
    return _tb_cmd(log_dir)


def mkdir(*paths):
    for path in paths:
        os.makedirs(path, exist_ok=True)
