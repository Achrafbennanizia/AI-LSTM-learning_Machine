# Research outputs

Synced experiment results and TensorBoard roots for the architecture comparison and forgetting benchmark.

## Layout

```
research_outputs/
  artifacts/
    compare/     comparison_results.json — final test metrics per architecture
    forgetting/  forgetting_task1.json  — Task-1 retention benchmark
  tensorboard/
    compare/     Scalar logs from train_compare.py (sync after cluster runs)
    forgetting/  Scalar logs from benchmark_forgetting.py
```

## TensorBoard (local or via SSH tunnel)

```bash
tensorboard --logdir research_outputs/tensorboard --samples_per_plugin=scalars=10000000
```

After a cluster run, sync logs from your remote project directory (example):

```bash
rsync -avz REMOTE_HOST:~/path/to/project/research_outputs/tensorboard/ research_outputs/tensorboard/
```

Set `REMOTE_HOST` to your SSH target; keep cluster paths and credentials out of the repo.

## AI code provenance

See `docs/AI_CODE_ATTRIBUTION.md` for which modules were AI-assisted and which prompts were used.
