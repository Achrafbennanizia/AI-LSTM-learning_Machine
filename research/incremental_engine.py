"""
GuitarAI — Inkrementeller Lern-Engine
======================================
Verarbeitet Daten taskweise (z. B. Nutzer-Batches) mit EWC + Experience Replay.
DQN nutzt einen eigenen Loop (siehe research/train_compare.py).

Siehe research/README_research.md für Kontext.
"""
import torch
import torch.nn as nn
import random
from collections import deque


class EWC:
    """Elastic Weight Consolidation für inkrementelles Lernen."""

    def __init__(self, lambda_ewc=400.0):
        self.lambda_ewc = lambda_ewc
        self.fisher = {}
        self.params_star = {}
        self._n_tasks = 0

    def compute_and_update(self, models_dict, dataloader, loss_fn, device):
        all_params = {}
        for name, m in models_dict.items():
            for pname, p in m.named_parameters():
                if p.requires_grad:
                    all_params[f"{name}.{pname}"] = p

        task_fisher = {k: torch.zeros_like(p) for k, p in all_params.items()}

        n_batches = 0
        was_training = {n: m.training for n, m in models_dict.items()}
        for m in models_dict.values():
            m.eval()

        try:
            for batch in dataloader:
                for m in models_dict.values():
                    m.zero_grad(set_to_none=True)

                x, yf, yk, ykat = [b.to(device) for b in batch]
                loss = loss_fn(x, yf, yk, ykat)
                loss.backward()

                for k, p in all_params.items():
                    if p.grad is not None:
                        task_fisher[k] += p.grad.detach() ** 2

                n_batches += 1
        finally:
            for n, m in models_dict.items():
                m.train(was_training[n])

        if n_batches == 0:
            return

        for k in task_fisher:
            task_fisher[k] /= n_batches
            if k not in self.fisher:
                self.fisher[k] = task_fisher[k].clone()
            else:
                self.fisher[k] = self.fisher[k] + task_fisher[k]

        for k, p in all_params.items():
            self.params_star[k] = p.detach().clone()

        self._n_tasks += 1

    def penalty(self, models_dict):
        if not self.fisher:
            return torch.tensor(0.0, device=next(
                (p.device for m in models_dict.values() for p in m.parameters()),
                torch.device("cpu"),
            ))

        all_params = {}
        for name, m in models_dict.items():
            for pname, p in m.named_parameters():
                if p.requires_grad:
                    all_params[f"{name}.{pname}"] = p

        total = None
        device = None
        for k, p in all_params.items():
            if k in self.fisher and k in self.params_star:
                device = p.device
                term = (self.fisher[k] * (p - self.params_star[k]) ** 2).sum()
                total = term if total is None else total + term
        if total is None:
            return torch.tensor(0.0, device=device or torch.device("cpu"))
        return self.lambda_ewc * total

    @property
    def n_tasks_seen(self):
        return self._n_tasks


class ExperienceReplayBuffer:
    """Experience Replay für Anti-Forgetting (klassische Überlagerung mit neuen Batches)."""

    def __init__(self, capacity=2000):
        self.buffer = deque(maxlen=capacity)

    def add_batch(self, x, yf, yk, ykat):
        for i in range(x.size(0)):
            self.buffer.append((
                x[i].cpu(),
                yf[i].cpu(),
                yk[i].cpu(),
                ykat[i].cpu(),
            ))

    def sample(self, n):
        n = min(n, len(self.buffer))
        if n == 0:
            return None
        batch = random.sample(list(self.buffer), n)
        xs, yfs, yks, ykats = zip(*batch)
        return (
            torch.stack(xs),
            torch.stack(yfs),
            torch.stack(yks),
            torch.stack(ykats),
        )

    def __len__(self):
        return len(self.buffer)


class IncrementalTrainer:
    """Orchestriert Vortraining + inkrementelle Updates für MLP / LSTM / GRU."""

    def __init__(
        self,
        models_dict,
        optimizer,
        loss_fn,
        device,
        use_ewc=True,
        ewc_lambda=400.0,
        use_replay=True,
        replay_capacity=2000,
        replay_ratio=0.3,
        incremental_epochs=5,
        tb_writer=None,
    ):
        self.models = models_dict
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device
        self.use_ewc = use_ewc
        self.use_replay = use_replay
        self.replay_ratio = replay_ratio
        self.incremental_epochs = incremental_epochs
        self.tb_writer = tb_writer

        self.ewc = EWC(lambda_ewc=ewc_lambda) if use_ewc else None
        self.replay = ExperienceReplayBuffer(replay_capacity) if use_replay else None

        self._global_step = 0
        self._task_idx = 0

    def pretrain(self, dataloader, epochs, scheduler=None, tag="pretrain"):
        for m in self.models.values():
            m.train()

        for epoch in range(1, epochs + 1):
            epoch_loss = 0.0
            for batch in dataloader:
                x, yf, yk, ykat = [b.to(self.device) for b in batch]
                self.optimizer.zero_grad(set_to_none=True)
                loss = self.loss_fn(x, yf, yk, ykat)
                loss.backward()
                nn.utils.clip_grad_norm_(
                    [p for m in self.models.values() for p in m.parameters()], 1.0
                )
                self.optimizer.step()
                epoch_loss += loss.item()

                if self.tb_writer:
                    self.tb_writer.add_scalar(f"{tag}/loss_batch", loss.item(), self._global_step)
                self._global_step += 1

                if self.replay is not None:
                    self.replay.add_batch(x, yf, yk, ykat)

            if scheduler:
                scheduler.step(epoch_loss / max(len(dataloader), 1))

            if self.tb_writer:
                self.tb_writer.add_scalar(
                    f"{tag}/loss_epoch", epoch_loss / max(len(dataloader), 1), epoch
                )
                self.tb_writer.flush()

        if self.ewc is not None:
            self.ewc.compute_and_update(self.models, dataloader, self.loss_fn, self.device)
            if self.tb_writer:
                self.tb_writer.add_scalar("ewc/tasks_seen", self.ewc.n_tasks_seen, 0)

    def incremental_update(self, new_dataloader, task_name="task"):
        for m in self.models.values():
            m.train()

        self._task_idx += 1
        task_losses = []

        for epoch in range(self.incremental_epochs):
            for batch in new_dataloader:
                x_new, yf_new, yk_new, ykat_new = [b.to(self.device) for b in batch]

                if self.replay is not None and len(self.replay) > 0:
                    denom = 1.0 - self.replay_ratio
                    n_replay = max(1, int(x_new.size(0) * self.replay_ratio / denom))
                    replay_batch = self.replay.sample(n_replay)
                    if replay_batch is not None:
                        xr, yfr, ykr, ykatr = [b.to(self.device) for b in replay_batch]
                        x_new = torch.cat([x_new, xr], dim=0)
                        yf_new = torch.cat([yf_new, yfr], dim=0)
                        yk_new = torch.cat([yk_new, ykr], dim=0)
                        ykat_new = torch.cat([ykat_new, ykatr], dim=0)

                self.optimizer.zero_grad(set_to_none=True)
                loss = self.loss_fn(x_new, yf_new, yk_new, ykat_new)

                ewc_pen = None
                if self.ewc is not None and self.ewc.n_tasks_seen > 0:
                    ewc_pen = self.ewc.penalty(self.models)
                    loss = loss + ewc_pen

                loss.backward()
                nn.utils.clip_grad_norm_(
                    [p for m in self.models.values() for p in m.parameters()], 1.0
                )
                self.optimizer.step()
                task_losses.append(loss.item())

                if self.tb_writer:
                    self.tb_writer.add_scalar(
                        f"incremental/{task_name}/loss", loss.item(), self._global_step
                    )
                    if ewc_pen is not None and isinstance(ewc_pen, torch.Tensor):
                        self.tb_writer.add_scalar(
                            f"incremental/{task_name}/ewc_penalty",
                            ewc_pen.item(),
                            self._global_step,
                        )
                self._global_step += 1

        if self.replay is not None:
            for batch in new_dataloader:
                x, yf, yk, ykat = [b.to(self.device) for b in batch]
                self.replay.add_batch(x, yf, yk, ykat)

        if self.ewc is not None:
            self.ewc.compute_and_update(
                self.models, new_dataloader, self.loss_fn, self.device
            )
            if self.tb_writer:
                self.tb_writer.add_scalar(
                    "ewc/tasks_seen", self.ewc.n_tasks_seen, self._task_idx
                )
                self.tb_writer.flush()

        return sum(task_losses) / max(len(task_losses), 1)

    def evaluate(self, dataloader, is_mlp=False):
        for m in self.models.values():
            m.eval()

        total_loss = 0.0
        with torch.no_grad():
            for batch in dataloader:
                x, yf, yk, ykat = [b.to(self.device) for b in batch]
                loss = self.loss_fn(x, yf, yk, ykat)
                total_loss += loss.item()

        for m in self.models.values():
            m.train()

        return {"loss": total_loss / max(len(dataloader), 1)}
