"""Vortraining + inkrementelle Tasks mit EWC und Experience Replay."""
import random
from collections import deque

import torch
import torch.nn as nn


# [AI-assisted ~25%] tool=Cursor Composer | prompt=docs/ai_prompts/incremental_engine.md
class EWC:
    def __init__(self, lambda_ewc=400.0):
        self.lambda_ewc = lambda_ewc
        self.fisher = {}
        self.params_star = {}

    def update(self, models, dataloader, loss_fn, device):
        params = {}
        for name, model in models.items():
            for pname, p in model.named_parameters():
                params[f"{name}.{pname}"] = p

        fisher = {k: torch.zeros_like(p) for k, p in params.items()}

        for batch in dataloader:
            for model in models.values():
                model.zero_grad()
            x, yf, yk, ykat = [t.to(device) for t in batch]
            loss_fn(x, yf, yk, ykat).backward()
            for k, p in params.items():
                fisher[k] += p.grad.detach() ** 2

        n = len(dataloader)
        for k in fisher:
            fisher[k] /= n
            if k in self.fisher:
                self.fisher[k] = self.fisher[k] + fisher[k]
            else:
                self.fisher[k] = fisher[k].clone()
            self.params_star[k] = params[k].detach().clone()

    def penalty(self, models):
        params = {}
        for name, model in models.items():
            for pname, p in model.named_parameters():
                params[f"{name}.{pname}"] = p

        total = 0.0
        for k, p in params.items():
            total += (self.fisher[k] * (p - self.params_star[k]) ** 2).sum()
        return self.lambda_ewc * total


# --- end AI-assisted (EWC) ---


class ReplayBuffer:
    def __init__(self, capacity=2000):
        self.buffer = deque(maxlen=capacity)

    def add(self, x, yf, yk, ykat):
        for i in range(x.size(0)):
            self.buffer.append((x[i].cpu(), yf[i].cpu(), yk[i].cpu(), ykat[i].cpu()))

    def sample(self, n):
        n = min(max(1, n), len(self.buffer))
        batch = random.sample(self.buffer, n)
        xs, yfs, yks, ykats = zip(*batch)
        return torch.stack(xs), torch.stack(yfs), torch.stack(yks), torch.stack(ykats)


class IncrementalTrainer:
    def __init__(self, models, optimizer, loss_fn, device, ewc_lambda=400.0, incremental_epochs=5):
        self.models = models
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device
        self.incremental_epochs = incremental_epochs
        self.ewc = EWC(ewc_lambda)
        self.replay = ReplayBuffer()
        self.use_ewc = True

    def pretrain(self, dataloader, epochs, on_epoch=None):
        for model in self.models.values():
            model.train()

        for epoch in range(1, epochs + 1):
            epoch_loss = 0.0
            for batch in dataloader:
                x, yf, yk, ykat = [t.to(self.device) for t in batch]
                self.optimizer.zero_grad()
                loss = self.loss_fn(x, yf, yk, ykat)
                loss.backward()
                nn.utils.clip_grad_norm_(
                    [p for m in self.models.values() for p in m.parameters()], 1.0
                )
                self.optimizer.step()
                epoch_loss += loss.item()
                self.replay.add(x, yf, yk, ykat)

            if on_epoch:
                on_epoch(epoch, epoch_loss / len(dataloader))

        self.ewc.update(self.models, dataloader, self.loss_fn, self.device)

    # [AI-assisted] replay batch merge + EWC penalty — prompt=docs/ai_prompts/incremental_engine.md
    def incremental_update(self, dataloader):
        for model in self.models.values():
            model.train()

        task_losses = []
        for _ in range(self.incremental_epochs):
            for batch in dataloader:
                x, yf, yk, ykat = [t.to(self.device) for t in batch]

                n_replay = max(1, x.size(0) // 3)
                xr, yfr, ykr, ykatr = self.replay.sample(n_replay)
                xr, yfr, ykr, ykatr = [t.to(self.device) for t in (xr, yfr, ykr, ykatr)]
                x = torch.cat([x, xr])
                yf = torch.cat([yf, yfr])
                yk = torch.cat([yk, ykr])
                ykat = torch.cat([ykat, ykatr])

                self.optimizer.zero_grad()
                loss = self.loss_fn(x, yf, yk, ykat)
                if self.use_ewc:
                    loss = loss + self.ewc.penalty(self.models)
                loss.backward()
                nn.utils.clip_grad_norm_(
                    [p for m in self.models.values() for p in m.parameters()], 1.0
                )
                self.optimizer.step()
                task_losses.append(loss.item())

        for batch in dataloader:
            x, yf, yk, ykat = [t.to(self.device) for t in batch]
            self.replay.add(x, yf, yk, ykat)

        self.ewc.update(self.models, dataloader, self.loss_fn, self.device)
        return sum(task_losses) / len(task_losses)
    # --- end AI-assisted (incremental_update) ---
