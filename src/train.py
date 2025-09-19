from __future__ import annotations
import json, math, os, gc, random, time
from pathlib import Path
from typing import Dict, List

import torch
from torch import Tensor
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from accelerate import Accelerator
from trl import DPOTrainer, DPOConfig

from datasets import load_from_disk, DatasetDict
from tqdm import tqdm
import wandb, psutil, yaml

RESULTS_DIR = Path(".research/iteration1")
RESULTS_DIR.mkdir(exist_ok=True, parents=True)


def set_seeds(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def dpo_loss(delta: Tensor, beta: float) -> Tensor:
    return -torch.nn.functional.logsigmoid(beta * delta)


def discopop_loss(delta: Tensor, beta: float = 1.0, tau: float = 0.1) -> Tensor:
    p = torch.sigmoid(beta * delta)
    weight = 1.0 - torch.abs(p - 0.5) * 2
    return weight * (-torch.log(p + 1e-8))


def gdpo_loss(delta: Tensor, beta: float = 1.0, gamma: float = 2.0) -> Tensor:
    p = torch.sigmoid(beta * delta)
    return -(1 - p) ** gamma * torch.log(p + 1e-8)


def bppo_loss(delta: Tensor, beta: float = 1.0) -> Tensor:
    clipped = 2.0 * torch.tanh(beta * delta / 2.0)
    return -torch.nn.functional.logsigmoid(clipped)


class SCPSPOTrainer(DPOTrainer):
    """Implements the Self-Calibrating Proper-Scoring Preference Optimisation."""

    def __init__(
        self,
        alpha_lr: float,
        lam: float = 0.6,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.alpha = torch.nn.Parameter(torch.ones(1, device=self.accelerator.device))
        self.register_buffer("running_rms", torch.tensor(1.0, device=self.alpha.device))
        self.alpha_optimizer = torch.optim.AdamW([self.alpha], lr=alpha_lr)
        self.lam = lam

    def preference_loss(
        self,
        policy_chosen_logps,
        policy_rejected_logps,
        ref_chosen_logps,
        ref_rejected_logps,
        batch,
    ) -> Tensor:
        delta = (
            (policy_chosen_logps - policy_rejected_logps)
            - (ref_chosen_logps - ref_rejected_logps)
        )

        batch_rms = delta.float().pow(2).mean().sqrt().clamp_min(1e-6).detach()
        self.running_rms = 0.95 * self.running_rms + 0.05 * batch_rms
        p_hat = torch.sigmoid(self.alpha * delta / self.running_rms)

        kw = batch["k_w"].to(p_hat.device) + 1.0
        kl = batch["k_l"].to(p_hat.device) + 1.0
        y = kw / (kw + kl)
        unc = 1.0 / (kw + kl)
        weight = (1.0 - 4.0 * unc).clamp(min=0.0)

        brier = (p_hat - y).pow(2)
        dot = p_hat * y + (1 - p_hat) * (1 - y)
        spherical = 1.0 - dot / (
            torch.sqrt(p_hat**2 + (1 - p_hat) ** 2) * torch.sqrt(y**2 + (1 - y) ** 2)
            + 1e-8
        )
        loss = weight * (self.lam * brier + (1 - self.lam) * spherical)
        return loss.mean()

    def step(self, *args, **kwargs):
        super().step(*args, **kwargs)
        self.alpha_optimizer.zero_grad()
        self.alpha.grad = torch.autograd.grad(
            self.current_loss, self.alpha, retain_graph=True
        )[0]
        self.alpha_optimizer.step()

def train_one(
    cfg: Dict,
    seed: int,
    model_name: str,
    method: str,
    data: DatasetDict,
):
    set_seeds(seed)
    accelerator = Accelerator(log_with="wandb")
    run_name = f"{method}_{Path(model_name).name}_s{seed}"
    accelerator.init_trackers(
        project_name="SC-PSPO", config=cfg, init_kwargs={"wandb": {"name": run_name}}
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map={"": accelerator.process_index},
    )

    dpo_cfg = DPOConfig(
        beta=1.0,
        per_device_train_batch_size=cfg["training"]["per_device_train_batch"],
        per_device_eval_batch_size=cfg["training"]["per_device_train_batch"],
        gradient_accumulation_steps=cfg["training"]["grad_accum"],
        num_train_epochs=cfg["training"]["epochs"],
        learning_rate=cfg["training"]["lr"],
        max_length=2048,
        logging_steps=10,
        save_steps=0,
    )

    if method == "SC-PSPO":
        trainer_cls = SCPSPOTrainer
        extra_kwargs = dict(alpha_lr=cfg["training"]["alpha_lr"], lam=0.6)
    else:

        class _Baseline(DPOTrainer):
            loss_fn_map = {
                "DPO": lambda d: dpo_loss(d, beta=1.0),
                "DiscoPOP": lambda d: discopop_loss(d, beta=1.0),
                "GDPO": lambda d: gdpo_loss(d, beta=1.0),
                "BPPO": lambda d: bppo_loss(d, beta=1.0),
            }

            def preference_loss(
                self,
                policy_chosen_logps,
                policy_rejected_logps,
                ref_chosen_logps,
                ref_rejected_logps,
                batch,
            ):
                delta = (
                    (policy_chosen_logps - policy_rejected_logps)
                    - (ref_chosen_logps - ref_rejected_logps)
                )
                return self.loss_fn_map[method](delta).mean()

        trainer_cls = _Baseline
        extra_kwargs = {}

    trainer = trainer_cls(
        model,
        ref_model=model,
        args=dpo_cfg,
        train_dataset=data["train"],
        eval_dataset=data["validation"],
        tokenizer=tokenizer,
        **extra_kwargs,
    )

    print(f"Starting training {run_name} …")
    t0 = time.time()
    trainer.train()
    t1 = time.time()
    trainer.save_pretrained(str(RESULTS_DIR / f"{run_name}_ckpt"))
    wall_clock = t1 - t0
    tokens = trainer.state.global_step * cfg["training"]["per_device_train_batch"] * 2048
    stats = {
        "method": method,
        "model": model_name,
        "seed": seed,
        "tokens": tokens,
        "wall_clock_s": wall_clock,
        "gpu_mem": torch.cuda.max_memory_allocated() / 2**30,
    }
    with open(RESULTS_DIR / f"{run_name}_train.json", "w") as f:
        json.dump(stats, f, indent=2)
    accelerator.end_training()
    del trainer, model
    gc.collect()
    torch.cuda.empty_cache()
    return stats
