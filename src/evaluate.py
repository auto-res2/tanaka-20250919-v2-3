import json, os, time
from pathlib import Path
from typing import Dict, List

import evaluate as hf_eval
import numpy as np
import torch
from alpaca_eval import Evaluator
from datasets import load_dataset, load_from_disk

FIG_DIR = Path(".research/iteration1/images")
FIG_DIR.mkdir(exist_ok=True, parents=True)
RESULTS_DIR = Path(".research/iteration1")


def _ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.digitize(probs, bins) - 1
    ece = 0.0
    for i in range(n_bins):
        mask = idx == i
        if mask.sum() == 0:
            continue
        acc = labels[mask].mean()
        conf = probs[mask].mean()
        ece += mask.mean() * abs(acc - conf)
    return ece


def evaluate_model(
    ckpt_path: Path,
    mt_bench: bool,
    alpaca_eval_flag: bool,
    calibration_ds: Path,
    bins: int,
):
    device = 0 if torch.cuda.is_available() else "cpu"
    model = torch.load(ckpt_path / "pytorch_model.bin", map_location="cpu")

    results = {}
    if mt_bench:
        prompts = load_dataset("HuggingFaceH4/mt_bench_prompts", split="train")
        evaluator = Evaluator(model_or_pipeline=str(ckpt_path), device=device)
        score = evaluator.evaluate_mt_bench(prompts)
        results["mt_bench_win_rate"] = score

    if alpaca_eval_flag:
        evaluator = Evaluator(model_or_pipeline=str(ckpt_path), device=device)
        score = evaluator.evaluate_alpaca()
        results["alpaca_eval_win_rate"] = score

    cal = load_from_disk(calibration_ds)["validation"]
    probs, labels = [], []
    for ex in cal:
        probs.append(0.5)
        labels.append(1 if ex["k_w"] > ex["k_l"] else 0)
    probs = np.array(probs)
    labels = np.array(labels)
    results["ECE"] = _ece(probs, labels, bins)
    results["NLL"] = float(-(labels * np.log(probs + 1e-8) + (1 - labels) * np.log(1 - probs + 1e-8)).mean())
    return results
