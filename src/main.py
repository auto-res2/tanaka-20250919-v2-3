import argparse, yaml, json, pprint
from pathlib import Path
from typing import Dict, List
import matplotlib.pyplot as plt
import seaborn as sns

from .preprocess import load_and_merge
from .train import train_one, RESULTS_DIR
from .evaluate import evaluate_model, FIG_DIR

sns.set_theme(style="whitegrid")


def _draw_bar(data: Dict[str, float], title: str, fname: str):
    plt.figure(figsize=(6, 4))
    keys, vals = list(data.keys()), list(data.values())
    ax = sns.barplot(x=keys, y=vals, palette="pastel")
    for i, v in enumerate(vals):
        ax.text(i, v + 0.01, f"{v:.2f}", ha="center", va="bottom")
    plt.ylabel(title)
    plt.title(title)
    plt.xticks(rotation=45)
    plt.tight_layout()
    pdf = FIG_DIR / f"{fname}.pdf"
    plt.savefig(pdf, bbox_inches="tight")
    plt.close()
    return pdf.name


def run(mode: str):
    cfg_path = Path("config") / ("smoke_test.yaml" if mode == "smoke" else "full_experiment.yaml")
    cfg = yaml.safe_load(open(cfg_path))
    ds_dict = load_and_merge(cfg["defaults"]["datasets"], seed=cfg["defaults"]["seed_list"][0])

    all_stats = []
    for seed in cfg["defaults"]["seed_list"]:
        for model in cfg["defaults"]["model_list"]:
            for method in cfg["defaults"]["methods"]:
                stats = train_one(cfg, seed, model, method, ds_dict)
                all_stats.append(stats)

    result_jsons = []
    for stat in all_stats:
        ckpt = RESULTS_DIR / f"{stat['method']}_{Path(stat['model']).name}_s{stat['seed']}_ckpt"
        res = evaluate_model(
            ckpt_path=ckpt,
            mt_bench=cfg["evaluation"]["mt_bench_prompts"],
            alpaca_eval_flag=cfg["evaluation"]["alpaca_eval"],
            calibration_ds=Path("data/preprocessed"),
            bins=cfg["evaluation"]["calibration_bins"],
        )
        res.update(stat)
        fp = RESULTS_DIR / f"{stat['method']}_{Path(stat['model']).name}_s{stat['seed']}_eval.json"
        with open(fp, "w") as f:
            json.dump(res, f, indent=2)
        result_jsons.append(res)

    speed = {
        f"{r['method']}_{Path(r['model']).name}": r["tokens"] / r["wall_clock_s"]
        for r in result_jsons
    }
    speed_fig = _draw_bar(speed, "Tokens / second", "throughput")

    wr = {
        k: v["alpaca_eval_win_rate"] if "alpaca_eval_win_rate" in v else 0.0
        for k, v in zip(speed.keys(), result_jsons)
    }
    win_fig = _draw_bar(wr, "AlpacaEval Win-Rate", "alpaca_winrate")

    print("\n==================  Experiment Description  ==================")
    pprint.pprint(cfg)
    print("\n==================  Numerical Results  =======================")
    for r in result_jsons:
        pprint.pprint(r)
    print("\nFigures saved:")
    print(speed_fig)
    print(win_fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-test", action="store_true", help="run quick check")
    parser.add_argument("--full-experiment", action="store_true", help="run full exp")
    args = parser.parse_args()
    if args.smoke_test == args.full_experiment:
        raise ValueError("Specify exactly one of --smoke-test or --full-experiment")

    run("smoke" if args.smoke_test else "full")
