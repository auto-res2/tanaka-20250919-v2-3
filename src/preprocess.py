import os, json, random, hashlib, itertools
from pathlib import Path
from typing import Dict, Tuple, List
import datasets as ds
from datasets import load_dataset, DatasetDict
from sklearn.model_selection import train_test_split
from tqdm import tqdm

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)


def _sha1(x: str) -> str:
    return hashlib.sha1(x.encode()).hexdigest()


def load_and_merge(dset_cfg: Dict[str, str | int | None], seed: int) -> DatasetDict:
    """
    1. argilla/dpo-mix-7k  – binary
    2. argilla/reddit_tldr_pref_5votes – soft counts
    3. argilla/human_harmless_500 – noisy majority labels, but per-annotator votes provided
    """
    random.seed(seed)

    def _normalise(example, src: str) -> Dict:
        if src == "argilla/dpo-mix-7k":
            kw, kl = 1, 0
        else:
            kw, kl = example["k_w"], example["k_l"]
        return {
            "prompt": example["prompt"],
            "chosen": example["chosen"],
            "rejected": example["rejected"],
            "k_w": kw,
            "k_l": kl,
            "source": src,
        }

    records: list[dict] = []
    for name, limit in dset_cfg.items():
        split = load_dataset(name, split="train", token=os.getenv("HF_TOKEN"))
        if limit != "all":
            split = split.select(range(int(limit)))
        for ex in tqdm(split, desc=f"Normalising {name}"):
            records.append(_normalise(ex, name))

    key_fn = lambda r: _sha1(
        r["prompt"].lower() + r["chosen"].lower() + r["rejected"].lower()
    )
    uniq = {}
    for r in records:
        uniq[key_fn(r)] = r
    records = list(uniq.values())

    records = [r for r in records if len(r["prompt"] + r["chosen"] + r["rejected"]) < 8_000]

    sources = [r["source"] for r in records]
    train, tmp, y_train, y_tmp = train_test_split(
        records, sources, test_size=0.2, stratify=sources, random_state=seed
    )
    val, test, _, _ = train_test_split(
        tmp, y_tmp, test_size=0.5, stratify=y_tmp, random_state=seed
    )

    ds_dict = DatasetDict(
        {
            "train": ds.Dataset.from_list(train),
            "validation": ds.Dataset.from_list(val),
            "test": ds.Dataset.from_list(test),
        }
    )
    ds_dict.save_to_disk(str(DATA_DIR / "preprocessed"))
    return ds_dict
