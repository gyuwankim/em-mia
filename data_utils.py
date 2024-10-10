import json
import os
from tqdm import tqdm

import numpy as np

from datasets import load_dataset


def load_jsonl(path):
    with open(path, "r") as f:
        data = [json.loads(line) for line in f]
    return data


def dump_jsonl(data, path):
    with open(path, "w") as f:
        for line in data:
            f.write(json.dumps(line) + "\n")


def add_id(data):
    all_data = []
    for id, ex in enumerate(data):
        ex["id"] = id
        all_data.append(ex)
    return all_data


def prepare_data(dataset_name, tokenizer=None):
    mimir_domains = {
        "we": "wikipedia_(en)", 
        "gh": "github", 
        "pc": "pile_cc", 
        "pm": "pubmed_central", 
        "ax": "arxiv", 
        "dm": "dm_mathematics", 
        "hn": "hackernews",
    }
    if dataset_name[:2] == "wm":  # WikiMIA
        length = int(dataset_name[2:])
        dataset = load_dataset("swj0419/WikiMIA", split=f"WikiMIA_length{length}")
        # convert huggingface data to list dic
        data = [dataset[i] for i in range(len(dataset))]
    elif dataset_name[:2] in mimir_domains.keys():  # MIMIR
        ngram_str = dataset_name[2:]
        ngram = f"{int(ngram_str[:-1])}_0.{ngram_str[-1]}"
        domain = mimir_domains[dataset_name[:2]]
        dataset_name = f"MIMIR_ngram_{ngram}_{domain}"
        dataset = load_dataset("iamgroot42/mimir", domain, trust_remote_code=True)[f"ngram_{ngram}"]
        data = [{"input": x, "label": 1} for x in dataset["member"]] + [{"input": x, "label": 0} for x in dataset["nonmember"]]
    elif dataset_name.startswith("OLMo"):
        data = load_jsonl(f"data/{dataset_name}.jsonl")
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}. Please modify the code to include the dataset.")

    print(f"Loaded {dataset_name} | # of data: {len(data)}")
    if tokenizer is not None:
        lengths = np.array([tokenizer(ex["input"], return_tensors="pt").input_ids.size(-1) for ex in data])
        print(f"Length: mean {lengths.mean():.0f}, min {lengths.min()}, max {lengths.max()}, median {np.median(lengths):.0f}")
    return data


def load_scores(score_dir, methods, keep_used=False, if_exists=False):
    scores = {}
    labels = {}
    used = 0
    for method in tqdm(methods, desc="load scores", leave=False):
        scores_path = os.path.join(score_dir, f"{method}.jsonl")
        if not os.path.isfile(scores_path):
            assert if_exists, f"{method}.jsonl file does not exist in {score_dir}"
        results = load_jsonl(scores_path)
        scores[method] = np.array([r["score"] for r in results])
        scores[method][np.isposinf(scores[method])] = scores[method][~np.isposinf(scores[method])].max() + 1e-6
        scores[method][np.isneginf(scores[method])] = scores[method][~np.isneginf(scores[method])].min() - 1e-6
        scores[method][np.isnan(scores[method])] = 0
        labels[method] = np.array([r["label"] for r in results])
        # get intersection of data never used as a prefix
        if not keep_used:
            used = np.array([r.pop("used", 0) for r in results]) | used

    if not keep_used:
        for method in methods:
            scores[method] = scores[method][used == 0]
            labels[method] = labels[method][used == 0]
    return scores, labels