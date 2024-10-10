import argparse
import json
import os
import warnings
from collections import defaultdict
from tqdm import tqdm

import numpy as np
from sklearn.metrics import auc, roc_curve
from sklearn.exceptions import UndefinedMetricWarning

from data_utils import load_scores
from figure_utils import draw_score_distributions, plot_roc_curves

warnings.filterwarnings("ignore", category=UndefinedMetricWarning)


def calc_auc(scores, labels):
    return auc(*roc_curve(labels, scores)[:2])


def evaluate(scores, labels, return_roc_curve=False):
    outputs = {}
    fpr, tpr, _ = roc_curve(labels, scores)
    if return_roc_curve:
        outputs["roc_curve"] = (fpr, tpr)

    auc_roc = auc(fpr, tpr)
    acc = np.max(1 - (fpr + (1 - tpr)) / 2)
    metrics = {"AUC-ROC": auc_roc, "ACC": acc}
    metrics.update({f"TPR@{c}%FPR": tpr[np.where(fpr < c / 100.)[0][-1]] for c in [0.1, 1, 5, 10, 20]})
    outputs["metrics"] = metrics
    return outputs
    

def analyze_distribution(scores, labels):
    stats = []
    hists = []
    _, bins = np.histogram(scores, bins="auto")
    for l in [0, 1]:
        s = scores[labels == l]
        stats.append({"mean": s.mean(), "std": s.std(), "median": np.median(s), "min": s.min(), "max": s.max()})
        hists.append(np.histogram(s, bins=bins, density=True)[0])
    outputs = {
        "stats": {"non-member": stats[0], "member": stats[1]},
        "hist": (bins, (hists[0], hists[1])),
    }
    return outputs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("output_dir", type=str, help="The directory including socres for different methods")
    parser.add_argument("-m", "--method", type=str, nargs="*", default=[])
    parser.add_argument("-p", "--method_prefix", type=str, nargs="*", default=[])
    parser.add_argument("--keep_used", action="store_true", default=False, help="keep data used as prefixes from the test dataset")
    parser.add_argument("--roc_curve_filename", type=str, default=None, help="The filename to save a figure of roc curves")
    parser.add_argument("--plot_topk", type=int, default=0, help="only plot topk methods in each category")
    args = parser.parse_args()

    output_dir = args.output_dir
    score_dir = os.path.join(output_dir, "score")
    
    available_methods = sorted([file[:-6] for file in os.listdir(score_dir) if file.endswith(".jsonl")])
    methods = args.method + [m for m in available_methods if any(m.startswith(p) for p in args.method_prefix)]
    methods = methods or available_methods
    print(f"# of methods: {len(methods)} / {len(available_methods)}")

    scores, labels = load_scores(score_dir, methods, keep_used=args.keep_used)
    print(f"# of data: {len(next(iter(labels.values())))}")

    outputs = {}
    metrics_dir = os.path.join(output_dir, "metrics")
    stats_dir = os.path.join(output_dir, "stats")
    hist_dir = os.path.join(output_dir, "hist")
    os.makedirs(metrics_dir, exist_ok=True)
    os.makedirs(stats_dir, exist_ok=True)
    os.makedirs(hist_dir, exist_ok=True)
    for method in tqdm(methods, desc="evaluate", leave=False):
        outputs[method] = evaluate(scores[method], labels[method], return_roc_curve=True)
        outputs[method].update(analyze_distribution(scores[method], labels[method]))

        metrics_path = os.path.join(metrics_dir, f"{method}.json")
        with open(metrics_path, "w") as f:
            json.dump(outputs[method]["metrics"], f)

        stats_path = os.path.join(stats_dir, f"{method}.json")
        with open(stats_path, "w") as f:
            json.dump(outputs[method]["stats"], f)

        hist_path = os.path.join(hist_dir, f"{method}.png")
        draw_score_distributions(*outputs[method]["hist"], fig_path=hist_path)

    print("-" * 60 + "  METRICS  " + "-" * 60)
    for method in methods:
        print(f"{method:<18}: " + ", ".join(f"{k} {v * 100.:4.1f}" for k, v in outputs[method]["metrics"].items()))

    print("-" * 60 + "   STATS   " + "-" * 60)
    for method in methods:
        for label, stat in outputs[method]["stats"].items():
            print(f"{method:<18} {label:>10}: " + ", ".join(f"{k} {v:.4f}" for k, v in stat.items()))

    if args.plot_topk > 0:
        methods_by_category = defaultdict(list)
        for method in methods:
            methods_by_category[method.split("_")[0]].append(method)
        best = []
        for methods_in_category in methods_by_category.values():
            auc_rocs = [outputs[method]["metrics"]["AUC-ROC"] for method in methods_in_category]
            best_indices = np.argsort(np.array(auc_rocs), kind="stable")[-args.plot_topk:]
            best.extend([methods_in_category[i] for i in best_indices])
    else:
        best = methods
    roc_curves = {method: outputs[method]["roc_curve"] for method in best}

    if args.roc_curve_filename is not None:
        roc_curve_path = os.path.join(output_dir, args.roc_curve_filename)
        plot_roc_curves(roc_curves, fig_path=roc_curve_path)
