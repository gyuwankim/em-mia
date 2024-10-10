import argparse
import os
import random
from tqdm import tqdm

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import rankdata, kendalltau
from accelerate import Accelerator

from modeling_utils import load_model, infer
from data_utils import prepare_data, load_scores, dump_jsonl
from eval import calc_auc


if __name__ == "__main__":
    prefix_score_function = {
        "AUC-ROC": lambda x, y: calc_auc(x, y > np.median(y)),
        "Kendall-Tau": lambda x, y: kendalltau(x, y).statistic,
        "RankDist": lambda x, y: 1. - np.abs(rankdata(x, axis=-1) - rankdata(y, axis=-1)).mean() / (x.shape[-1] - 1),
    }

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # model
    parser.add_argument("--target_model", type=str, default="EleutherAI/pythia-6.9b", help="the model to attack")
    parser.add_argument("--olmo_step", type=int, default=None)
    # data
    parser.add_argument("--dataset_name", type=str, default="wm128", help="dataset name")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--output_dir", type=str, default="output")
    # EM-MIA
    parser.add_argument("-i", "--init", type=str, nargs="*", default=["Loss", "Ref_stablelm-base-alpha-3b-v2", "Zlib", "Min-K_20", "Min-K++_20"], help="initial membership scores")
    parser.add_argument("-m", "--membership_score", type=str, default="NegPref", choices=["NegPref", "TopPref", "Avg"], help="update rule for membership scores")
    parser.add_argument("-n", "--num_top_prefixes", type=int, default=12, help="the number of top prefixes")
    parser.add_argument("-o", "--prefixes_order", type=str, default="F", choices=["F", "B", "R"], help="order of prefixes: forward/backward/random")
    parser.add_argument("-p", "--prefix_score", type=str, nargs="*", default=["AUC-ROC"], choices=prefix_score_function.keys(), help="metric for the effectiveness as a prefix for ReCaLL")
    parser.add_argument("--max_iter", type=int, default=10, help="the maximum number of iteration for EM-MIA")
    args = parser.parse_args()

    # load models
    target_model_name = args.target_model.split("/")[-1] + (f"_{args.olmo_step}" if args.olmo_step is not None else "")
    target_model, target_tokenizer = None, None
    if args.membership_score in ["TopPref"]:
        target_model, target_tokenizer = load_model(args.target_model, olmo_step=args.olmo_step)
        accelerator = Accelerator()
        target_model = accelerator.prepare(target_model)

    # prepare data
    data = prepare_data(args.dataset_name, tokenizer=target_tokenizer)

    output_dir = os.path.join(args.output_dir, target_model_name, args.dataset_name)
    score_dir = os.path.join(output_dir, "score")

    init_scores, labels = load_scores(score_dir, list(set(args.init + ["Loss"])), keep_used=True)
    labels = next(iter(labels.values()))
    recall_all_names = [f"ReCaLL_{args.dataset_name}-idx{idx:04d}" for idx in range(len(data))]
    recall_all_scores, _ = load_scores(score_dir, recall_all_names, keep_used=True)
    recall_all_scores = np.stack(list(recall_all_scores.values()))

    def recall(prefix):
        recall_scores = []
        for ex_idx, ex in tqdm(enumerate(data), total=len(data), leave=False):
            target_data = ex["input"]
            ll = init_scores["Loss"][ex_idx]
            ll_nm, _, _ = infer(target_model, target_tokenizer, target_data, prefix=prefix)
            recall_scores.append(ll_nm.mean(-1) / ll)
        recall_scores = np.array(recall_scores)
        return recall_scores
    
    def save_scores(method, membership_scores, print_auc=False):
        results = [{"score": score, "label": int(label)} for score, label in zip(membership_scores, labels)]
        results_path = os.path.join(score_dir, f"{method}.jsonl")
        dump_jsonl(results, results_path)
        if print_auc:
            print(f"{method:<16}: {calc_auc(membership_scores, labels) * 100.:4.1f}")

    if args.membership_score == "TopPref":
        prefix_scores = np.array([calc_auc(recall_all_scores[idx], labels) for idx in range(len(data))])
        top_prefixes = prefix_scores.argsort()[::-1][:args.num_top_prefixes]
        for order in ["F", "B", "R"]:
            if order == "F":
                ordered_top_prefixes = top_prefixes
            elif order == "B":
                ordered_top_prefixes = top_prefixes[::-1]
            elif order == "R":
                random.seed(args.seed)
                ordered_top_prefixes = top_prefixes.copy()
                random.shuffle(ordered_top_prefixes)
            prefix = " ".join([data[prefix_idx]["input"] for prefix_idx in ordered_top_prefixes])
            save_scores(f"ReCaLL-TopPref{args.num_top_prefixes:02d}{order}", recall([prefix]), print_auc=True)
    elif args.membership_score == "Avg":
        save_scores(f"ReCaLL-Avg", recall_all_scores.mean(0), print_auc=True)
        save_scores(f"ReCaLL-AvgP", recall_all_scores.mean(1), print_auc=True)
    else:
        assert args.membership_score == "NegPref"
        iteration_dir = os.path.join(output_dir, "iteration")
        os.makedirs(iteration_dir, exist_ok=True)
        final_aucs = {}
        for metric in args.prefix_score:
            _, ax = plt.subplots(figsize=(6, 4))
            for init in args.init:
                emmia_method = f"EM-MIA_{metric}_{init}"
                membership_scores = init_scores[init]
                membership_scores[np.isnan(membership_scores)] = 0
                aucs = [calc_auc(membership_scores, labels)]
                pbar = tqdm(range(1, args.max_iter + 1), leave=False)
                for it in pbar:
                    pbar.set_description(f"{emmia_method} - {aucs[-1] * 100.:4.1f}")
                    pbar.refresh()
                    
                    # update prefix scores
                    prefix_scores = np.array([prefix_score_function[metric](recall_all_scores[idx], membership_scores) for idx in range(len(data))])

                    # update membership scores
                    membership_scores = -prefix_scores
                    membership_scores[np.isnan(membership_scores)] = 0
                    aucs.append(calc_auc(membership_scores, labels))

                    results = [{"score": score, "label": int(label)} for score, label in zip(membership_scores, labels)]
                    results_path = os.path.join(score_dir, f"{emmia_method}_iter{it:02d}.jsonl")
                    dump_jsonl(results, results_path)
                plt.plot(range(args.max_iter + 1), aucs, 'o-', label=init)
                final_aucs[emmia_method] = aucs[-1]
            ax.set_xlabel("Iteration")
            ax.set_ylabel("AUC-ROC")
            ax.legend(loc="center right", fontsize=12)
            plt.tight_layout()
            plt.savefig(os.path.join(iteration_dir, f"EM-MIA_{metric}.png"), bbox_inches='tight', pad_inches=0, dpi=300)
            plt.close()

        for emmia_method, auc in final_aucs.items():
            print(f"{emmia_method:<30}: {auc * 100.:4.1f}")
