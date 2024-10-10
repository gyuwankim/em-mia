import argparse
import os
import random
import zlib
from collections import defaultdict
from tqdm import tqdm

import numpy as np
import torch
import torch.nn.functional as F
import transformers
from accelerate import Accelerator

from modeling_utils import load_model, infer
from data_utils import prepare_data, dump_jsonl
from openai_utils import api_key_setup, gpt_paraphrase
from eval import evaluate


def set_seed(seed: int = 0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    transformers.set_seed(seed)


if __name__ == "__main__":
    default_methods = ["Loss", "Ref", "Zlib", "Min-K", "Min-K++"]
    available_methods = default_methods + ["ReCaLL-Rand", "ReCaLL-RandM", "ReCaLL-RandNM", "ReCaLL-all"]

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # model
    parser.add_argument("--target_model", type=str, default="EleutherAI/pythia-6.9b", help="the model to attack")
    parser.add_argument("--olmo_step", type=int, default=None)
    parser.add_argument("--ref_model", type=str, default="EleutherAI/pythia-70m")  # stabilityai/stablelm-base-alpha-3b-v2
    # data
    parser.add_argument("--dataset_name", type=str, default="wm128", help="dataset name")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--output_dir", type=str, default="output")
    # attack
    parser.add_argument("-m", "--method", type=str, nargs="+", default=default_methods, choices=available_methods)
    parser.add_argument("--mink_ratio", nargs="+", type=float, default=[10, 20, 30, 40, 50])
    # ReCaLL
    parser.add_argument("--max_num_shots", type=int, default=28, help="the maximum number of shots for prefix")
    parser.add_argument("--num_shots", type=eval, default=None, help="the number of shots for prefix")
    parser.add_argument("--synthetic_prefix", action="store_true", default=False, help="whether to use synehtic prefix")
    parser.add_argument("--api_key_path", type=str, default=None, help="path to the api key file for OpenAI API if using synehtic prefix")
    # ReCaLL-all
    parser.add_argument("--prefix_dataset_name", type=str, default=None, help="prefix dataset name")
    parser.add_argument("--batch_size", type=int, default=1, help="batch size for ReCaLL-all")
    parser.add_argument("--start_idx", type=int, default=None, help="start idx for ReCaLL-all")
    parser.add_argument("--end_idx", type=int, default=None, help="start idx for ReCaLL-all")
    args = parser.parse_args()

    set_seed(args.seed)
    if args.synthetic_prefix: 
        assert args.api_key_path is not None
        api_key_setup(args.api_key_path)

    # load models
    target_model_name = args.target_model.split("/")[-1] + (f"_{args.olmo_step}" if args.olmo_step is not None else "")
    target_model, target_tokenizer = load_model(args.target_model, olmo_step=args.olmo_step)
    if "Ref" in args.method:
        assert args.ref_model is not None
        ref_model_name = args.ref_model.split("/")[-1]
        ref_model, ref_tokenizer = load_model(args.ref_model)
    else:
        ref_model_name = ""
        ref_model, ref_tokenizer = None, None
    accelerator = Accelerator()
    target_model, ref_model = accelerator.prepare(target_model, ref_model)

    # prepare data
    data = prepare_data(args.dataset_name, tokenizer=target_tokenizer)
    if "ReCaLL-all" in args.method:
        if args.prefix_dataset_name is None or args.prefix_dataset_name == args.dataset_name:
            args.prefix_dataset_name = args.dataset_name
            prefix_data = data
        else:
            prefix_data = prepare_data(args.prefix_dataset_name, tokenizer=target_tokenizer)
    
    # prepare random prefixes for ReCaLL
    if args.num_shots is None:
        args.num_shots = [(n, 1) for n in range(1, args.max_num_shots + 1)]
    else:
        if not isinstance(args.num_shots, list):
            args.num_shots = [args.num_shots]
        num_shots = []
        for x in args.num_shots:
            x = x if isinstance(x, tuple) else (x, 1)
            assert len(x) == 2
            num_shots.append(x)
        args.num_shots = list(sorted(set(num_shots)))
        args.max_num_shots = max([n * g for n, g in args.num_shots])

    random_cond = {
        "": (lambda ex: True),
        "M": (lambda ex: ex["label"] == 1),
        "NM": (lambda ex: ex["label"] == 0),
    }

    random_indices = {}
    random_shots = {}
    for c, r in random_cond.items():
         if f"ReCaLL-Rand{c}" in args.method:
            random.seed(args.seed)
            random_indices[c] = [idx for idx, ex in enumerate(data) if r(ex)]
            random_indices[c] = random.sample(random_indices[c], min(args.max_num_shots, len(random_indices[c])))
            random_shots[c] = [data[idx]["input"] for idx in random_indices[c]]
            if args.synthetic_prefix:
                random_shots[c] = [gpt_paraphrase(shot) for shot in random_shots[c]]        

    # calculate MIA scores
    all_results = defaultdict(list)
    for ex_idx, ex in tqdm(enumerate(data), total=len(data)):
        target_data = ex["input"]

        ll, logits, labels = infer(target_model, target_tokenizer, target_data)
        
        # baselines 
        pred = {}
        if "Loss" in args.method:
            pred["Loss"] = ll
        if "Ref" in args.method:
            ll_ref, _, _ = infer(ref_model, ref_tokenizer, target_data)
            pred[f"Ref_{ref_model_name}"] = ll - ll_ref
        if "Zlib" in args.method:
            pred["Zlib"] = ll / len(zlib.compress(bytes(target_data, "utf-8")))

        # Min-K and Min-K++
        if any(m in args.method for m in ["Min-K", "Min-K++"]):
            logits = logits.to(torch.float32)
            probs = F.softmax(logits, dim=-1)
            logprobs = F.log_softmax(logits, dim=-1)
            token_logprobs = logprobs.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
            mu = (probs * logprobs).sum(-1)
            sigma = (probs * torch.square(logprobs)).sum(-1) - torch.square(mu)
            normalized_token_logprobs = (token_logprobs - mu) / sigma.sqrt()
            for m, tlps in [("Min-K", token_logprobs), ("Min-K++", normalized_token_logprobs)]:
                if m in args.method:
                    for ratio in args.mink_ratio:
                        pred[f"{m}_{ratio}"] = tlps.topk(int(len(logits) * ratio / 100.), largest=False)[0].mean().item()

        used = {}
        # ReCaLL-Rand*
        for c in random_cond.keys():
            if f"ReCaLL-Rand{c}" in args.method:
                for n, g in args.num_shots:
                    conditional_ll, _, _ = infer(
                        target_model, target_tokenizer, target_data, 
                        prefix=[" ".join(random_shots[c][i * n:(i + 1) * n]) for i in range(g)]
                    )
                    recall_name = f"ReCaLL-Rand{c}_" + (f"{n:02d}*{g}" if g > 1 else f"{n:02d}")
                    pred[recall_name] = conditional_ll.mean(-1) / ll    # average ensemble
                    # used as prefixes
                    used[recall_name] = int(ex_idx in random_indices[c][:n])
        
        if "ReCaLL-all" in args.method:
            args.start_idx = max(args.start_idx or 0, 0)
            args.end_idx = min(args.end_idx or len(prefix_data), len(prefix_data))
            for start_idx in range(args.start_idx, args.end_idx, args.batch_size):
                end_idx = min(start_idx + args.batch_size, len(prefix_data))
                conditional_ll, _, _ = infer(
                    target_model, target_tokenizer, target_data, 
                    prefix=[prefix_data[idx]["input"] for idx in range(start_idx, end_idx)]
                )
                for idx in range(start_idx, end_idx):                        
                    recall_name = f"ReCaLL_{args.prefix_dataset_name}-idx{idx:04d}"
                    pred[recall_name] = conditional_ll[idx - start_idx] / ll
                    # used as prefixes
                    used[recall_name] = int(ex_idx == idx)

        for method, score in pred.items():
            r = {"score": float(score), "label": ex["label"]}
            if used.get(method, 0):
                r["used"] = 1
            all_results[method].append(r)

    # save scores
    output_dir = os.path.join(args.output_dir, target_model_name, args.dataset_name)
    score_dir = os.path.join(output_dir, "score")
    os.makedirs(score_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")
    for method, results in all_results.items():
        results_path = os.path.join(score_dir, f"{method}.jsonl")
        dump_jsonl(results, results_path)
        
        metrics = evaluate([r["score"] for r in results], [r["label"] for r in results])["metrics"]
        print(f"{method:<18}: " + ", ".join(f"{k} {v * 100.:4.1f}" for k, v in metrics.items()))
