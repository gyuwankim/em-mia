import argparse
import os
import random
from tqdm import tqdm

import numpy as np

from olmo.config import TrainConfig
from olmo.data import build_memmap_dataset
from hf_olmo import OLMoTokenizerFast

from data_utils import dump_jsonl


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--model", type=str, default="allenai/OLMo-7B")
    parser.add_argument("--config", type=str, default="olmo/configs/OLMo-7B.yaml")
    parser.add_argument("--global_indices", type=str, default="olmo/global_indices/OLMo-7B_epoch1.npy")
    parser.add_argument("--step_start", type=int, required=True)
    parser.add_argument("--step_end", type=int, required=True)
    parser.add_argument("--sample_start", type=int, required=True)
    parser.add_argument("--sample_end", type=int, required=True)
    parser.add_argument("--output_dir", type=str, default="data")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    model_name = args.model.split("/")[-1]
    tokenizer = OLMoTokenizerFast.from_pretrained(args.model, trust_remote_code=True)

    cfg = TrainConfig.load(args.config)
    batch_size = cfg.global_train_batch_size
    dataset = build_memmap_dataset(cfg, cfg.data)

    global_indices = np.memmap(args.global_indices, mode="r+", dtype=np.uint32)

    def get_samples(idx):
        try:
            instance = dataset[global_indices[idx]]["input_ids"]
        except:
            return []
        eos_positions = (instance == tokenizer.eos_token_id).nonzero().view(-1)
        if len(eos_positions) >= 2:
            return [(instance[s + 1:e], paragraph_idx) for paragraph_idx, (s, e) in enumerate(zip(eos_positions[:-1], eos_positions[1:]), 1)]
        return []
    
    assert args.step_start < args.step_end
    assert args.sample_start < args.sample_end

    output_dir = os.path.join(args.output_dir, "sample", f"{model_name}_step{args.step_start}:{args.step_end}")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{args.sample_start}:{args.sample_end}.jsonl")

    random.seed(args.seed)
    indices = random.sample(range(args.step_start * batch_size, args.step_end * batch_size), args.sample_end)[args.sample_start:]

    data = []
    for idx in tqdm(indices):
        for input_ids, paragraph_idx in get_samples(idx):
            data.append(
                {
                    "input": tokenizer.decode(input_ids),
                    "idx": (idx, paragraph_idx),
                    "length": len(input_ids),
                }
            )
    print(f"# of data: {len(data)}")
    dump_jsonl(data, output_path)
