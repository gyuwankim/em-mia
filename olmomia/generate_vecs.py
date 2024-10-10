import argparse
import os

import numpy as np
import torch
from transformers import AutoModel

from data_utils import prepare_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--dataset_name", type=str, default="OLMo-7B_128_100000", help="dataset name")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=32)
    parser.add_argument("--output_dir", type=str, default="data/vecs/")
    args = parser.parse_args()

    data = prepare_data(args.dataset_name)

    embedder = AutoModel.from_pretrained(
        "nvidia/NV-Embed-v2", 
        trust_remote_code=True, 
        torch_dtype=torch.float16, 
        device_map="auto",
    )

    vecs = embedder._do_encode(
        [x["input"] for x in data],
        batch_size=args.batch_size,
        instruction="", 
        max_length=args.max_length, 
        num_workers=32,
        return_numpy=True,
    )

    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, f"{args.dataset_name}.npy"), "wb") as f:
        np.save(f, vecs)
