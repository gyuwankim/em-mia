import argparse
import os

from data_utils import load_jsonl, dump_jsonl


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--model", type=str, default="allenai/OLMo-7B")
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--member_step_start", type=int, required=0)
    parser.add_argument("--member_step_end", type=int, required=100000)
    parser.add_argument("--nonmember_step_start", type=int, required=400000)
    parser.add_argument("--nonmember_step_end", type=int, required=450000)
    parser.add_argument("--num_data", type=int, default=1000)
    parser.add_argument("--output_dir", type=str, default="data")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    model_name = args.model.split("/")[-1]

    def extract(label, step_start, step_end, num_data):
        sample_dir = os.path.join(args.output_dir, "sample", f"{model_name}_step{step_start}:{step_end}")
        assert os.path.isdir(sample_dir)

        intervals = [tuple(int(x) for x in file[:-6].split(":")) for file in os.listdir(sample_dir) if file.endswith(".jsonl")]
        intervals = sorted(intervals, key=lambda x: (x[0], -x[1]))

        data = []
        count = 0
        current_end = 0
        current_idx = None
        for sample_start, sample_end in intervals:
            if sample_start < current_end:
                continue
            samples = load_jsonl(os.path.join(sample_dir, f"{sample_start}:{sample_end}.jsonl"))
            for sample in samples:
                if sample["idx"][0] != current_idx and args.max_length // 2 < sample["length"] <= args.max_length:
                    data.append({"input": sample["input"], "label": label})
                    count += 1
                    if count >= num_data:
                        break
                    current_idx = sample["idx"][0]
            current_end = sample_end
            if count >= num_data:
                break
        return data
    
    members = extract(1, args.member_step_start, args.member_step_end, (args.num_data + 1) // 2)
    nonmembers = extract(0, args.nonmember_step_start, args.nonmember_step_end, args.num_data // 2)

    print(f"# of members: {len(members)}")
    print(f"# of nonmembers: {len(nonmembers)}")
    if len(members) + len(nonmembers) < args.num_data:
        print("Insufficient samples!!!")
    else:
        os.makedirs(args.output_dir, exist_ok=True)
        output_path = os.path.join(args.output_dir, f"{model_name}_{args.max_length}_{args.num_data}.jsonl")
        print(f"Saving data to {output_path}")
        dump_jsonl(members + nonmembers, output_path)
