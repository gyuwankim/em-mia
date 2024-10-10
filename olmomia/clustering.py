import argparse
import os
import random
from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt
import faiss

from data_utils import load_jsonl, dump_jsonl

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--model_name", type=str, default="OLMo-7B")
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--num_data", type=int, default=100000)
    parser.add_argument("--num_sample", type=int, default=1000)
    parser.add_argument("--num_cluster", type=int, default=50)
    parser.add_argument("--kmeans_niter", type=int, default=50)
    parser.add_argument("--min_inter_dist", type=float, default=None)
    parser.add_argument("--min_cross_dist", type=float, default=None)
    parser.add_argument("--output_dir", type=str, default="data")
    parser.add_argument("--do_not_save", action="store_true", default=False)
    parser.add_argument("--together", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    data = load_jsonl(os.path.join(args.output_dir, f"{args.model_name}_{args.max_length}_{args.num_data}.jsonl"))

    vecs = np.load(os.path.join(args.output_dir, "vecs", f"{args.model_name}_{args.max_length}_{args.num_data}.npy"))
    vecs = vecs / np.linalg.norm(vecs, axis=1, keepdims=True)
    dim = vecs.shape[1]

    if args.together:
        raise NotImplementedError
        print("K-means clustering - together")        
        kmeans = faiss.Kmeans(dim, args.num_cluster, niter=args.kmeans_niter, verbose=True, spherical=True, seed=args.seed)
        kmeans.train(vecs)
        labels = kmeans.index.search(vecs, 1)[1].reshape((-1,))
    else:
        vecs_m = vecs[:args.num_data // 2]
        vecs_nm = vecs[args.num_data // 2:]

        print("K-means clustering - member")
        kmeans_m = faiss.Kmeans(dim, args.num_cluster, niter=args.kmeans_niter, verbose=True, spherical=True, seed=args.seed)
        kmeans_m.train(vecs_m)
        labels_m = kmeans_m.index.search(vecs_m, 1)[1].reshape((-1,))
        cluster_size_m = np.zeros((args.num_cluster,), dtype=np.int64)
        for l in labels_m:
            cluster_size_m[l] += 1
        print(cluster_size_m)

        print("K-means clustering - nonmember")
        kmeans_nm = faiss.Kmeans(dim, args.num_cluster, niter=args.kmeans_niter, verbose=True, spherical=True, seed=args.seed)
        kmeans_nm.train(vecs_nm)
        labels_nm = kmeans_nm.index.search(vecs_nm, 1)[1].reshape((-1,))
        cluster_size_nm = np.zeros((args.num_cluster,), dtype=np.int64)
        for l in labels_nm:
            cluster_size_nm[l] += 1
        print(cluster_size_nm)

        num_sample_m = (args.num_sample + 1) // 2
        num_sample_nm = args.num_sample // 2

        print("Distance computation - inter (member)")
        # inter_dist_m = np.zeros((args.num_cluster,))
        valid_clusters_m = []
        for i in tqdm(range(args.num_cluster), leave=False):
            pairwise_dist = (1 - vecs_m[labels_m == i] @ vecs_m[labels_m == i].T)
            n = cluster_size_m[i]
            # inter_dist_m[i] = pairwise_dist[np.triu_indices(n, 1)].mean()
            if args.min_inter_dist:  # greedily remove too close instances
                use = np.ones((n,), dtype=np.int64)
                for a in range(n):
                    if use[a] == 0:
                        continue
                    for b in range(a + 1, n):
                        use[b] = use[b] * (pairwise_dist[a, b] >= (args.min_inter_dist or -np.inf))
                for x, u in zip((labels_m == i).nonzero()[0], use):
                    if u == 0:
                        labels_m[x] = -1
                        n -= 1
                cluster_size_m[i] = n
            if n >= num_sample_m:
                valid_clusters_m.append(i)
        print(f"  # of valid member clusters: {len(valid_clusters_m)} / {args.num_cluster}")
        print(cluster_size_m)

        print("Distance computation - inter (nonmember)")
        # inter_dist_nm = np.zeros((args.num_cluster,))
        valid_clusters_nm = []
        for j in tqdm(range(args.num_cluster), leave=False):
            pairwise_dist = (1 - vecs_nm[labels_nm == j] @ vecs_nm[labels_nm == j].T)
            n = cluster_size_nm[j]
            # inter_dist_nm[j] = pairwise_dist[np.triu_indices(n, 1)].mean()
            if args.min_inter_dist:  # greedily remove too close instances
                use = np.ones((n,), dtype=np.int64)
                for a in range(n):
                    if use[a] == 0:
                        continue
                    for b in range(a + 1, n):
                        use[b] = use[b] * (pairwise_dist[a, b] >= (args.min_inter_dist or -np.inf))
                for x, u in zip((labels_nm == j).nonzero()[0], use):
                    if u == 0:
                        labels_nm[x] = -1
                        n -= 1
                cluster_size_nm[j] = n
            if n >= num_sample_nm:
                valid_clusters_nm.append(j)
        print(f"  # of valid nonmember clusters: {len(valid_clusters_nm)} / {args.num_cluster}")
        print(cluster_size_nm)

        print("Distance computation - cluster")
        # valid_clusters_m = range(args.num_cluster)
        # valid_clusters_nm = range(args.num_cluster)
        cluster_dist = []
        valid_pairs = [(i, j) for i in valid_clusters_m for j in valid_clusters_nm]
        for i, j in tqdm(valid_pairs, leave=False):
            cluster_dist.append(((i, j), (1 - vecs_m[labels_m == i] @ vecs_nm[labels_nm == j].T).mean()))
        cluster_dist.sort(key=lambda x: x[1])

        print("Sampling")
        hard_pair_idx = np.searchsorted(np.array([x[1] for x in cluster_dist]), args.min_cross_dist or -np.inf)
        pair_idx = {
            "easy": len(cluster_dist) - 1,
            "medium": (hard_pair_idx + len(cluster_dist)) // 2,
            "hard": hard_pair_idx,
        }
        dists = {}
        for difficulty, idx in pair_idx.items():
            cluster_m, cluster_nm = cluster_dist[idx][0]
            cross_dist = 1 - vecs_m[labels_m == cluster_m] @ vecs_nm[labels_nm == cluster_nm].T

            if difficulty == "easy":
                indices_m = cross_dist.mean(1).argsort()[::-1][:num_sample_m]
                indices_nm = cross_dist.mean(0).argsort()[::-1][:num_sample_nm]
            elif difficulty == "hard":
                indices_m = cross_dist.mean(1).argsort()[:num_sample_m]
                indices_nm = cross_dist.mean(0).argsort()[:num_sample_nm]
            else:
                random.seed(args.seed + 1); indices_m = np.array(random.sample(range(cross_dist.shape[0]), num_sample_m))
                random.seed(args.seed + 2); indices_nm = np.array(random.sample(range(cross_dist.shape[1]), num_sample_nm))
            
            avg_cross_dist = cross_dist[indices_m, :][:, indices_nm].mean()
            indices_m = (labels_m == cluster_m).nonzero()[0][indices_m]
            indices_nm = (labels_nm == cluster_nm).nonzero()[0][indices_nm]
            avg_inter_dist_m = (1 - vecs_m[indices_m] @ vecs_m[indices_m].T)[np.triu_indices(num_sample_m, 1)].mean()
            avg_inter_dist_nm = (1 - vecs_nm[indices_nm] @ vecs_nm[indices_nm].T)[np.triu_indices(num_sample_nm, 1)].mean()
            print(f"{difficulty:<6}: member cluster {cluster_m:2d} - nonmember cluster {cluster_nm:2d} | "
                  f"average distance cross {avg_cross_dist:6.4f}, inter (member) {avg_inter_dist_m:6.4f}, inter (nonmember) {avg_inter_dist_nm:6.4f}")
            
            dists[f"{difficulty}-member"] = 1 - vecs_m[indices_m] @ vecs_m[labels_m > 0].T
            dists[f"{difficulty}-nonmember"] = 1 - vecs_nm[indices_nm] @ vecs_m[labels_m > 0].T
        
            if not args.do_not_save:
                indices = list(indices_m) + list(indices_nm + args.num_data // 2)
                sampled_data = [data[idx] for idx in indices]
                dump_jsonl(sampled_data, os.path.join(args.output_dir, f"{args.model_name}_{args.max_length}_{args.num_sample}_{difficulty}.jsonl"))

        _, ax = plt.subplots(figsize=(6, 4))
        _, bins = np.histogram(list(dists.values()), bins="auto")
        for label, d in dists.items():
            ax.stairs(np.histogram(d, bins=bins, density=True)[0], bins, label=label, fill=True, alpha=.2)
        ax.set_xlabel(f"Avg distance with all members")
        ax.set_ylabel("Density")
        ax.set_title(f"{args.model_name}_{args.max_length}")
        ax.legend(loc="upper left", fontsize=10)
        plt.tight_layout()
        # plt.show()
        plt.savefig(os.path.join(args.output_dir, f"{args.model_name}_{args.max_length}_{args.num_sample}.png"), dpi=300)
        plt.close()               
