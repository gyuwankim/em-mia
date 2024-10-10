# OLMoMIA Construction
We use intermediate checkpoints of [OLMo-7B](https://huggingface.co/allenai/OLMo-7B) trained with 100k, 200k, 300k, and 400k steps as target models.
One epoch is slightly larger than 450k steps.
For those checkpoints, training data between 0 ~ 100k steps are considered as members and training data between 400k ~ 450k are considered as non-members.
Let the time cutoffs be `T0 = 0`, `T1 = 100000`, `T2 = 400000`, and `T3 = 450000`.

## Random Sampling
You can randomly sample training instances from any training step and an index in a batch using [the data order file](https://olmo-checkpoints.org/ai2-llm/olmo-medium/wvc30anm/train_data/global_indices.npy).
Each training instance is split into samples based on the `EOS` token.
After processing, each data point contains `input` (in plain text), `idx` (a tuple of `idx` and `paragraph_idx`), and `length` (in tokens).

A fixed random seed is used to get a reproducible dataset.
You may run this process in parallel with different disjoint intervals `[S0, S1]`.
Sampling and processing an interval with length 1000 takes about 6-7 minutes and results in 2000 ~ 3000 data points.
Processed data points are saved in `data/sample/OLMo-7B_step{${T0}:${T1} or ${T2}:${T3}}/${S0}:${S1}.jsonl`.

```
# members
python sample.py --step_start ${T0} --step_end ${T1} --sample_start ${S0} --sample_end ${S1}
# non-members
python sample.py --step_start ${T2} --step_end ${T3} --sample_start ${S0} --sample_end ${S1}
```

## Length Filtering
You can extract data points whose sequence length are within `(L/2, L]` from the preprocessed files corresponding to `[T0, T1]` steps in the order of a sample index until `N/2` members are collected in total.
Similarly, `N/2` non-members are collected in total with sequence length within `(L/2, L]` from `[T2, T3]` steps.
Members and non-members are merged to construct a balanced dataset with the size `N`.
The filtered data are saved in `data/OLMo-7B_${L}_${N}.jsonl`.
```
python filter.py --member_step_start ${T0} --member_step_end ${T1} --nonmember_step_start ${T2} --nonmember_step_end ${T3} --num_data ${N} --max_length ${L}
```

## Clustering
Sampled data are mapped into embedding vectors using any embedding model (e.g., [nvidia/NV-Embed-v2](https://huggingface.co/nvidia/NV-Embed-v2)).
After K-means clustering, you can get datasets with different settings (`Easy`, `Medium`, and `Hard`) with varying difficulties by controlling the degree of overlap between the distribution of members and non-members.
You can easily construct datasets with `Random` or `Mix` settings based on the generated files.
```
python generate_vecs.py --dataset_name OLMo-7B_${L}_${N}
python clustering.py --model_name OLMo-7B --max_length ${L} --num_data ${N} --num_sample ${M} --num_cluster 50 --min_inter_dist 0.6
```
