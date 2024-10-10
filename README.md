# 🕵🏻 Detecting Training Data of Large Language Models via Expectation Maximization

[![arXiv](https://img.shields.io/badge/arXiv-2410.07582-b31b1b.svg)](https://arxiv.org/abs/2410.07582)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

This is the repository for the official implementation of [Detecting Training Data of Large Language Models via Expectation Maximization](https://arxiv.org/abs/2410.07582).


## Setup
```bash
pip install -r requirements.txt
```


## 📘 Data
We mainly use [WikiMIA](https://huggingface.co/datasets/swj0419/WikiMIA) and OLMoMIA (will be uploaded in 🤗 soon, or you can create your own using our [scripts](https://github.com/gyuwankim/em-mia/tree/main/olmomia)) as our benchmark datasets.
You may use [MIMIR](https://huggingface.co/datasets/iamgroot42/mimir) or your own dataset by modifying `prepare_data` function in `data_utils.py` to return a list of dictionaries with a format of `{"input": text, "label", label}`, where `label` is either 1 (member) or 0 (non-member).


## 🚀 Running Experiments
Running `python run.py` (or `emmia.py`) with `--target_model ${MODEL} --dataset_name ${DATASET}` performs membership inference attack on a target model `${MODEL}` and a target dataset `${DATASET}` and stores membership scores in `output/${MODEL}/${DATASET}/score/${METHOD}.jsonl"` for each `${METHOD}`.
For [OLMo](https://github.com/allenai/OLMo) models, you should specify `--olmo_step` to select an intermediate checkpoint.
You can specify methods to use with the `-m` argument.

### Baselines
The default baseline methods (by not specifiying the `-m` argument) are `Loss`, `Ref`, `Zlib`, `Min-K`, and `Min-K++`.
For `Ref`, you need to specify a reference model with the `--ref_model` argument.
A default reference model is `EleutherAI/pythia-70m` for WikiMIA and `stabilityai/stablelm-base-alpha-3b-v2` for OLMoMIA.

### ReCaLL-based Baselines
You can apply ReCaLL using a prefix by concatenating randomly selected `n` shots defined by the `--num_shots` argument.
Depending on where shots come from, there are three different methods: `ReCaLL-Rand` from the entire dataset, `ReCaLL-RandM` from members in the dataset, and `ReCaLL-RandM` from non-members in the dataset.
For example, you can add arguments like `-m ReCaLL-RandM ReCaLL-Rand ReCaLL-RandNM --num_shots "[1,2,4,8,12]"`.

With the `-m ReCaLL-all` argument, you can apply ReCaLL on all data in the dataset by using each data in the dataset as a prefix. 
After that, you can apply average baselines (`Avg`, `AvgP`) with the `-m Avg` argument and `TopPref` baseline with the `-m TopPref -n $n` argument in `emmia.py`.

### EM-MIA
You can run `EM-MIA` by specifying initialization method(s) and prefix score update function(s) such as `-i Loss Min-K++_20 -p AUC-ROC` in `emmia.py`.

### Evaluation
To compare different MIA methods, you can plot AUC-ROC curves and calucate evaluation metrics, AUC-ROC and TPR @ k% FPR (k=0.1, 1, 5, 10, 20).
You can also get score statistics and draw histograms for members and non-members.
You can specify methods to compare by their name or their prefixes like `python eval.py output/${MODEL}/${DATASET} -m Loss Zlib Min-K_20 Min-K++_20 -p Ref ReCaLL-Avg ReCaLL-Rand --keep_used`.


## Acknowledgement
This codebase is adapted from the following repositories: [Min-K%](https://github.com/swj0419/detect-pretrain-code), [Min-K%++](https://github.com/zjysteven/mink-plus-plus), and [ReCaLL](https://github.com/ruoyuxie/recall).


## Citation
⭐ If you find our work (paper, implementation, and datasets) helpful, please consider citing our paper:
```bibtex
@article{kim2024detecting,
  title={Detecting Training Data of Large Language Models via Expectation Maximization},
  author={Kim, Gyuwan and Li, Yang and Spiliopoulou, Evangelia and Ma, Jie and Ballesteros, Miguel and Wang, William Yang},
  journal={arXiv preprint arXiv:2410.07582},
  year={2024}
}
```


## Contact
For any inquiry, please open an [issue](https://github.com/gyuwankim/em-mia/issues) or [contact](https://gyuwankim.github.io) the authors directly.
