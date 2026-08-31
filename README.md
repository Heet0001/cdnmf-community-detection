# CDNMF Community Detection (Paper Reproduction)

Reproduction of **Contrastive Deep Nonnegative Matrix Factorization for Community Detection** (ICASSP 2024) on Cora, Citeseer, and PubMed.

- Paper: [arXiv:2311.02357](https://arxiv.org/abs/2311.02357)
- Official code: [6lyc/CDNMF](https://github.com/6lyc/CDNMF)

**Group 4** — IIIT Vadodara, International Campus Diu  
Heet Ladani (202311035) · Namrakumar Koyani (202311057) · Pooja (202311063) · Shailendra Singh (202311078)

## Repository layout

```
notebooks/          Colab notebooks (one dataset each)
  Lab1CDNMF_Colab_1.ipynb   Cora
  Lab1CDNMF_Colab_2.ipynb   Citeseer
  Lab1CDNMF_Colab_3.ipynb   PubMed
Model/              CDNMF model (memory-safe contrastive loss)
Dataset/            graph loader (walks.txt optional)
PreTrainer/         layer-wise NMF pre-training
Utils/              metrics and helpers
script_*.py         original-style training entry points
slides/             Beamer slides + training-curve figures
```

Data files (`Database/`) are **not** in this repo. The notebooks clone the [official CDNMF repo](https://github.com/6lyc/CDNMF) on Colab and load Cora / Citeseer / PubMed from there.

## How to run

1. Open a notebook in Google Colab.
2. Runtime → GPU (T4).
3. Run all cells top to bottom.

| Notebook | Dataset | Paper epochs | Our setting |
|----------|---------|--------------|-------------|
| `Lab1CDNMF_Colab_1.ipynb` | Cora | 550 × 20 runs | 20 runs |
| `Lab1CDNMF_Colab_2.ipynb` | Citeseer | 1000 × 20 runs | 20 runs |
| `Lab1CDNMF_Colab_3.ipynb` | PubMed | 600 × 20 runs | 1 run (T4 memory / time) |

Local Python (after cloning this repo and the official data):

```bash
pip install -r requirements.txt
python script_cora.py
```

## Changes from the official code

- **Citeseer / PubMed** ship without `walks.txt`. The loader skips missing or empty walk files instead of crashing (`KeyError: ''`).
- **PubMed** has 19,717 nodes. The original contrastive loss builds several dense \(N \times N\) GPU matrices (~1.45 GiB each) and OOMs on a T4. We use chunked similarity and avoid extra dense \(N \times N\) reconstruction / Laplacian tensors.

## Results (ACC / NMI)

| Dataset | Paper | Ours |
|---------|-------|------|
| Cora | 0.6081 / 0.4006 | 0.6006 ± 0.016 / 0.3942 ± 0.026 (20 runs) |
| Citeseer | 0.4756 / 0.2559 | 0.4700 ± 0.025 / 0.2500 ± 0.021 (20 runs) |
| PubMed | 0.6653 / 0.2330 | 0.6600 / 0.2232 (1 run) |

## Citation

```bibtex
@inproceedings{li2024contrastive,
  title={Contrastive deep nonnegative matrix factorization for community detection},
  author={Li, Yuecheng and Chen, Jialong and Chen, Chuan and Yang, Lei and Zheng, Zibin},
  booktitle={ICASSP 2024-2024 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  pages={6725--6729},
  year={2024},
  organization={IEEE}
}
```
