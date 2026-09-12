# Featured Graph Coarsening (FGC) — Runnable Reproduction

A reproduction of [**GraphCoarsening/Featured-Graph-Coarsening**](https://github.com/GraphCoarsening/Featured-Graph-Coarsening), the official code for:

> **Featured Graph Coarsening with Similarity Guarantees**
> Manoj Kumar, Anurag Sharma, Shashwat Saxena, Sandeep Kumar — *ICML 2023*
> [PMLR v202](https://proceedings.mlr.press/v202/kumar23a) · Indian Institute of Technology Delhi
>
> Extended journal version: **A Unified Framework for Optimization-Based Graph Coarsening**
> [arXiv:2210.00437](https://arxiv.org/abs/2210.00437) (JMLR) — the upstream repo is named after this version; the same code covers both.

**Purpose of this reproduction:** upstream ships **only Jupyter notebooks**, and the
main experiment notebook cannot be executed non-interactively. This adds a
command-line driver so FGC can be run reproducibly, without modifying the algorithm.

All methods and the original implementation are the authors' work. The original README is preserved as [`README_ORIGINAL.md`](README_ORIGINAL.md). Upstream is Apache-2.0 licensed; that [`LICENSE`](LICENSE) is retained.

---

## Status

Runs end-to-end on CPU (macOS arm64, Python 3.9.6, torch 2.8.0, PyG 2.6.1).

Full faithful settings — 100 inner `update_C` steps per iteration × 10 iterations:

| Dataset | Ratio | Nodes | Loss (first → last) | Relative eigen error (top-100) |
|---------|-------|-------|---------------------|-------------------------------|
| cora | 0.3 | 2,708 → 812 | 14,963,217 → 9,003,976 | **0.0977** |
| cora | 0.5 | 2,708 → 1,354 | 18,706,824 → 8,090,770 | **0.0609** |

For contrast, a truncated run (2 iterations × 10 inner steps) gives an eigen error
of **0.99** — convergence matters, and the full run reproduces the paper's
low-single-digit-percent regime.

---

## Why upstream could not be run non-interactively

`FGC_experiment.ipynb` is an exploratory notebook, not a script:

- It contains IPython magics and shell escapes (`%load_ext autoreload`, `%matplotlib inline`, `!pip install deeprobust`).
- Several cells **each overwrite `L` and `X`** with a different dataset — Cora, then polblogs (with generated features), then PyGSP graphs, then synthetic graphs. Executing the notebook top-to-bottom silently mixes datasets and produces meaningless results.
- Later cells reference variables (`Ans`, `C1`, `C2`) that are never defined in the notebook.
- `deeprobust`, its data loader, uses the NumPy alias `np.int`, removed in NumPy ≥ 1.24.

## What this reproduction adds

| File | Description |
|------|-------------|
| `fgc_core.py` | The `solver_v2` class extracted **verbatim** from `FGC_experiment.ipynb`. It still reads the module-level globals `L` and `k` exactly as originally written, so the algorithm is byte-for-byte unchanged. A header records its origin. |
| `run_fgc.py` | New CLI driver: loads exactly one dataset, builds `L = D − A`, runs the solver, and reports loss, Dirichlet-energy traces, relative eigen error, and non-empty supernode count. Includes a NumPy alias shim so `deeprobust` imports. |

`run_fgc.py` also exposes `--inner_iters`. Omitting it preserves the notebook's
hardcoded 100 inner steps; lowering it trades fidelity for speed during smoke tests.

The three original notebooks are unmodified.

---

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install "numpy<2" torch torch_geometric scipy scikit-learn \
            networkx pygsp matplotlib seaborn tqdm pandas
pip install deeprobust --no-deps    # dataset loading
```

Verified with: `torch 2.8.0`, `torch_geometric 2.6.1`, `numpy 1.26.4`, `scipy 1.13.1`, `pygsp 0.6.1`, `deeprobust 0.2.11`.

> Install `deeprobust` with `--no-deps`: its pinned requirements would otherwise downgrade PyTorch.

## Running

```bash
# faithful reproduction (paper default: 100 inner steps per iteration)
python -W ignore run_fgc.py --dataset cora --ratio 0.3 --iterations 10 --eig

# fast smoke test
python -W ignore run_fgc.py --dataset citeseer --ratio 0.5 \
  --iterations 2 --inner_iters 10 --eig
```

**Arguments**

| Flag | Meaning |
|------|---------|
| `--dataset` | `cora`, `citeseer`, `polblogs` (via deeprobust) · `airfoil`, `minnesota`, `bunny` (via PyGSP) |
| `--ratio` | coarsening ratio `r`; the coarse graph has `k = r·n` supernodes |
| `--iterations` | outer iterations of the FGC objective |
| `--inner_iters` | inner `update_C` steps per iteration (default: 100, as upstream) |
| `--lambda_param`, `--alpha_param`, `--gamma_param` | FGC's three hyperparameters (`gamma` defaults to `X.shape[1]/2`, as in the notebook) |
| `--eig` | also compute the relative eigen error (dense eigendecomposition; slow) |

Datasets download automatically on first use.

### Performance and memory
- **~1 min/iteration at `r=0.3`, ~2 min/iteration at `r=0.5`** on CPU. Each of the 100 inner steps performs a dense `k × k` pseudo-inverse plus `n × n` matrix products.
- **Memory-hungry:** several dense 2708×2708 and 2708×1354 matrices are held at once. Close other applications, or the process will swap heavily (an observed run degraded from 112 s to 1,649 s per iteration under memory pressure).

---

## How FGC works (brief)

FGC jointly learns the coarsening matrix `C` and the coarsened feature matrix `X̃`
by minimising a single objective containing:

- `tr(X̃ᵀ Cᵀ L C X̃)` — a smoothness / Dirichlet-energy term,
- `−γ · log det(Cᵀ L C + J)` — a log-determinant term keeping the coarse graph connected and non-degenerate,
- `(α/2)‖X − C X̃‖²` — feature reconstruction,
- `(λ/2)‖C 1‖²` — an ℓ₁,₂ penalty encouraging a clean node-to-supernode assignment.

Optimisation alternates: gradient steps on `C` (projected to stay non-negative and
row-normalised), then a closed-form update of `X̃`. Unlike purely spectral methods,
FGC uses node **features** as well as graph structure, and comes with
ε-similarity guarantees between the original and coarsened graphs.

### The other notebooks (unmodified)
- `Local variation.ipynb` — baselines: Loukas local variation (LVE / LVN), Kron reduction, HEM, with REE / DE / HE / RE metrics. Support code in `libraries/`.
- `Node_classification_FGC_with_sparsity.ipynb` — node classification with FGC + sparsity, written for CUDA (would need adapting for CPU).

---

## Licence and attribution

Upstream is **Apache-2.0** licensed; the original [`LICENSE`](LICENSE) is retained
here, as required. `fgc_core.py` is the authors' code, extracted verbatim and
attributed in its header. `run_fgc.py` is my own contribution.

If you use this method, please cite the original paper:

```bibtex
@inproceedings{kumar2023featured,
  title     = {Featured Graph Coarsening with Similarity Guarantees},
  author    = {Kumar, Manoj and Sharma, Anurag and Saxena, Shashwat and Kumar, Sandeep},
  booktitle = {International Conference on Machine Learning (ICML)},
  year      = {2023}
}
```

The bundled `libraries/` code implements
**Loukas, *Graph Reduction with Spectral and Cut Guarantees*, JMLR 2019**
([code](https://github.com/loukasa/graph-coarsening)), used as the comparison baseline.

*Authors: if you would prefer any part of this handled differently, please open an issue.*
