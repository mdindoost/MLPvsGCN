# MLPvsGCN — A Lecture-Ready Demo on Why Edges Matter

This repo is a compact, **reproducible** demo showing why a plain **MLP** underperforms a **GCN** on node classification (Cora), and how injecting **graph-derived features** (degree, clustering coefficient, k-core, PageRank, and **Laplacian Positional Encodings**) or doing **SGC-style pre-propagation** narrows the gap.

---

## What you’ll see

- **GCN vs MLP** on Cora with identical splits.
- **MLP + Graph Features** (degree, clustering, core number, PageRank).
- **MLP + LPE (k eigenvectors of the Laplacian)**: graph-aware positional coordinates.
- **MLP + SGC preprop**: precompute \(\tilde{X}=(\hat{A})^K X\) offline, then train an MLP (no edges during training).

You’ll get clear printouts:
- Split sizes, training/validation/test accuracy.
- Hidden activations samples (not probabilities).
- Neighbor agreement (fraction of edges whose endpoints get the same predicted label).
- Optional summary bar chart if `matplotlib` is available.

---

## Installation

### Option A: pip (CPU or CUDA wheel available from PyTorch)

```bash
# Create and activate a Python 3 virtual environment
python3 -m venv gnn_env
source gnn_env/bin/activate   # Windows: gnn_env\Scripts\activate

# Upgrade pip inside the venv (never system-wide)
python -m pip install --upgrade pip

# Install PyTorch (choose CPU or CUDA)
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
# or, for CUDA 12.1:
# python -m pip install torch --index-url https://download.pytorch.org/whl/cu121

# Install remaining dependencies
python -m pip install -r requirements.txt

# Run the demo
python cora_demo_stepwise_v2.py
```

> **Note:** PyTorch Geometric may pull additional tiny wheels at first import; the single `torch-geometric` meta-package works well for Planetoid/Cora demos.

### Option B: conda (optional)

```bash
conda create -n gnn_env python=3.11 -y
conda activate gnn_env
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install torch-geometric networkx scipy matplotlib scikit-learn
```

---

## Quick start

```bash
python cora_demo_stepwise_v2.py
```

The script will:
1. Load **Cora** (public split: train=140, val=500, test=1000).
2. Train **GCN** and **MLP (raw)** and report accuracy.
3. Compute graph features (degree, clustering coefficient, k-core, PageRank) and **LPE** (first 8 Laplacian eigenvectors), concatenate, **z-score** all columns, then train **MLP(+feats)**.
4. Rerun with a **random 60/20/20 split** for robustness.
5. Try **32D LPE** (if SciPy eigensolver available).
6. Run **SGC preprop** (K=2) and train an MLP on \(\tilde{X}\).

If `matplotlib` is installed, you’ll also see a tiny bar chart summarizing test accuracy across steps.

---

## What are Laplacian Positional Encodings (LPE)?

- Build the (normalized) **graph Laplacian** \(L = I - D^{-1/2} A D^{-1/2}\).
- Solve the **eigenproblem** \(L v_i = \lambda_i v_i\).
- Take the first \(k\) eigenvectors (smallest eigenvalues). Stack them to get an \(N \times k\) matrix — **k coordinates per node**.
- These provide a **low-frequency, smooth basis** that *spans* the large-scale structure of the graph.
- Concatenating them to the node features gives an MLP a **positional prior** in “graph space.”

**Projection vs Spanning?** The full set of eigenvectors spans all node functions; taking the first *k* amounts to **projecting** into the low-frequency subspace.

---

## What is SGC-style pre-propagation?

**Simple Graph Convolution (SGC)** removes nonlinearities and collapses message passing into a single preprocessing step:

$$
\tilde{X} = (\hat{A})^{K} X, \quad \hat{A} = D^{-1/2}(A+I)D^{-1/2}.
$$

You compute $\tilde{X}$ once, then train a plain **MLP** on $\tilde{X}$.  
This isolates the value of structural smoothing **without** using edges during training.

---

## Expected ballpark (seed-dependent)

| Model | Uses edges during training? | Extra features | Test Acc (Cora) |
|------|------------------------------|----------------|------------------|
| MLP (raw) | No | – | ~0.55–0.60 |
| MLP + graph feats (+8 LPE) | No | deg, clustering, core, PR (+LPE) | ~0.62–0.72 |
| MLP + SGC-preprop (K=2) | No (edges offline) | smoothed X | ~0.80–0.87 |
| GCN (2-layer) | Yes | – | ~0.80–0.83 |

Numbers vary with seeds and package versions; the **ordering** is robust: **GCN ≥ SGC-preprop > MLP+feats > MLP**.

---

## Files

- `cora_demo_stepwise_v2.py` — main runnable script (clean prints, z-scoring, LPE utility, SGC preprop).
- `requirements.txt` — minimal pip requirements.
- `LICENSE` — MIT.
- `.gitignore` — typical Python ignores.

---

## How to cite

If you use this repo in teaching or research, please cite:

> **Mohammad Dindoost.** *MLPvsGCN — A Lecture-Ready Demo on Why Edges Matter.* GitHub repository, 2025.  
> https://github.com/mdindoost/MLPvsGCN

**BibTeX**
```bibtex
@software{Dindoost2025MLPvsGCN,
  author  = {Mohammad Dindoost},
  title   = {MLPvsGCN — A Lecture-Ready Demo on Why Edges Matter},
  year    = {2025},
  url     = {https://github.com/mdindoost/MLPvsGCN},
  note    = {Version 1.0}
}
```

## Support, issues, and discussion

- Open an issue: https://github.com/mdindoost/MLPvsGCN/issues  
- Email: md724@njit.edu

## Contributing and forking

Forks and PRs are welcome. Please open an issue first for major changes so we can discuss the design.

---

## License

MIT — do what you want, just keep the notice.