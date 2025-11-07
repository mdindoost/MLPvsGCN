![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.2-red.svg)
![PyG](https://img.shields.io/badge/PyTorch_Geometric-2.5-orange.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Status](https://img.shields.io/badge/status-Active-success.svg)
![Made with ❤️](https://img.shields.io/badge/Made%20with-%F0%9F%96%A4%20and%20%F0%9F%A4%9E-blueviolet)

# MLPvsGCN — Demo on **Why Edges Matter**

This repo is a compact, **reproducible** demo showing why a plain **MLP** underperforms a **GCN** on node classification (Cora), and how progressively injecting **graph-derived signals** (degree, clustering coefficient, k-core, PageRank, **Laplacian Positional Encodings**), and/or doing **SGC-style pre-propagation** makes an edge-free MLP behave more and more like a GCN.

No rigid step numbers here—the teaching arc matters more than the checklist. The script prints clear section headers and short “Narration / Insight” lines so you can talk through the evolution naturally.

---

## What you’ll see

- **GCN vs MLP** on Cora with identical splits and seeds.
- A **ladder of structure** for the MLP, added one hint at a time:
  - + **Degree** (local crowd size)
  - + **Clustering** (triangle density)
  - + **k-Core** (meso-scale cohesion)
  - + **PageRank** (global influence)
  - + **LPE (k Laplacian eigenvectors)**: smooth positional coordinates in graph space
- A **consolidated recap** that appends all features (and optionally +8D LPE) at once.
- A **robustness** rerun with a random 60/20/20 split.
- **Epoch sensitivity** (e.g., 50 vs 200 epochs) to illustrate optimization vs generalization.
- **Richer LPE (32D)** to span more of the smooth subspace.
- **SGC pre-prop** (K=2): compute \\(\\tilde{X}=(\\hat{A})^K X\\) once, then train a plain MLP (no edges during training).
- **Variant**: train MLP on **[X | SGC(X)]** to closely mimic GCN behavior while still edge-free at train time.
- **Optional** 2-hop ego feature to showcase a light second-order structural cue.
- Optional **summary bar chart** if `matplotlib` is available.

Along the way, the script also reports:
- **Connected components, giant component, diameter, average degree**.
- **Label homophily** (fraction of edges whose endpoints share the same label).
- **Neighbor agreement** of model predictions.
- **Hidden activation samples** (to demystify “these are activations, not probabilities”).

---

## Installation

### pip (recommended; CPU or CUDA wheel available from PyTorch)

```bash
# Create and activate a Python 3 virtual environment
python3 -m venv gnn_env
source gnn_env/bin/activate   # Windows: gnn_env\Scripts\activate

# Upgrade pip inside the venv (avoid system-wide installs)
python -m pip install --upgrade pip

# Install PyTorch (choose one line)
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
# or (example) CUDA 12.1:
# python -m pip install torch --index-url https://download.pytorch.org/whl/cu121

# Install remaining dependencies
python -m pip install -r requirements.txt
```

> **Note:** `torch-geometric` may fetch small extra wheels on first import. The meta-package `torch-geometric` plus `networkx`, `scipy`, and `matplotlib` are enough for this demo.

### conda (optional)

```bash
conda create -n gnn_env python=3.11 -y
conda activate gnn_env
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install torch-geometric networkx scipy matplotlib scikit-learn
```

---

## Quick start

```bash
# Main runnable script with the lecture-friendly ladder, structural overview,
# consolidated recap, robustness & sensitivity checks, LPE(32), and SGC variants.
python cora_demo_stepwise_v3.py
```

The script will:
- Load **Cora** (public split).
- Print a **structural overview**: number of connected components, largest component size, **diameter**, **average degree**, and **label homophily** (fraction of edges whose endpoints share the same label).
- Train **GCN** and **MLP (raw)** and compare.
- Walk an **MLP ladder** where one structural hint is added at a time (+degree → +clustering → +k-core → +PageRank → +LPE). Each rung prints **Narration** (what changed) and **Insight** (why the metric moved).
- Run a **consolidated recap** that appends all features (and optionally +8D LPE) and retrains.
- Check **robustness** on a random 60/20/20 split.
- Probe **epoch sensitivity**.
- Explore **LPE(32)** for a richer smooth basis.
- Try **SGC pre-prop (K=2)** and a **[X | SGC(X)]** variant.
- Optionally, add a lightweight **2-hop ego** feature.
- Save a tiny **bar chart** (`accuracy_summary.png`) if `matplotlib` is installed.

---

## Why can “adding features” hurt initially?

Adding degree, clustering, and core **increases dimensionality** and changes feature distributions—but an MLP still has **no relational inductive bias**. With small training splits, it can **overfit** the new numeric columns or get confused by **correlated** structural stats.  
GCNs avoid this by **smoothing over neighbors** via message passing. In this demo you’ll see occasional drops when adding early features, followed by clear improvements once we introduce **positional priors (LPE)** or **pre-diffused features (SGC)** that *integrate* structure rather than just appending columns.

---

## What are Laplacian Positional Encodings (LPE)?

- Build the (normalized) Laplacian: \\(L = I - D^{-1/2} A D^{-1/2}\\).
- Solve \\(L v_i = \\lambda_i v_i\\) and take the first \\(k\\) eigenvectors (smallest \\(\\lambda\\)).
- Stack them into an \\(N \\times k\\) matrix = **k smooth coordinates** per node.  
  Low-frequency modes capture **large-scale structure**; concatenating them gives the MLP a **positional prior**.
- Using more (e.g., **32D**) spans more of the smooth subspace.

**Projection vs Spanning?** All eigenvectors span all node functions; picking the first \\(k\\) is a **projection** into the low-frequency subspace.

---

## What is SGC-style pre-propagation?

**Simple Graph Convolution (SGC)** collapses message passing and linear layers into a single preprocessing step:

\\[
\\tilde{X} = (\\hat{A})^{K} X, \\quad \\hat{A} = D^{-1/2}(A+I)D^{-1/2}.
\\]

Compute \\(\\tilde{X}\\) once, then train a plain **MLP** on \\(\\tilde{X}\\).  
This injects the **smoothness prior** while keeping training **edge-free**. A practical variant is to train on **[X | SGC(X)]**, which often gets very close to GCN behavior.

---

## Expected ballpark (seed- & version-dependent)

| Model | Uses edges *during training*? | Extra features | Test Acc (Cora) |
|------|-------------------------------|----------------|------------------|
| MLP (raw) | No | – |  |
| MLP + graph feats (+8 LPE) | No | deg, clustering, core, PR (+LPE) |  |
| MLP + SGC-preprop (K=2) | No (edges offline) | smoothed X | |
| GCN (2-layer) | Yes | – |  |


---

## Files

- `cora_demo_stepwise_v3.py` — main runnable script with the ladder, structural overview, consolidated recap, robustness/sensitivity checks, LPE(32), and SGC variants.
- `requirements.txt` — minimal pip requirements.
- `LICENSE` — MIT.
- `.gitignore` — typical Python ignores.

---

## FAQ snippets you can quote during the lecture

- **“What exactly is label homophily here?”**  
  The **fraction of edges whose endpoints share the same label** (computed on the undirected graph).

- **“Why did accuracy dip after adding degree/clustering/core?”**  
  More columns without edge-based smoothing can amplify noise and redundancy; with small train splits, the MLP overfits. LPE or SGC add **structure in a way the model can actually use**.

- **“Is PageRank always helpful?”**  
  It adds a faint **multi-hop** signal, so expect **small** but sometimes consistent bumps—not a silver bullet without message passing.

- **“Why z-score after concatenation?”**  
  To give new columns a **fair voice**; raw BoW dominance or scale mismatch can otherwise drown them out.

---

## How to cite

> **Mohammad Dindoost.** *MLPvsGCN — A Lecture-Ready Demo on Why Edges Matter.* GitHub repository, 2025.

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

---

## License

MIT — do what you want, just keep the notice.
