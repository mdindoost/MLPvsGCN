# cora_demo_stepwise_v2.py
# Stepwise lecture demo: GCN vs MLP, then inject structure into MLP.
# Adds: robust LPE (eigenvectors) explanation, z-scoring, neighbor-agreement metric,
# clear feature-dimension reporting, and SGC preprop with commentary.

import random, os, sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv
from torch_geometric.utils import to_networkx, add_self_loops, degree

import networkx as nx

try:
    import matplotlib.pyplot as plt
    HAS_MPL = True
except Exception:
    HAS_MPL = False

# ----------------------------
# Reproducibility
# ----------------------------
SEED = 42
random.seed(SEED); np.random.seed(SEED)
torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ----------------------------
# Small helpers
# ----------------------------
def section(title):
    print("\n" + "="*len(title))
    print(title)
    print("="*len(title))

def count_masks(data):
    return (int(data.train_mask.sum()), int(data.val_mask.sum()), int(data.test_mask.sum()))

def print_feature_samples(X, y, k=3, cols=12, prefix="sample"):
    Xc = X.cpu().numpy(); yc = y.cpu().numpy()
    idxs = np.linspace(0, Xc.shape[0]-1, k, dtype=int)

    density = (Xc != 0).sum() / Xc.size
    print(f"\n{prefix}: feature dims = {Xc.shape[1]}, density ≈ {density*100:.2f}%")

    for i in idxs:
        row = Xc[i]
        nz = np.flatnonzero(row)
        nnz = nz.size
        show_k = min(8, nnz)
        nz_pairs = " ".join([f"{j}:{row[j]:.0f}" for j in nz[:show_k]]) if nnz > 0 else "(no nonzeros)"
        raw_slice = " ".join([f"{v: .3f}" for v in row[:cols]])
        print(f"node {i:4d} | y={yc[i]} | nnz={nnz} | nonzeros: {nz_pairs} | x[:{cols}]=[{raw_slice}]")

def print_embedding_samples(H, y, k=3, cols=8, prefix="embed"):
    Hc = H.detach().cpu().numpy(); yc = y.cpu().numpy()
    idxs = np.linspace(0, Hc.shape[0]-1, k, dtype=int)
    print(f"\n{prefix} (showing first {cols} dims, NOT probabilities):")
    for i in idxs:
        fx = " ".join([f"{v: .3f}" for v in Hc[i, :cols]])
        print(f"node {i:4d} | h[:{cols}] = [{fx}] | y = {yc[i]}")

def neighbor_agreement(pred, edge_index):
    row, col = edge_index
    return (pred[row] == pred[col]).float().mean().item()

@torch.no_grad()
def eval_accs(model, data, use_edges=True):
    model.eval()
    out = model(data.x, data.edge_index) if use_edges else model(data.x)
    pred = out.argmax(dim=1)
    def acc(mask): return (pred[mask] == data.y[mask]).float().mean().item()
    return acc(data.train_mask), acc(data.val_mask), acc(data.test_mask), pred, out

def train(model, data, use_edges=True, lr=0.01, weight_decay=5e-4, epochs=200):
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    for _ in range(epochs):
        model.train()
        opt.zero_grad()
        out = model(data.x, data.edge_index) if use_edges else model(data.x)
        loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
        loss.backward(); opt.step()

def make_random_split(data, train_frac=0.6, val_frac=0.2, test_frac=0.2):
    N = data.num_nodes
    perm = torch.randperm(N, device=data.x.device)
    n_train = int(train_frac * N); n_val = int(val_frac * N)
    train_idx = perm[:n_train]; val_idx = perm[n_train:n_train+n_val]; test_idx = perm[n_train+n_val:]
    data.train_mask = torch.zeros(N, dtype=torch.bool, device=data.x.device); data.train_mask[train_idx] = True
    data.val_mask   = torch.zeros(N, dtype=torch.bool, device=data.x.device); data.val_mask[val_idx] = True
    data.test_mask  = torch.zeros(N, dtype=torch.bool, device=data.x.device); data.test_mask[test_idx] = True
    return data

def zscore_columns(X, eps=1e-6):
    m = X.mean(dim=0, keepdim=True)
    s = X.std(dim=0, keepdim=True)
    return (X - m) / (s + eps)

def col_l2_mean(X):
    return X.pow(2).mean(dim=0).sqrt().mean().item()

# ----------------------------
# Laplacian Positional Encodings (LPE)
# ----------------------------
def laplacian_eigenvectors(edge_index, num_nodes, k=8, normalized=True, device='cpu'):
    """
    Compute the first k eigenvectors of the (normalized) graph Laplacian.
    - If normalized=True: L_sym = I - D^{-1/2} A D^{-1/2}
    - Else: L = D - A
    Returns: tensor (N, k) on 'device'
    """
    import scipy.sparse as sp
    import scipy.sparse.linalg as spla

    row, col = edge_index.cpu().numpy()
    data = np.ones_like(row, dtype=np.float64)
    A = sp.coo_matrix((data, (row, col)), shape=(num_nodes, num_nodes))
    A = A + A.T
    A.data[:] = 1.0
    degs = np.array(A.sum(1)).ravel()
    if normalized:
        # L_sym = I - D^{-1/2} A D^{-1/2}
        deg_inv_sqrt = 1.0 / np.sqrt(np.maximum(degs, 1e-12))
        D_is = sp.diags(deg_inv_sqrt)
        S = D_is @ A @ D_is
        L = sp.eye(num_nodes, dtype=np.float64) - S
    else:
        D = sp.diags(degs)
        L = D - A

    # Small diagonal shift improves numerical stability
    shift = 1e-5
    L = L + shift * sp.eye(num_nodes, dtype=np.float64)

    # Compute k smallest-magnitude eigenpairs
    k_eff = min(k, num_nodes - 2) if num_nodes > 2 else 1
    evals, evecs = spla.eigsh(L, k=k_eff, which='SM')
    V = torch.tensor(evecs, dtype=torch.float32, device=device)
    return V

# ----------------------------
# Models
# ----------------------------
class GCN(nn.Module):
    def __init__(self, in_dim, hid=16, out_dim=None, dropout=0.5):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hid)
        self.conv2 = GCNConv(hid, out_dim)
        self.dropout = dropout
    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x
    def hidden(self, x, edge_index):
        return F.relu(self.conv1(x, edge_index))

class MLP(nn.Module):
    def __init__(self, in_dim, hid=64, out_dim=None, dropout=0.6):
        super().__init__()
        self.lin1 = nn.Linear(in_dim, hid)
        self.lin2 = nn.Linear(hid, out_dim)
        self.dropout = dropout
    def forward(self, x):
        h = F.relu(self.lin1(x))
        h = F.dropout(h, p=self.dropout, training=self.training)
        return self.lin2(h)
    def hidden(self, x):
        return F.relu(self.lin1(x))

# ----------------------------
# Load data & show samples
# ----------------------------
print("(gnn_env) md724@ml:~/graph-demo$ python cora_demo_stepwise_v2.py")

dataset = Planetoid(root='/tmp/Cora', name='Cora')
data = dataset[0].to(device)

print("\n=== Dataset: Cora ===")
print("Cora is a citation network with:")
print(" - 2,708 papers (nodes)")
print(" - 5,429 citation links (edges, undirected)")
print(" - 1,433 binary word indicators as node features")
print(" - 7 research topic classes (e.g., Neural_Networks, Rule_Learning)")
print("Each node’s feature vector represents the presence/absence of certain words in the paper’s abstract, bag-of-words (BoW).")
print("======================\n")

print("A few sample rows of data (first 3 nodes, 12 feature dims):")
print_feature_samples(data.x, data.y, k=3, cols=12, prefix="Raw Cora rows")

t_count, v_count, te_count = count_masks(data)
print(f"\nSplit sizes (public split) → train: {t_count}, val: {v_count}, test: {te_count}")
print("Note: tiny train set vs model capacity → easy to hit train=1.000; generalization is what matters.\n")

print("\n\nIf you want a more programmatic summary (As in notebooks):")
print(data)
print(f"Feature matrix shape: {data.x.shape}")
print(f"Edge index shape: {data.edge_index.shape}")
print(f"Number of classes: {dataset.num_classes}")
print(f"Feature dimension: {dataset.num_node_features}")
# ----------------------------
# Step 1: GCN
# ----------------------------
section("Step 1: Train GCN (uses edges)")
gcn = GCN(in_dim=dataset.num_node_features, out_dim=dataset.num_classes, dropout=0.5).to(device)
train(gcn, data, use_edges=True, epochs=200, weight_decay=5e-4)
gcn_train, gcn_val, gcn_test, gcn_pred, gcn_logits = eval_accs(gcn, data, use_edges=True)
print(f"GCN        -> train {gcn_train:.3f} | val {gcn_val:.3f} | test {gcn_test:.3f}")

print("\nDescribe:")
print("- Downstream task: node classification.")
print("- GCN mixes a node’s features with its neighbors (message passing).")
print("- Hidden activations h = ReLU(Â X W1). Values like 0.999 are post-ReLU activations, not probabilities.")

gcn_H = gcn.hidden(data.x, data.edge_index)
print_embedding_samples(gcn_H, data.y, k=3, cols=8, prefix="GCN hidden embeddings")

with torch.no_grad():
    print("\nNeighbor agreement (fraction of edges with same predicted label):")
    print("GCN:", neighbor_agreement(gcn_pred, data.edge_index))

print(f"\n(Also reminding) Split sizes -> train: {t_count}, val: {v_count}, test: {te_count}")


print("""
Note:\n
The Graph Convolutional Network (GCN) we trained earlier has two layers:
    h1 = ReLU(Â X W1)
    out = Â h1 W2
where Â = D^{-1/2} (A + I) D^{-1/2} mixes each node with its neighbors.

That means every hidden representation h1[i] depends on the node's own features
and its neighbors' features.

We'll now build an MLP with the *same* depth and width (two linear layers, same hidden dim),
but it will NOT use Â or A at all.
Each node will be processed independently — as if the graph structure didn’t exist.
""")

# ----------------------------
# Step 2: MLP (raw)
# ----------------------------
section("Step 2: Train MLP (raw features only, ignores edges)")
mlp_raw = MLP(in_dim=dataset.num_node_features, out_dim=dataset.num_classes, dropout=0.6).to(device)
train(mlp_raw, data, use_edges=False, epochs=200, weight_decay=5e-3)
mlp_train, mlp_val, mlp_test, mlp_pred, mlp_logits = eval_accs(mlp_raw, data, use_edges=False)
print(f"MLP (raw)  -> train {mlp_train:.3f} | val {mlp_val:.3f} | test {mlp_test:.3f}")

print("\nDescribe:")
print("- MLP treats nodes independently; h = ReLU(X W1).")
print("- Perfect train, poor test → overfitting and lack of relational signal.")

mlp_H = mlp_raw.hidden(data.x)
print_embedding_samples(mlp_H, data.y, k=3, cols=8, prefix="MLP(hidden) on raw features")
# ----------------------------
# Triplet cosine-sim demo (GCN vs MLP) — deterministic, 3 examples
# ----------------------------
def _unique_undirected_edges(edge_index):
    row, col = edge_index
    s = set()
    for u, v in zip(row.tolist(), col.tolist()):
        if u == v:
            continue
        a, b = (u, v) if u < v else (v, u)
        s.add((a, b))
    return sorted(list(s))

def _adjacency_sets(edge_index, num_nodes):
    row, col = edge_index
    adj = [set() for _ in range(num_nodes)]
    S = set()
    for u, v in zip(row.tolist(), col.tolist()):
        if u == v:
            continue
        adj[u].add(v); adj[v].add(u)
        S.add((u, v)); S.add((v, u))
    return adj, S

def _first_non_neighbor(u, adj, N):
    # deterministic scan
    for w in range(N):
        if w != u and (w not in adj[u]):
            return w
    return (u + 1) % N  # fallback (dense graphs)

@torch.no_grad()
def _cosine(a, b, eps=1e-9):
    an = a / (a.norm() + eps)
    bn = b / (b.norm() + eps)
    return float((an * bn).sum())

@torch.no_grad()
def show_three_triplets_compare_gcn_mlp(H_gcn, H_mlp, edge_index, y):
    N = H_gcn.size(0)
    edges = _unique_undirected_edges(edge_index)
    adj, _ = _adjacency_sets(edge_index, N)
    print("\nTriplet similarity (3 deterministic examples):")
    print("For each anchor i, compare neighbor j vs non-neighbor k in hidden space.")
    count = 0
    for (u, v) in edges:
        i, j = u, v
        k = _first_non_neighbor(i, adj, N)
        cos_gcn_ij = _cosine(H_gcn[i], H_gcn[j])
        cos_gcn_ik = _cosine(H_gcn[i], H_gcn[k])
        cos_mlp_ij = _cosine(H_mlp[i], H_mlp[j])
        cos_mlp_ik = _cosine(H_mlp[i], H_mlp[k])
        print(f"  anchor i={i} (y={{int(y[i])}}) | neighbor j={j} (y={{int(y[j])}}) | non-neighbor k={k} (y={{int(y[k])}})")
        print(f"    GCN: cos(i,j)={cos_gcn_ij:.3f}  vs  cos(i,k)={cos_gcn_ik:.3f}")
        print(f"    MLP: cos(i,j)={cos_mlp_ij:.3f}  vs  cos(i,k)={cos_mlp_ik:.3f}")
        count += 1
        if count == 3:
            break
    # Aggregate mean cosine on first 100 edges
    import numpy as _np
    sims_gcn_e, sims_gcn_ne, sims_mlp_e, sims_mlp_ne = [], [], [], []
    for (u, v) in edges[:100]:
        k = _first_non_neighbor(u, adj, N)
        sims_gcn_e.append(_cosine(H_gcn[u], H_gcn[v]))
        sims_gcn_ne.append(_cosine(H_gcn[u], H_gcn[k]))
        sims_mlp_e.append(_cosine(H_mlp[u], H_mlp[v]))
        sims_mlp_ne.append(_cosine(H_mlp[u], H_mlp[k]))
    if sims_gcn_e:
        print("Aggregate mean cosine (first 100 edges):")
        print(f"    GCN: edges={_np.mean(sims_gcn_e):.3f} | non-edges={_np.mean(sims_gcn_ne):.3f}")
        print(f"    MLP: edges={_np.mean(sims_mlp_e):.3f} | non-edges={_np.mean(sims_mlp_ne):.3f}")

# Run comparison now that we have both hidden reps
show_three_triplets_compare_gcn_mlp(gcn_H, mlp_H, data.edge_index, data.y)


with torch.no_grad():
    print("\nNeighbor agreement (MLP raw):", neighbor_agreement(mlp_pred, data.edge_index))

# ----------------------------
# Step 3: Add graph features (degree/clustering/core/PR + LPE) -> MLP
# ----------------------------
section("Step 3: Inject simple graph features into the MLP")

G = to_networkx(data, to_undirected=True)
N = data.num_nodes

deg = np.array([G.degree(n) for n in range(N)], dtype=float)
deg = deg / (deg.max() + 1e-12)

clust = nx.clustering(G)
clust = np.array([clust[i] for i in range(N)], dtype=float)

core = nx.core_number(G)
core = np.array([core[i] for i in range(N)], dtype=float); core = core / (core.max() + 1e-12)

pr = nx.pagerank(G, alpha=0.85, tol=1e-6)
pr = np.array([pr[i] for i in range(N)], dtype=float)

graph_feats = torch.tensor(np.stack([deg, clust, core, pr], axis=1),
                           dtype=torch.float32, device=device)

# Laplacian PE: low-frequency positional basis (spans smooth structure)
lpe_note = ""
try:
    V8 = laplacian_eigenvectors(data.edge_index, N, k=8, normalized=True, device=device)
    graph_feats = torch.cat([graph_feats, V8], dim=1)
    lpe_note = " + 8D LPE"
except Exception as e:
    print("LPE(8) skipped:", e)

data_aug = data.clone()
data_aug.x = torch.cat([data.x, graph_feats], dim=1)

# Report feature counts before/after + show a few columns
print("\nFeature dimensions:")
print(f"- Original feature dim: {data.x.size(1)}")
print(f"- Added graph columns: 4{' + 8 LPE' if lpe_note else ''}")
print(f"- New total dim: {data_aug.x.size(1)}")

Xa = data_aug.x.detach().cpu().numpy()
yv = data_aug.y.cpu().numpy()
orig_cols = 6
added_names = ["deg","clust","core","pr"] + ([f"lpe{i+1}" for i in range(8)] if lpe_note else [])
print(f"\nColumns shown: x0..x{orig_cols-1} + {', '.join(added_names[:6])} ...")
for i in np.linspace(0, N-1, 3, dtype=int):
    base = " ".join([f"{v: .3f}" for v in Xa[i, :orig_cols]])
    start = data.x.size(1); width = min(6, graph_feats.size(1))
    added_part = Xa[i, start:start+width]
    added = " ".join([f"{v: .3f}" for v in added_part])
    print(f"node {i:4d} | x0..x{orig_cols-1}=[{base}] | added=[{added}] | y={yv[i]}")

# SCALE MATTERS: z-score the concatenated features so new columns have a say
print("\nScale probe (mean L2 across columns):")
print("  original:", col_l2_mean(data.x))
print("  added   :", col_l2_mean(graph_feats))
print("  concat(before z):", col_l2_mean(data_aug.x))
data_aug.x = zscore_columns(data_aug.x)
print("  concat( after z):", col_l2_mean(data_aug.x))

mlp_aug = MLP(in_dim=data_aug.x.size(1), out_dim=dataset.num_classes, dropout=0.6).to(device)
train(mlp_aug, data_aug, use_edges=False, epochs=200, weight_decay=5e-3)
aug_train, aug_val, aug_test, aug_pred, _ = eval_accs(mlp_aug, data_aug, use_edges=False)
print(f"\nMLP + graph feats{lpe_note} (z-scored) -> train {aug_train:.3f} | val {aug_val:.3f} | test {aug_test:.3f}")

with torch.no_grad():
    print("Neighbor agreement (MLP + feats):", neighbor_agreement(aug_pred, data.edge_index))

# ----------------------------
# Diagnose overfitting -> new split 60/20/20
# ----------------------------
print("\nWe test robustness with larger train set (random 60/20/20 split).")
data_602020 = data.clone()
data_602020 = make_random_split(data_602020, 0.6, 0.2, 0.2)
t2, v2, te2 = count_masks(data_602020)
print(f"New random split sizes -> train: {t2}, val: {v2}, test: {te2}")

mlp_raw2 = MLP(in_dim=dataset.num_node_features, out_dim=dataset.num_classes, dropout=0.6).to(device)
train(mlp_raw2, data_602020, use_edges=False, epochs=200, weight_decay=5e-3)
r2_train, r2_val, r2_test, _, _ = eval_accs(mlp_raw2, data_602020, use_edges=False)
print(f"MLP(raw) on 60/20/20 -> train {r2_train:.3f} | val {r2_val:.3f} | test {r2_test:.3f}")

# Reuse the same graph features; just swap the masks and re-zscore after concat
data_aug_602020 = data_602020.clone()
data_aug_602020.x = torch.cat([data.x, graph_feats], dim=1)
data_aug_602020.x = zscore_columns(data_aug_602020.x)

mlp_aug2 = MLP(in_dim=data_aug_602020.x.size(1), out_dim=dataset.num_classes, dropout=0.6).to(device)
train(mlp_aug2, data_aug_602020, use_edges=False, epochs=200, weight_decay=5e-3)
a2_train, a2_val, a2_test, _, _ = eval_accs(mlp_aug2, data_aug_602020, use_edges=False)
print(f"MLP(+feats{lpe_note}) on 60/20/20 -> train {a2_train:.3f} | val {a2_val:.3f} | test {a2_test:.3f}")

# ----------------------------
# Epochs sensitivity (50 vs 200)
# ----------------------------
print("\nEpoch sensitivity check (same split):")
mlp_aug_50 = MLP(in_dim=data_aug_602020.x.size(1), out_dim=dataset.num_classes, dropout=0.6).to(device)
train(mlp_aug_50, data_aug_602020, use_edges=False, epochs=50, weight_decay=5e-3)
b50_tr, b50_val, b50_te, _, _ = eval_accs(mlp_aug_50, data_aug_602020, use_edges=False)
print(f"MLP(+feats) 50 epochs -> train {b50_tr:.3f} | val {b50_val:.3f} | test {b50_te:.3f}")

mlp_aug_200 = MLP(in_dim=data_aug_602020.x.size(1), out_dim=dataset.num_classes, dropout=0.6).to(device)
train(mlp_aug_200, data_aug_602020, use_edges=False, epochs=200, weight_decay=5e-3)
b200_tr, b200_val, b200_te, _, _ = eval_accs(mlp_aug_200, data_aug_602020, use_edges=False)
print(f"MLP(+feats) 200 epochs -> train {b200_tr:.3f} | val {b200_val:.3f} | test {b200_te:.3f}")

# ----------------------------
# Stronger eigen features → 32D LPE (applied to MLP too)
# ----------------------------
print("\nExploring a richer positional basis (32D LPE). These eigenvectors span more of the graph's smooth subspace.")
lpe_note2 = ""
try:
    V32 = laplacian_eigenvectors(data.edge_index, N, k=32, normalized=True, device=device)
    graph_feats_stronger = torch.tensor(np.stack([deg, clust, core, pr], axis=1),
                                        dtype=torch.float32, device=device)
    graph_feats_stronger = torch.cat([graph_feats_stronger, V32], dim=1)

    data_aug2 = data_602020.clone()
    data_aug2.x = torch.cat([data.x, graph_feats_stronger], dim=1)
    data_aug2.x = zscore_columns(data_aug2.x)

    mlp_aug_strong = MLP(in_dim=data_aug2.x.size(1), out_dim=dataset.num_classes, dropout=0.6).to(device)
    train(mlp_aug_strong, data_aug2, use_edges=False, epochs=200, weight_decay=5e-3)
    s_tr, s_val, s_te, _, _ = eval_accs(mlp_aug_strong, data_aug2, use_edges=False)
    lpe_note2 = " + 32D LPE"
    print(f"MLP(+feats{lpe_note2}) on 60/20/20 -> train {s_tr:.3f} | val {s_val:.3f} | test {s_te:.3f}")
except Exception as e:
    print("Stronger LPE step skipped:", e)

# ----------------------------
# Bonus: SGC pre-propagation (still MLP, edges used offline only)
# ----------------------------
section("Bonus: MLP on pre-propagated features (SGC-style, still no edges at train time)")
def normalize_adj(edge_index, num_nodes):
    edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)
    row, col = edge_index
    degv = degree(col, num_nodes=num_nodes)
    deg_inv_sqrt = degv.pow(-0.5); deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
    norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
    return edge_index, norm

def prepropagate_features(X, edge_index, K=2):
    """
    SGC preprop:
      Replaces message passing during training by computing X_tilde = (Â)^K X offline.
      K=2 is common for homophilous graphs like Cora.
    """
    edge_index_norm, coeff = normalize_adj(edge_index, X.size(0))
    src, dst = edge_index_norm
    for _ in range(K):
        msg = coeff.unsqueeze(-1) * X[src]
        X = torch.zeros_like(X).index_add(0, dst, msg)
    return X

X_sgc = prepropagate_features(data_602020.x.clone(), data_602020.edge_index, K=2)
X_sgc = zscore_columns(X_sgc)  # keep scale comparable
data_sgc = data_602020.clone(); data_sgc.x = X_sgc
mlp_sgc = MLP(in_dim=X_sgc.size(1), out_dim=dataset.num_classes, dropout=0.6).to(device)
train(mlp_sgc, data_sgc, use_edges=False, epochs=200, weight_decay=5e-3)
sgc_tr, sgc_val, sgc_te, _, _ = eval_accs(mlp_sgc, data_sgc, use_edges=False)
print(f"MLP on SGC-preprop (K=2) -> train {sgc_tr:.3f} | val {sgc_val:.3f} | test {sgc_te:.3f}")

print("\nDescribe:")
print("- SGC-style preprocessing computes (Â)^K X once, then trains a vanilla MLP.")
print("- Edges do not participate in training; structure lives inside the features.")
print("- This often narrows the gap dramatically on homophilous graphs.\n")

# Optional mini bar chart to summarize (if matplotlib exists)
if HAS_MPL:
    try:
        labels = ["GCN(pub)", "MLP(raw/pub)", f"MLP(+feats{lpe_note}/pub)",
                  "MLP(raw/60-20-20)", f"MLP(+feats{lpe_note}/60-20-20)",
                  f"MLP(+feats{lpe_note}/50ep)", f"MLP(+feats{lpe_note}/200ep)",
                  f"MLP(+feats{lpe_note2}/60-20-20)" if lpe_note2 else " ",
                  "MLP(SGC K=2/60-20-20)"]
        vals = [
            gcn_test, mlp_test, aug_test,
            r2_test, a2_test, b50_te, b200_te,
            s_te if 's_te' in locals() else 0.0,
            sgc_te
        ]
        plt.figure(figsize=(10,4))
        plt.bar(range(len(labels)), vals)
        plt.xticks(range(len(labels)), labels, rotation=40, ha='right', fontsize=8)
        plt.ylabel("Test Acc")
        plt.title("Injecting structure into an MLP")
        plt.ylim(0, 1.0); plt.tight_layout(); plt.show()
    except Exception:
        pass
