# cora_demo_stepwise_v3.py
# Stepwise lecture demo: GCN vs MLP, then evolve the MLP one hint at a time toward GCN-like behavior.
# Keeps your original flow, fixes numbering, adds structural overview, insight lines after each rung,
# and renames the consolidated block to "Step 4: Consolidated recap" to avoid numbering confusion.

import random, os, sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv, SAGEConv
from torch_geometric.utils import to_networkx, add_self_loops, degree, subgraph as tg_subgraph

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
    return (int(data.train_mask.sum()), int(data.val_mask.sum()), int(data.val_mask.sum()+0*0)+int(data.test_mask.sum())-int(data.val_mask.sum()))  # keep shape; not used, but preserved
    # ^ small safeguard: original function worked; here we won't rely on this return elsewhere

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

# New minimal helpers for inductive demo (kept very local)
@torch.no_grad()
def eval_accs_on_edgeindex(model, data, edge_index):
    """Evaluate a message-passing model on a custom edge_index (e.g., full graph at test time)."""
    model.eval()
    out = model(data.x, edge_index)
    pred = out.argmax(dim=1)
    def acc(mask): return (pred[mask] == data.y[mask]).float().mean().item()
    return acc, pred, out

def train_only_edge_index(model, data, edge_index, lr=0.01, weight_decay=5e-4, epochs=200):
    """Train a message-passing model using a custom edge_index (e.g., subgraph for inductive)."""
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    for _ in range(epochs):
        model.train(); opt.zero_grad()
        out = model(data.x, edge_index)
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

# Helper: split ONLY inside an allowed subset (used by Step 11)
def make_split_from_allowed_mask(mask_allowed, train_frac=0.6, val_frac=0.2, test_frac=0.2, device='cpu'):
    idx = torch.nonzero(mask_allowed, as_tuple=False).view(-1)
    N = idx.numel()
    perm = idx[torch.randperm(N, device=idx.device)]
    n_train = int(train_frac * N); n_val = int(val_frac * N)
    train_idx = perm[:n_train]; val_idx = perm[n_train:n_train+n_val]; test_idx = perm[n_train+n_val:]
    train_mask = torch.zeros(mask_allowed.size(0), dtype=torch.bool, device=device); train_mask[train_idx] = True
    val_mask   = torch.zeros_like(train_mask); val_mask[val_idx] = True
    test_mask  = torch.zeros_like(train_mask); test_mask[test_idx] = True
    return train_mask, val_mask, test_mask


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

# New: very small GraphSAGE for inductive demo
class SAGE(nn.Module):
    def __init__(self, in_dim, hid=64, out_dim=None, dropout=0.5):
        super().__init__()
        self.conv1 = SAGEConv(in_dim, hid)  # mean aggregator
        self.conv2 = SAGEConv(hid, out_dim)
        self.dropout = dropout
    def forward(self, x, edge_index):
        h = F.relu(self.conv1(x, edge_index))
        h = F.dropout(h, p=self.dropout, training=self.training)
        out = self.conv2(h, edge_index)
        return out

# ----------------------------
# Load data & show samples
# ----------------------------
print("(gnn_env) md724@ml:~/graph-demo$ python cora_demo_stepwise_v3.py")

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

t_count, v_count, te_count = (int(data.train_mask.sum()), int(data.val_mask.sum()), int(data.test_mask.sum()))
print(f"\nSplit sizes (public split) → train: {t_count}, val: {v_count}, test: {te_count}")
print("Note: tiny train set vs model capacity → easy to hit train=1.000; generalization is what matters.\n")

print("\n\nIf you want a more programmatic summary (As in notebooks):")
print(data)
print(f"Feature matrix shape: {data.x.shape}")
print(f"Edge index shape: {data.edge_index.shape}")
print(f"Number of classes: {dataset.num_classes}")
print(f"Feature dimension: {dataset.num_node_features}")

# ----------------------------
# Step 0: Cora structure at a glance
# ----------------------------
section("Step 0: Cora structure at a glance")
G = to_networkx(data, to_undirected=True)
N = data.num_nodes
M_undirected = G.number_of_edges()

# Connected components
comps = list(nx.connected_components(G))
num_cc = len(comps)
giant = G.subgraph(max(comps, key=len)).copy()

# Diameter only on the giant component
try:
    diam = nx.diameter(giant)
except Exception:
    diam = None

# Average degree (undirected: 2M/N)
avg_deg = 2.0 * M_undirected / N

# Ground-truth homophily (micro): fraction of edges whose endpoints share the same label
y_np = data.y.cpu().numpy()
gt_agree = [int(y_np[u] == y_np[v]) for u, v in G.edges()]
homophily = float(np.mean(gt_agree))

print(f"Connected components: {num_cc}")
print(f"Largest component size: {giant.number_of_nodes()} nodes, {giant.number_of_edges()} edges")
print(f"Diameter (largest component): {diam if diam is not None else 'N/A'}")
print(f"Average degree (undirected): {avg_deg:.2f}")
print(f"Label homophily (fraction of edges whose endpoints share the same label): {homophily:.3f}")

# ----------------------------
# Step 1: GCN (uses edges)
# ----------------------------
section("Step 1: Train GCN (uses edges)")
gcn = GCN(in_dim=dataset.num_node_features, out_dim=dataset.num_classes, dropout=0.5).to(device)
train(gcn, data, use_edges=True, epochs=200, weight_decay=5e-4)
gcn_train, gcn_val, gcn_test, gcn_pred, gcn_logits = eval_accs(gcn, data, use_edges=True)
print(f"GCN        -> train {gcn_train:.3f} | val {gcn_val:.3f} | test {gcn_test:.3f}")

print("- GCN mixes a node’s features with its neighbors (message passing).")
print("- Hidden activations h = ReLU(Â X W1). Values like 0.999 are post-ReLU activations, not probabilities.")

gcn_H = gcn.hidden(data.x, data.edge_index)
print_embedding_samples(gcn_H, data.y, k=3, cols=8, prefix="GCN hidden embeddings")

with torch.no_grad():
    print("\nNeighbor agreement (fraction of edges with same predicted label):")
    print("GCN:", neighbor_agreement(gcn_pred, data.edge_index))

# ----------------------------
# Step 2: MLP (raw) ignores edges
# ----------------------------
section("Step 2: Train MLP (raw features only, ignores edges)")
mlp_raw = MLP(in_dim=dataset.num_node_features, out_dim=dataset.num_classes, dropout=0.6).to(device)
train(mlp_raw, data, use_edges=False, epochs=200, weight_decay=5e-3)
mlp_train, mlp_val, mlp_test, mlp_pred, mlp_logits = eval_accs(mlp_raw, data, use_edges=False)
print(f"MLP (raw)  -> train {mlp_train:.3f} | val {mlp_val:.3f} | test {mlp_test:.3f}")

print("- MLP treats nodes independently; h = ReLU(X W1).")
print("- Perfect train, poorer test → overfitting and lack of relational signal.\n")

mlp_H = mlp_raw.hidden(data.x)
print_embedding_samples(mlp_H, data.y, k=3, cols=8, prefix="MLP(hidden) on raw features")

# ----------------------------
# Triplet cosine-sim demo (GCN vs MLP)
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
    for u, v in zip(row.tolist(), col.tolist()):
        if u == v:
            continue
        adj[u].add(v); adj[v].add(u)
    return adj

def _first_non_neighbor(u, adj, N):
    for w in range(N):
        if w != u and (w not in adj[u]):
            return w
    return (u + 1) % N

@torch.no_grad()
def _cosine(a, b, eps=1e-9):
    an = a / (a.norm() + eps)
    bn = b / (b.norm() + eps)
    return float((an * bn).sum())

@torch.no_grad()
def show_three_triplets_compare_gcn_mlp(H_gcn, H_mlp, edge_index, y):
    N = H_gcn.size(0)
    edges = _unique_undirected_edges(edge_index)
    adj = _adjacency_sets(edge_index, N)
    print("\nTriplet similarity (3 deterministic examples):")
    print("For each anchor i, compare neighbor j vs non-neighbor k in hidden space.\n")
    count = 0
    for (u, v) in edges:
        i, j = u, v
        k = _first_non_neighbor(i, adj, N)
        cos_gcn_ij = _cosine(H_gcn[i], H_gcn[j])
        cos_gcn_ik = _cosine(H_gcn[i], H_gcn[k])
        cos_mlp_ij = _cosine(H_mlp[i], H_mlp[j])
        cos_mlp_ik = _cosine(H_mlp[i], H_mlp[k])
        print(f"  anchor i={i} (y={int(y[i])}) | neighbor j={j} (y={int(y[j])}) | non-neighbor k={k} (y={int(y[k])})")
        print(f"    GCN: cos(i,j)={cos_gcn_ij:.3f}  vs  cos(i,k)={cos_gcn_ik:.3f}")
        print(f"    MLP: cos(i,j)={cos_mlp_ij:.3f}  vs  cos(i,k)={cos_mlp_ik:.3f}")
        count += 1
        if count == 3:
            break
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

show_three_triplets_compare_gcn_mlp(gcn_H, mlp_H, data.edge_index, data.y)
with torch.no_grad():
    print("\nNeighbor agreement (MLP raw):", neighbor_agreement(mlp_pred, data.edge_index))

# ----------------------------
# Step 3: The ladder — one structural hint at a time (public split)
# ----------------------------
# 3A) + degree
section("Step 3A: MLP + degree (one new hint)")
deg = np.array([G.degree(n) for n in range(N)], dtype=float)
deg = deg / (deg.max() + 1e-12)
deg_t = torch.tensor(deg, dtype=torch.float32, device=device).unsqueeze(1)

data_deg = data.clone()
data_deg.x = torch.cat([data.x, deg_t], dim=1)
data_deg.x = zscore_columns(data_deg.x)

mlp_deg = MLP(in_dim=data_deg.x.size(1), out_dim=dataset.num_classes, dropout=0.6).to(device)
train(mlp_deg, data_deg, use_edges=False, epochs=200, weight_decay=5e-3)
d_tr, d_val, d_te, d_pred, _ = eval_accs(mlp_deg, data_deg, use_edges=False)
print(f"MLP + degree -> train {d_tr:.3f} | val {d_val:.3f} | test {d_te:.3f}")
print("Neighbor agreement (MLP + degree):", neighbor_agreement(d_pred, data.edge_index))
print("Narration: Now it knows how crowded its block is (local popularity).")
print("Insight: More features ≠ better generalization without relational bias; with tiny train splits, the MLP can overfit new numeric columns.")

# 3B) + clustering
section("Step 3B: MLP + degree + clustering")
clust = nx.clustering(G)
clust = np.array([clust[i] for i in range(N)], dtype=float)
clust_t = torch.tensor(clust, dtype=torch.float32, device=device).unsqueeze(1)

data_deg_cl = data.clone()
data_deg_cl.x = torch.cat([data.x, deg_t, clust_t], dim=1)
data_deg_cl.x = zscore_columns(data_deg_cl.x)

mlp_deg_cl = MLP(in_dim=data_deg_cl.x.size(1), out_dim=dataset.num_classes, dropout=0.6).to(device)
train(mlp_deg_cl, data_deg_cl, use_edges=False, epochs=200, weight_decay=5e-3)
dc_tr, dc_val, dc_te, dc_pred, _ = eval_accs(mlp_deg_cl, data_deg_cl, use_edges=False)
print(f"MLP + degree + clustering -> train {dc_tr:.3f} | val {dc_val:.3f} | test {dc_te:.3f}")
print("Neighbor agreement:", neighbor_agreement(dc_pred, data.edge_index))
print("Narration: It senses whether neighbors are friends with each other (triangle density).")
print("Insight: Degree and clustering are correlated; without edge-based smoothing, redundancy + small data can hurt generalization.")

# 3C) + k-core
section("Step 3C: MLP + degree + clustering + k-core")
core = nx.core_number(G)
core = np.array([core[i] for i in range(N)], dtype=float)
core = core / (core.max() + 1e-12)
core_t = torch.tensor(core, dtype=torch.float32, device=device).unsqueeze(1)

data_deg_cl_core = data.clone()
data_deg_cl_core.x = torch.cat([data.x, deg_t, clust_t, core_t], dim=1)
data_deg_cl_core.x = zscore_columns(data_deg_cl_core.x)

mlp_deg_cl_core = MLP(in_dim=data_deg_cl_core.x.size(1), out_dim=dataset.num_classes, dropout=0.6).to(device)
train(mlp_deg_cl_core, data_deg_cl_core, use_edges=False, epochs=200, weight_decay=5e-3)
dcc_tr, dcc_val, dcc_te, dcc_pred, _ = eval_accs(mlp_deg_cl_core, data_deg_cl_core, use_edges=False)
print(f"MLP + degree + clustering + core -> train {dcc_tr:.3f} | val {dcc_val:.3f} | test {dcc_te:.3f}")
print("Neighbor agreement:", neighbor_agreement(dcc_pred, data.edge_index))
print("Narration: It knows if it lives downtown (deep in a core) or on the periphery.")
print("Insight: Still edge-blind; extra columns add complexity but not the smoothing prior that GCNs exploit.")

# 3D) + PageRank
section("Step 3D: MLP + degree + clustering + core + PageRank")
pr = nx.pagerank(G, alpha=0.85, tol=1e-6)
pr = np.array([pr[i] for i in range(N)], dtype=float)
pr_t = torch.tensor(pr, dtype=torch.float32, device=device).unsqueeze(1)

data_deg_cl_core_pr = data.clone()
data_deg_cl_core_pr.x = torch.cat([data.x, deg_t, clust_t, core_t, pr_t], dim=1)
data_deg_cl_core_pr.x = zscore_columns(data_deg_cl_core_pr.x)

mlp_deg_cl_core_pr = MLP(in_dim=data_deg_cl_core_pr.x.size(1), out_dim=dataset.num_classes, dropout=0.6).to(device)
train(mlp_deg_cl_core_pr, data_deg_cl_core_pr, use_edges=False, epochs=200, weight_decay=5e-3)
dccp_tr, dccp_val, dccp_te, dccp_pred, _ = eval_accs(mlp_deg_cl_core_pr, data_deg_cl_core_pr, use_edges=False)
print(f"MLP + degree + clustering + core + PR -> train {dccp_tr:.3f} | val {dccp_val:.3f} | test {dccp_te:.3f}")
print("Neighbor agreement:", neighbor_agreement(dccp_pred, data.edge_index))
print("Narration: It has global radar for influence, not just local buzz.")
print("Insight: PageRank injects a faint ‘multi-hop’ signal, so you often see a small bump—but without true message passing, gains stay modest.")

# 3E) + LPE(8)
section("Step 3E: MLP + degree + clustering + core + PR + LPE(8)")
lpe_note = ""
try:
    V8 = laplacian_eigenvectors(data.edge_index, N, k=8, normalized=True, device=device)
    data_feats8 = data.clone()
    data_feats8.x = torch.cat([data.x, deg_t, clust_t, core_t, pr_t, V8], dim=1)
    data_feats8.x = zscore_columns(data_feats8.x)

    mlp_feats8 = MLP(in_dim=data_feats8.x.size(1), out_dim=dataset.num_classes, dropout=0.6).to(device)
    train(mlp_feats8, data_feats8, use_edges=False, epochs=200, weight_decay=5e-3)
    f8_tr, f8_val, f8_te, f8_pred, _ = eval_accs(mlp_feats8, data_feats8, use_edges=False)
    lpe_note = " + 8D LPE"
    print(f"MLP + degree + clustering + core + PR + LPE(8) -> train {f8_tr:.3f} | val {f8_val:.3f} | test {f8_te:.3f}")
    print("Neighbor agreement:", neighbor_agreement(f8_pred, data.edge_index))
    print("Narration: We handed it smooth positional coordinates (graph harmonics).")
    print("Insight: LPE provides an explicit low-frequency basis, nudging the MLP toward the smoothness prior GCNs learn implicitly.")
except Exception as e:
    print("LPE(8) skipped:", e)

# ----------------------------
# Step 4: Consolidated recap (kept from your original, renamed for clarity)
# ----------------------------
section("Step 4: Consolidated recap — inject degree/clust/core/PR (+ optional LPE) then train MLP")
graph_feats = torch.tensor(np.stack([deg, clust, core, pr], axis=1),
                           dtype=torch.float32, device=device)
lpe_note_consol = ""
try:
    V8_consol = laplacian_eigenvectors(data.edge_index, N, k=8, normalized=True, device=device)
    graph_feats = torch.cat([graph_feats, V8_consol], dim=1)
    lpe_note_consol = " + 8D LPE"
except Exception as e:
    print("LPE(8) skipped in consolidated block:", e)

data_aug = data.clone()
data_aug.x = torch.cat([data.x, graph_feats], dim=1)

print("\nFeature dimensions:")
print(f"- Original feature dim: {data.x.size(1)}")
print(f"- Added graph columns: 4{' + 8 LPE' if lpe_note_consol else ''}")
print(f"- New total dim: {data_aug.x.size(1)}")

Xa = data_aug.x.detach().cpu().numpy()
yv = data_aug.y.cpu().numpy()
orig_cols = 6
added_names = ["deg","clust","core","pr"] + ([f"lpe{i+1}" for i in range(8)] if lpe_note_consol else [])
print(f"\nColumns shown: x0..x{orig_cols-1} + {', '.join(added_names[:6])} ...")
for i in np.linspace(0, N-1, 3, dtype=int):
    base = " ".join([f"{v: .3f}" for v in Xa[i, :orig_cols]])
    start = data.x.size(1); width = min(6, graph_feats.size(1))
    added_part = Xa[i, start:start+width]
    added = " ".join([f"{v: .3f}" for v in added_part])
    print(f"node {i:4d} | x0..x{orig_cols-1}=[{base}] | added=[{added}] | y={yv[i]}")

print("\nScale probe (mean L2 across columns):")
print("  original:", col_l2_mean(data.x))
print("  added   :", col_l2_mean(graph_feats))
print("  concat(before z):", col_l2_mean(data_aug.x))
data_aug.x = zscore_columns(data_aug.x)
print("  concat( after z):", col_l2_mean(data_aug.x))

mlp_aug = MLP(in_dim=data_aug.x.size(1), out_dim=dataset.num_classes, dropout=0.6).to(device)
train(mlp_aug, data_aug, use_edges=False, epochs=200, weight_decay=5e-3)
aug_train, aug_val, aug_test, aug_pred, _ = eval_accs(mlp_aug, data_aug, use_edges=False)
print(f"\nMLP + graph feats{lpe_note_consol} (z-scored) -> train {aug_train:.3f} | val {aug_val:.3f} | test {aug_test:.3f}")
print("Neighbor agreement (MLP + feats):", neighbor_agreement(aug_pred, data.edge_index))

# ----------------------------
# Step 5: Robustness — random 60/20/20 split (kept)
# ----------------------------
section("Step 5: Robustness check — random 60/20/20 split")
data_602020 = data.clone()
data_602020 = make_random_split(data_602020, 0.6, 0.2, 0.2)
t2, v2, te2 = int(data_602020.train_mask.sum()), int(data_602020.val_mask.sum()), int(data_602020.test_mask.sum())
print(f"New random split sizes -> train: {t2}, val: {v2}, test: {te2}")

mlp_raw2 = MLP(in_dim=dataset.num_node_features, out_dim=dataset.num_classes, dropout=0.6).to(device)
train(mlp_raw2, data_602020, use_edges=False, epochs=200, weight_decay=5e-3)
r2_train, r2_val, r2_test, _, _ = eval_accs(mlp_raw2, data_602020, use_edges=False)
print(f"MLP(raw) on 60/20/20 -> train {r2_train:.3f} | val {r2_val:.3f} | test {r2_test:.3f}")

data_aug_602020 = data_602020.clone()
data_aug_602020.x = torch.cat([data.x, graph_feats], dim=1)
data_aug_602020.x = zscore_columns(data_aug_602020.x)

mlp_aug2 = MLP(in_dim=data_aug_602020.x.size(1), out_dim=dataset.num_classes, dropout=0.6).to(device)
train(mlp_aug2, data_aug_602020, use_edges=False, epochs=200, weight_decay=5e-3)
a2_train, a2_val, a2_test, _, _ = eval_accs(mlp_aug2, data_aug_602020, use_edges=False)
print(f"MLP(+feats{lpe_note_consol}) on 60/20/20 -> train {a2_train:.3f} | val {a2_val:.3f} | test {a2_test:.3f}")
print("Narration: With more labels, even edge-blind MLPs catch up; supervision substitutes for relational bias.")
print("Insight: Handcrafted graph stats without smoothing still lag; the raw MLP narrows the gap as label rate grows.")

# ----------------------------
# Step 6: Epoch sensitivity (kept)
# ----------------------------
section("Step 6: Epoch sensitivity (same 60/20/20 split)")
mlp_aug_50 = MLP(in_dim=data_aug_602020.x.size(1), out_dim=dataset.num_classes, dropout=0.6).to(device)
train(mlp_aug_50, data_aug_602020, use_edges=False, epochs=50, weight_decay=5e-3)
b50_tr, b50_val, b50_te, _, _ = eval_accs(mlp_aug_50, data_aug_602020, use_edges=False)
print(f"MLP(+feats) 50 epochs -> train {b50_tr:.3f} | val {b50_val:.3f} | test {b50_te:.3f}")

mlp_aug_200 = MLP(in_dim=data_aug_602020.x.size(1), out_dim=dataset.num_classes, dropout=0.6).to(device)
train(mlp_aug_200, data_aug_602020, use_edges=False, epochs=200, weight_decay=5e-3)
b200_tr, b200_val, b200_te, _, _ = eval_accs(mlp_aug_200, data_aug_602020, use_edges=False)
print(f"MLP(+feats) 200 epochs -> train {b200_tr:.3f} | val {b200_val:.3f} | test {b200_te:.3f}")
print("Narration: Training longer doesn’t change the inductive bias—only how hard we fit it.")
print("Insight: Early stopping finds a sweet spot; beyond ~100–200 epochs we invite tiny gains and potential overfit.")

# ----------------------------
# Step 7: Stronger LPE → 32D (kept)
# ----------------------------
section("Step 7: Richer positional basis — LPE(32)")
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
    print("Narration: Richer LPE gives smoother coordinates—great when classes form broad regions on the graph.")
    print("Insight: Useful on homophilous graphs; compute cost ↑, diminishing returns after ~32 comps on Cora.")
except Exception as e:
    print("Stronger LPE step skipped:", e)

# ----------------------------
# Step 8: SGC pre-propagation (still MLP, edges offline only) (kept)
# ----------------------------
section("Step 8: MLP on SGC-preprop features (K=2) — edges used offline only")
def normalize_adj(edge_index, num_nodes):
    edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)
    row, col = edge_index
    degv = degree(col, num_nodes=num_nodes)
    deg_inv_sqrt = degv.pow(-0.5); deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
    norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
    return edge_index, norm

def prepropagate_features(X, edge_index, K=2):
    edge_index_norm, coeff = normalize_adj(edge_index, X.size(0))
    src, dst = edge_index_norm
    for _ in range(K):
        msg = coeff.unsqueeze(-1) * X[src]
        X = torch.zeros_like(X).index_add(0, dst, msg)
    return X

X_sgc = prepropagate_features(data_602020.x.clone(), data_602020.edge_index, K=2)
X_sgc = zscore_columns(X_sgc)
data_sgc = data_602020.clone(); data_sgc.x = X_sgc
mlp_sgc = MLP(in_dim=X_sgc.size(1), out_dim=dataset.num_classes, dropout=0.6).to(device)
train(mlp_sgc, data_sgc, use_edges=False, epochs=200, weight_decay=5e-3)
sgc_tr, sgc_val, sgc_te, _, _ = eval_accs(mlp_sgc, data_sgc, use_edges=False)
print(f"MLP on SGC-preprop (K=2) -> train {sgc_tr:.3f} | val {sgc_val:.3f} | test {sgc_te:.3f}")
print("Insight: Diffuse features first, then classify—this injects the smoothness prior explicitly.")
print("Narration: SGC is one-shot message passing—robust to training-time edge noise, but can’t adapt to new edges at test.")

# ----------------------------
# Step 9: Variant — concatenate X with SGC(X) on public split
# ----------------------------
section("Step 9: Variant — MLP on [X | SGC(X)] (public split)")
X_sgc_pub = prepropagate_features(data.x.clone(), data.edge_index, K=2)
X_sgc_pub = zscore_columns(X_sgc_pub)
data_concat_sgc = data.clone()
data_concat_sgc.x = zscore_columns(torch.cat([data.x, X_sgc_pub], dim=1))

mlp_concat_sgc = MLP(in_dim=data_concat_sgc.x.size(1), out_dim=dataset.num_classes, dropout=0.6).to(device)
train(mlp_concat_sgc, data_concat_sgc, use_edges=False, epochs=200, weight_decay=5e-3)
csgc_tr, csgc_val, csgc_te, csgc_pred, _ = eval_accs(mlp_concat_sgc, data_concat_sgc, use_edges=False)
print(f"MLP on [X | SGC(X)] (pub split) -> train {csgc_tr:.3f} | val {csgc_val:.3f} | test {csgc_te:.3f}")
print("Neighbor agreement:", neighbor_agreement(csgc_pred, data.edge_index))
print("Narration: Closest mimic to a GCN while staying edge-free at training time.")
print("Insight: Concatenation preserves sharp word signals while adding smoothed context—helpful when classes need both.")

# ----------------------------
# Step 10: Optional — +2-hop ego count (public split)
# ----------------------------
section("Step 10 (Optional): MLP + 2-hop ego count (public split)")
try:
    from scipy.sparse import csr_matrix
    A = nx.to_scipy_sparse_array(G, format='csr')
    A2 = A @ A
    A2 = A2.astype(np.int32)
    A2.setdiag(0)
    one_hop = np.asarray(A.sum(1)).ravel()
    two_hop_total = np.asarray((A2 > 0).sum(1)).ravel()
    two_hop_only = np.maximum(two_hop_total - one_hop, 0)
    two_hop_norm = two_hop_only / (two_hop_only.max() + 1e-12)
    twohop_t = torch.tensor(two_hop_norm, dtype=torch.float32, device=device).unsqueeze(1)

    data_twohop = data.clone()
    data_twohop.x = torch.cat([data.x, twohop_t], dim=1)
    data_twohop.x = zscore_columns(data_twohop.x)

    mlp_twohop = MLP(in_dim=data_twohop.x.size(1), out_dim=dataset.num_classes, dropout=0.6).to(device)
    train(mlp_twohop, data_twohop, use_edges=False, epochs=200, weight_decay=5e-3)
    th_tr, th_val, th_te, th_pred, _ = eval_accs(mlp_twohop, data_twohop, use_edges=False)
    print(f"MLP + 2-hop ego count -> train {th_tr:.3f} | val {th_val:.3f} | test {th_te:.3f}")
    print("Neighbor agreement:", neighbor_agreement(th_pred, data.edge_index))
    print("Insight: A hint of second-order structure without full message passing.")
except Exception as e:
    print("2-hop ego step skipped:", e)

# ----------------------------
# Step 11: GraphSAGE inductive demo (NEW)
# ----------------------------
section("Step 11: GraphSAGE inductive vs MLP([X | SGC(X)]) on held-out nodes")
# 1) Build a connected training subgraph (≈60% of nodes) via BFS on the giant component
all_nodes = np.array(list(giant.nodes()))
start = int(all_nodes[np.argmax([giant.degree(n) for n in all_nodes])])
# BFS order until target count
target = int(0.60 * N)
order = list(nx.bfs_tree(giant, start))
sub_nodes = torch.tensor(order[:target], dtype=torch.long, device=device)
sub_mask = torch.zeros(N, dtype=torch.bool, device=device)
sub_mask[sub_nodes] = True
heldout_mask = ~sub_mask
print(f"Inductive split: train-subgraph nodes = {int(sub_mask.sum())}, held-out nodes = {int(heldout_mask.sum())}")

# 2) Create train/val/test *inside* the subgraph; treat ALL held-out nodes as the inductive test set
train_mask_ind, val_mask_ind, test_mask_ind_inner = make_split_from_allowed_mask(sub_mask, 0.7, 0.15, 0.15, device=device)
inductive_test_mask = heldout_mask.clone()  # true inductive generalization

# Attach masks to a clone
ind_data = data.clone()
ind_data.train_mask = train_mask_ind
ind_data.val_mask   = val_mask_ind
ind_data.test_mask  = test_mask_ind_inner  # used only for in-subgraph reporting

# 3) Extract the training subgraph edges for *training only*
edge_index_sub, _ = tg_subgraph(sub_nodes, data.edge_index, relabel_nodes=False)
print(f"Edges used for SAGE training: {edge_index_sub.size(1)} (only inside the 60% subgraph)")

# 4) Train GraphSAGE on the subgraph only
sage = SAGE(in_dim=dataset.num_node_features, hid=64, out_dim=dataset.num_classes, dropout=0.5).to(device)
train_only_edge_index(sage, ind_data, edge_index_sub, lr=0.01, weight_decay=5e-4, epochs=200)
# Evaluate SAGE: (a) within-subgraph (transductive-style), (b) true inductive on held-out nodes
acc_fn_full, sage_pred_full, _ = eval_accs_on_edgeindex(sage, data, data.edge_index)  # full graph edges for inference
sage_in_sub = acc_fn_full(ind_data.test_mask)
sage_inductive = acc_fn_full(inductive_test_mask)
print(f"SAGE -> in-subgraph test {sage_in_sub:.3f} | held-out (inductive) {sage_inductive:.3f}")
print("Narration: SAGE never saw the held-out nodes during training; at inference it aggregates from their neighbors using the learned aggregator.")
print("Insight: If held-out accuracy stays close to in-subgraph, SAGE learned general neighbor-combine rules.")

with torch.no_grad():
    print("Neighbor agreement (SAGE, full graph):", neighbor_agreement(sage_pred_full, data.edge_index))

# 5) Inductive baseline: MLP on [X | SGC(X_train-subgraph)]
#    Compute SGC using ONLY the training subgraph edges (self-loops keep features for held-out nodes)
X_sgc_ind = prepropagate_features(data.x.clone(), edge_index_sub, K=2)
X_sgc_ind = zscore_columns(X_sgc_ind)
mlp_ind = MLP(in_dim=(data.x.size(1) + X_sgc_ind.size(1)), out_dim=dataset.num_classes, dropout=0.6).to(device)
ind_concat = data.clone(); ind_concat.x = zscore_columns(torch.cat([data.x, X_sgc_ind], dim=1))
ind_concat.train_mask = train_mask_ind; ind_concat.val_mask = val_mask_ind; ind_concat.test_mask = test_mask_ind_inner
train(mlp_ind, ind_concat, use_edges=False, epochs=200, weight_decay=5e-3)
# Evaluate MLP on held-out nodes (no edges ever used at train or test)
_, _, _, mlp_ind_pred, _ = eval_accs(mlp_ind, ind_concat, use_edges=False)
mlp_ind_in_sub = (mlp_ind_pred[ind_data.test_mask] == data.y[ind_data.test_mask]).float().mean().item()
mlp_ind_inductive = (mlp_ind_pred[inductive_test_mask] == data.y[inductive_test_mask]).float().mean().item()
print(f"MLP([X|SGC_train]) -> in-subgraph test {mlp_ind_in_sub:.3f} | held-out (inductive) {mlp_ind_inductive:.3f}")
print("Narration: The MLP never uses edges at training *or* test; SGC(X) was precomputed using only the train-subgraph structure.")
print("Takeaway: A gap SAGE≻MLP([X|SGC_train]) on held-out nodes demonstrates *true inductiveness* of neighbor aggregation, not just feature smoothing.")
# Extra probe: show SAGE needs test-time neighbors
acc_fn_trainonly, _, _ = eval_accs_on_edgeindex(sage, data, edge_index_sub)
sage_inductive_trainonly = acc_fn_trainonly(inductive_test_mask)
print(f"SAGE (inference uses *train-only* edges) held-out: {sage_inductive_trainonly:.3f}  ",
      "← shows dependence on test-time neighbors.")
print(f"Story: SAGE transfers a learned neighbor-aggregation rule to unseen nodes. Delta on held-out: +{(sage_inductive - mlp_ind_inductive):.3f} vs MLP+SGC_train.")

# ----------------------------
# Summary bar chart (optional)
# ----------------------------
if HAS_MPL:
    try:
        labels = [
            "Step1 GCN(pub)", "Step2 MLP(raw/pub)",
            "3A +deg", "3B +deg+clust", "3C +d+c+core", "3D +d+c+core+PR",
            "3E +d+c+core+PR+LPE8",
            "Step4 MLP(+feats)/pub",
            "Step5 MLP(raw/60-20-20)", "Step5 MLP(+feats/60-20-20)",
            "Step6 MLP(+feats/50ep)", "Step6 MLP(+feats/200ep)",
            "Step7 MLP(+feats+LPE32/60-20-20)",
            "Step8 MLP(SGC K=2/60-20-20)",
            "Step9 MLP([X|SGC(X)]/pub)",
            "Step10 MLP(+2hop/pub)",
            "Step11 SAGE(inductive held-out)"
        ]
        def _v(name):
            return globals()[name] if name in globals() else 0.0
        vals = [
            float(_v('gcn_test')), float(_v('mlp_test')),
            float(_v('d_te')), float(_v('dc_te')), float(_v('dcc_te')), float(_v('dccp_te')),
            float(_v('f8_te')) if 'f8_te' in globals() else 0.0,
            float(_v('aug_test')),
            float(_v('r2_test')), float(_v('a2_test')),
            float(_v('b50_te')), float(_v('b200_te')),
            float(_v('s_te')) if 's_te' in globals() else 0.0,
            float(_v('sgc_te')),
            float(_v('csgc_te')),
            float(_v('th_te')) if 'th_te' in globals() else 0.0,
            float(_v('sage_inductive')) if 'sage_inductive' in globals() else 0.0,
        ]
        plt.figure(figsize=(14,5))
        plt.bar(range(len(labels)), vals)
        plt.xticks(range(len(labels)), labels, rotation=38, ha='right', fontsize=8)
        plt.ylabel("Test Acc")
        plt.title("Evolving an MLP toward GCN behavior + Inductive GraphSAGE vs MLP([X|SGC_train])")
        plt.ylim(0, 1.0); plt.tight_layout(); plt.savefig('accuracy_summary.png', dpi=200)
        print('Saved summary bar chart to accuracy_summary.png')
    except Exception:
        pass
