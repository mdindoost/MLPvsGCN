# spectral_book_all.py
# Run examples from spectral graph theory "book-style" demos.
# Install: pip install numpy scipy matplotlib networkx

import argparse
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy.linalg import eigh
from scipy.spatial import Delaunay

# -------------------- Core linear-algebra helpers --------------------

def sorted_eig(L):
    """All eigenpairs for symmetric L (ascending)."""
    lam, V = eigh(L)
    return lam, V

def fix_signs(V, rule="maxabs_positive"):
    """Flip eigenvector signs for stable visuals; does not change eigenspaces."""
    V = V.copy()
    for i in range(V.shape[1]):
        j = np.argmax(np.abs(V[:, i]))
        if V[j, i] < 0:
            V[:, i] = -V[:, i]
    return V

def combinatorial_laplacian(A):
    d = A.sum(axis=1)
    D = np.diag(d)
    L = D - A
    return D, L

def normalized_laplacian(A):
    d = A.sum(axis=1)
    with np.errstate(divide='ignore'):
        inv_sqrt = 1.0 / np.sqrt(np.maximum(d, 1e-15))
    Dm12 = np.diag(inv_sqrt)
    Lsym = np.eye(A.shape[0]) - Dm12 @ A @ Dm12
    return Lsym

def print_eigs_and_first3(name, lam, V):
    print(f"\n=== {name} ===")
    print("Eigenvalues (ascending):")
    print(" ".join(f"{x:.10f}" for x in lam))
    np.set_printoptions(precision=6, suppress=True, linewidth=120)
    for k in range(min(3, V.shape[1])):
        print(f"\nEigenvector v{k+1} (λ{k+1} = {lam[k]:.10f}):")
        print(V[:, k])

def corr_and_best_scale_error(u, v):
    u0 = u - u.mean(); v0 = v - v.mean()
    denom = (np.linalg.norm(u0) * np.linalg.norm(v0))
    corr = float(u0 @ v0 / denom) if denom > 0 else np.nan
    vv = float(v @ v)
    alpha = float((v @ u) / vv) if vv > 0 else 0.0
    err = float(np.linalg.norm(u - alpha * v))
    return corr, alpha, err

# -------------------- Analytic path modes (for L = D - A) --------------------

def analytic_path_mode(k, n):
    j = np.arange(1, n+1)
    v = np.sin(k*np.pi*j/(n+1))
    nrm = np.linalg.norm(v)
    return v/nrm if nrm > 0 else v

# -------------------- Graph builders --------------------

def path_graph(n):
    G = nx.path_graph(n)                 # nodes 0..n-1
    A = nx.to_numpy_array(G, nodelist=range(n), dtype=float)
    return G, A

def grid_graph(r=3, c=4):
    G = nx.grid_2d_graph(r, c)
    nodes = [(i, j) for i in range(r) for j in range(c)]
    A = nx.to_numpy_array(G, nodelist=nodes, dtype=float)
    return G, A, nodes

# ---- Your synthetic Y generator (verbatim) ----
def synthetic_Y_points(n_points=500, noise=0.05, variant='classic', seed=None):
    rng = np.random.default_rng(seed)
    n_branch = n_points // 3
    stem_x = np.zeros(n_branch)
    stem_y = np.linspace(-1, 0, n_branch)
    left_x = -0.5 * np.linspace(0, 1, n_branch)
    left_y = np.linspace(0, 1, n_branch)
    right_x = 0.5 * np.linspace(0, 1, n_branch)
    right_y = np.linspace(0, 1, n_branch)
    X = np.vstack([
        np.stack([stem_x, stem_y], axis=1),
        np.stack([left_x, left_y], axis=1),
        np.stack([right_x, right_y], axis=1)
    ])
    if variant == 'tilted':
        angle = np.deg2rad(30)
        R = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        X = X @ R.T
    elif variant == 'curved':
        X[:, 1] = np.sign(X[:, 1]) * np.abs(X[:, 1])**1.2
    X += rng.normal(scale=noise, size=X.shape)
    return X

def largest_connected_component_adjacency(A):
    G = nx.from_numpy_array(A)
    if G.number_of_nodes() == 0:
        return A, np.arange(0)
    comps = sorted(nx.connected_components(G), key=len, reverse=True)
    idx = np.array(sorted(list(comps[0])))
    A_cc = A[np.ix_(idx, idx)]  # <- correct single underscore
    return A_cc, idx

def delaunay_graph_from_points(P, prune_factor=2.5, keep_lcc=True):
    tri = Delaunay(P)
    edges = set()
    for simplex in tri.simplices:
        i, j, k = simplex
        edges.add(tuple(sorted((i, j))))
        edges.add(tuple(sorted((j, k))))
        edges.add(tuple(sorted((i, k))))
    edges = np.array(sorted(list(edges)))
    E = P[edges[:,0]] - P[edges[:,1]]
    lengths = np.linalg.norm(E, axis=1)
    med = np.median(lengths)
    keep = lengths <= prune_factor * med
    edges = edges[keep]
    n = len(P)
    A = np.zeros((n, n), dtype=float)
    for u, v in edges:
        A[u, v] = 1.0
        A[v, u] = 1.0
    if keep_lcc:
        A_cc, idx = largest_connected_component_adjacency(A)
        P_cc = P[idx]
        G = nx.from_numpy_array(A_cc)
        return G, A_cc, P_cc
    else:
        G = nx.from_numpy_array(A)
        return G, A, P

# -------------------- Experiments --------------------

def demo_path(n=10, overlay_analytic=True):
    print(f"\n--- PATH GRAPH (n={n}) ---")
    _, A = path_graph(n)
    D, L = combinatorial_laplacian(A)
    Lsym = normalized_laplacian(A)
    lam_c, V_c = sorted_eig(L); V_c = fix_signs(V_c)
    lam_n, V_n = sorted_eig(Lsym); V_n = fix_signs(V_n)

    print_eigs_and_first3("Combinatorial Laplacian  L = D - A", lam_c, V_c)
    print_eigs_and_first3("Normalized Laplacian    L_sym = I - D^{-1/2} A D^{-1/2}", lam_n, V_n)

    x = np.arange(1, n+1)

    # Low-frequency (L)
    plt.figure(figsize=(7,4))
    if n >= 2: plt.plot(x, V_c[:,1], 'o-', label='v2')
    if n >= 3: plt.plot(x, V_c[:,2], 'o-', label='v3')
    if n >= 4: plt.plot(x, V_c[:,3], 'o-', label='v4')
    if overlay_analytic and n >= 4:
        plt.plot(x, analytic_path_mode(1, n), '--', label='v2 analytic')
        plt.plot(x, analytic_path_mode(2, n), '--', label='v3 analytic')
        plt.plot(x, analytic_path_mode(3, n), '--', label='v4 analytic')
    plt.xlabel("Vertex Number"); plt.ylabel("Value in Eigenvector")
    plt.title("Path graph – Low-frequency eigenvectors (L)")
    plt.legend(); plt.tight_layout()

    # Highest frequency (L)
    plt.figure(figsize=(7,4))
    plt.plot(x, V_c[:,-1], 'o-', label=f'v{n}')
    plt.xlabel("Vertex Number"); plt.ylabel("Value in Eigenvector")
    plt.title("Path graph – Highest-frequency eigenvector (L)")
    plt.legend(); plt.tight_layout()

    # Low-frequency (L_sym)
    plt.figure(figsize=(7,4))
    if n >= 2: plt.plot(x, V_n[:,1], 'o-', label='v2 (norm)')
    if n >= 3: plt.plot(x, V_n[:,2], 'o-', label='v3 (norm)')
    if n >= 4: plt.plot(x, V_n[:,3], 'o-', label='v4 (norm)')
    plt.xlabel("Vertex Number"); plt.ylabel("Value in Eigenvector")
    plt.title("Path graph – Low-frequency eigenvectors (Normalized L)")
    plt.legend(); plt.tight_layout()

    # Highest frequency (L_sym)
    plt.figure(figsize=(7,4))
    plt.plot(x, V_n[:,-1], 'o-', label=f'v{n} (norm)')
    plt.xlabel("Vertex Number"); plt.ylabel("Value in Eigenvector")
    plt.title("Path graph – Highest-frequency eigenvector (Normalized L)")
    plt.legend(); plt.tight_layout()

    # Correlations between L and L_sym eigenvectors (v2,v3)
    for k in [1,2]:
        if n > k:
            corr, alpha, err = corr_and_best_scale_error(V_c[:,k], V_n[:,k])
            print(f"Compare ψ{k+1} (L vs L_sym): corr={corr:.6f}, α={alpha:.6f}, ‖·‖₂-error={err:.6f}")

def demo_permutation_invariance(n=10, seed=42):
    print(f"\n--- PERMUTATION INVARIANCE (path n={n}) ---")
    _, A = path_graph(n)
    _, L = combinatorial_laplacian(A)
    lam, V = sorted_eig(L)

    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    P = np.eye(n)[perm]            # permutation matrix
    Lp = P @ L @ P.T
    lamp, Vp = sorted_eig(Lp)
    print("‖eigs(L) - eigs(P L P^T)‖_∞ =", np.max(np.abs(lam - lamp)))

def demo_grid_spectral_drawing(r=3, c=4, noise=0.08, affine=True, seed=0):
    print(f"\n--- GRID SPECTRAL DRAWING ({r}x{c}) ---")
    G, A, nodes = grid_graph(r, c)

    # Build a "messy" geometric layout (same topology)
    coords = np.array(nodes, dtype=float)
    rng = np.random.default_rng(seed)
    coords += rng.normal(scale=noise, size=coords.shape)  # jitter
    if affine:
        # random mild affine warp (rotation+shear)
        theta = rng.uniform(-0.6, 0.6)     # ~±34°
        shear = rng.uniform(-0.25, 0.25)
        R = np.array([[np.cos(theta), -np.sin(theta)],
                      [np.sin(theta),  np.cos(theta)]])
        S = np.array([[1.0, shear],
                      [0.0, 1.0]])
        Awarp = R @ S
        coords = coords @ Awarp.T

    # Plot messy geometric layout (original edges in noisy coords)
    plt.figure(figsize=(6,6))
    for (i,j) in G.edges():
        a = nodes.index(i); b = nodes.index(j)
        plt.plot([coords[a,0], coords[b,0]], [coords[a,1], coords[b,1]], color='lightgray', zorder=1)
    plt.scatter(coords[:,0], coords[:,1], s=25, c="tab:orange", zorder=2)
    plt.title("Messy geometric grid (same topology)")
    plt.axis('equal'); plt.tight_layout()

    # Spectral drawing from topology (combinatorial Laplacian)
    _, L = combinatorial_laplacian(A)
    lam, V = sorted_eig(L); V = fix_signs(V)
    x = V[:,1]; y = V[:,2]

    plt.figure(figsize=(6,6))
    for (i,j) in G.edges():
        a = nodes.index(i); b = nodes.index(j)
        plt.plot([x[a], x[b]], [y[a], y[b]], color='lightgray', zorder=1)
    plt.scatter(x, y, s=25, c="tab:blue", zorder=2)
    plt.title("Spectral drawing using (ψ2, ψ3) – Combinatorial L")
    plt.axis('equal'); plt.tight_layout()

    # Side-by-side comparison
    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    for (i,j) in G.edges():
        a = nodes.index(i); b = nodes.index(j)
        plt.plot([coords[a,0], coords[b,0]], [coords[a,1], coords[b,1]], color='gray', lw=1)
    plt.scatter(coords[:,0], coords[:,1], s=25, c="tab:orange"); plt.axis('equal')
    plt.title("Messy geometric grid")

    plt.subplot(1,2,2)
    for (i,j) in G.edges():
        a = nodes.index(i); b = nodes.index(j)
        plt.plot([x[a], x[b]], [y[a], y[b]], color='gray', lw=1)
    plt.scatter(x, y, s=25, c="tab:blue"); plt.axis('equal')
    plt.title("Spectral drawing (ψ2, ψ3) – Combinatorial L")
    plt.tight_layout()

def demo_Y_spectral(y_points=600, y_noise=0.02, y_variant="classic", prune_k=2.5, keep_lcc=True):
    print(f"\n--- Y SHAPE: points={y_points}, noise={y_noise}, variant={y_variant} ---")
    P = synthetic_Y_points(n_points=y_points, noise=y_noise, variant=y_variant, seed=0)

    # Show pure Y points (what audience sees first)
    plt.figure(figsize=(6,6))
    plt.scatter(P[:,0], P[:,1], s=20, color='navy')
    plt.title("Original Y (points only)")
    plt.axis('equal'); plt.tight_layout()

    # Build Delaunay graph (pruned)
    GY, AY, PY = delaunay_graph_from_points(P, prune_factor=prune_k, keep_lcc=keep_lcc)

    # Geometric graph
    plt.figure(figsize=(6,6))
    for (u,v) in GY.edges():
        plt.plot([PY[u,0], PY[v,0]], [PY[u,1], PY[v,1]], color='lightgray', zorder=1)
    plt.scatter(PY[:,0], PY[:,1], s=20, c="tab:purple", zorder=2)
    plt.title("Geometric Y-shaped graph (Delaunay, pruned)")
    plt.axis('equal'); plt.tight_layout()

    # Spectral drawings
    _, LY = combinatorial_laplacian(AY)
    lamY_c, VY_c = sorted_eig(LY); VY_c = fix_signs(VY_c)
    psi2_c, psi3_c = VY_c[:,1], VY_c[:,2]

    LYsym = normalized_laplacian(AY)
    lamY_n, VY_n = sorted_eig(LYsym); VY_n = fix_signs(VY_n)
    psi2_n, psi3_n = VY_n[:,1], VY_n[:,2]

    plt.figure(figsize=(6,6))
    for (u,v) in GY.edges():
        plt.plot([psi2_c[u], psi2_c[v]], [psi3_c[u], psi3_c[v]], color='lightgray', zorder=1)
    plt.scatter(psi2_c, psi3_c, s=20, c='tab:blue', zorder=2)
    plt.title("Spectral drawing (ψ2, ψ3) – Combinatorial L")
    plt.axis('equal'); plt.tight_layout()

    plt.figure(figsize=(6,6))
    for (u,v) in GY.edges():
        plt.plot([psi2_n[u], psi2_n[v]], [psi3_n[u], psi3_n[v]], color='lightgray', zorder=1)
    plt.scatter(psi2_n, psi3_n, s=20, c='tab:green', zorder=2)
    plt.title("Spectral drawing (ψ2, ψ3) – Normalized L")
    plt.axis('equal'); plt.tight_layout()

    # Comparison 3-up panel
    plt.figure(figsize=(15,5))
    plt.subplot(1,3,1)
    plt.scatter(P[:,0], P[:,1], s=20, color='navy'); plt.axis('equal')
    plt.title("Original Y (points only)")

    plt.subplot(1,3,2)
    for (u,v) in GY.edges():
        plt.plot([psi2_c[u], psi2_c[v]], [psi3_c[u], psi3_c[v]], color='gray', lw=1)
    plt.scatter(psi2_c, psi3_c, s=20, c='blue'); plt.axis('equal')
    plt.title("Spectral (ψ2, ψ3) – Combinatorial L")

    plt.subplot(1,3,3)
    for (u,v) in GY.edges():
        plt.plot([psi2_n[u], psi2_n[v]], [psi3_n[u], psi3_n[v]], color='gray', lw=1)
    plt.scatter(psi2_n, psi3_n, s=20, c='green'); plt.axis('equal')
    plt.title("Spectral (ψ2, ψ3) – Normalized L")
    plt.tight_layout()

# -------------------- CLI --------------------

def main():
    ap = argparse.ArgumentParser()
    # Path graph
    ap.add_argument("--n", type=int, default=10, help="path graph size")
    ap.add_argument("--overlay_analytic", action="store_true", help="overlay discrete sine modes on path plots")
    # Grid
    ap.add_argument("--grid_r", type=int, default=3, help="grid rows")
    ap.add_argument("--grid_c", type=int, default=4, help="grid cols")
    ap.add_argument("--grid_noise", type=float, default=0.08, help="Gaussian jitter for messy grid geometry")
    ap.add_argument("--grid_affine", action="store_true", help="apply a random affine warp to the messy grid")
    ap.add_argument("--grid_seed", type=int, default=0, help="random seed for messy grid")
    # Y
    ap.add_argument("--y_points", type=int, default=600, help="total points for Y")
    ap.add_argument("--y_noise", type=float, default=0.02, help="Gaussian noise for Y")
    ap.add_argument("--y_variant", type=str, default="classic", choices=["classic","tilted","curved"])
    ap.add_argument("--y_k", type=float, default=2.5, help="pruning factor for Delaunay edges")
    ap.add_argument("--y_keep_lcc", action="store_true", help="keep only largest CC (stable ψ2 if pruning fragments)")
    args = ap.parse_args()

    demo_path(n=args.n, overlay_analytic=args.overlay_analytic)
    demo_permutation_invariance(n=args.n, seed=42)
    demo_grid_spectral_drawing(r=args.grid_r, c=args.grid_c,
                               noise=args.grid_noise, affine=args.grid_affine, seed=args.grid_seed)
    demo_Y_spectral(y_points=args.y_points,
                    y_noise=args.y_noise,
                    y_variant=args.y_variant,
                    prune_k=args.y_k,
                    keep_lcc=args.y_keep_lcc)
    plt.show()

if __name__ == "__main__":
    main()
