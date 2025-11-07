#!/usr/bin/env python3
import numpy as np
from fractions import Fraction

def to_frac(M):
    rows = []
    for row in M:
        cells = []
        for x in row:
            f = Fraction(x).limit_denominator()
            if abs(float(f) - float(x)) < 1e-12:
                cells.append(str(f))
            else:
                cells.append(f"{x:.6g}")
        rows.append("[ " + ", ".join(cells) + " ]")
    return "[\n  " + "\n  ".join(rows) + "\n]"

def pprint_matrix(name, M, frac=False):
    print(f"\n{name} =")
    if frac:
        print(to_frac(M))
    else:
        with np.printoptions(precision=6, suppress=True):
            print(M)

def chain_graph(n):
    A = np.zeros((n,n), dtype=float)
    for i in range(n-1):
        A[i, i+1] = 1.0
        A[i+1, i] = 1.0
    return A

def star_graph(n):
    # n nodes: node 0 is center, 1..n-1 are leaves
    A = np.zeros((n,n), dtype=float)
    for i in range(1, n):
        A[0, i] = 1.0
        A[i, 0] = 1.0
    return A

def gcn_norm(A):
    I = np.eye(A.shape[0])
    A_tilde = A + I
    D_tilde = np.diag(A_tilde.sum(axis=1))
    D_tilde_inv_sqrt = np.diag(1.0 / np.sqrt(np.diag(D_tilde)))
    S = D_tilde_inv_sqrt @ A_tilde @ D_tilde_inv_sqrt
    return A_tilde, D_tilde, S

def spectral_info(S):
    w, V = np.linalg.eigh((S+S.T)/2.0)  # symmetric ensure numerics
    idx = np.argsort(w)[::-1]
    w = w[idx]
    V = V[:, idx]
    return w, V

def powers(S, ks):
    out = []
    M = np.eye(S.shape[0])
    last_k = 0
    for k in ks:
        # accumulate multiplication
        for _ in range(k - last_k):
            M = M @ S
        out.append((k, M.copy()))
        last_k = k
    return out

def variation_energy(Lsym, x):
    return float(x.T @ Lsym @ x)

def print_oversmoothing_demo(name, A, show_frac=True):
    print(f"\n========== {name}: Oversmoothing Demo ==========")
    A_tilde, D_tilde, S = gcn_norm(A)
    n = A.shape[0]
    I = np.eye(n)
    Lsym = I - S

    pprint_matrix("Adjacency A", A, frac=show_frac)
    pprint_matrix("Self-looped adjacency A_tilde = A + I", A_tilde, frac=show_frac)
    pprint_matrix("Self-looped degree D_tilde", D_tilde, frac=show_frac)
    pprint_matrix("GCN normalization S = D_tilde^{-1/2} A_tilde D_tilde^{-1/2}", S, frac=show_frac)

    w, V = spectral_info(S)
    print("\nEigenvalues(S) (descending):")
    with np.printoptions(precision=6, suppress=True):
        print(w)

    gap = w[0] - w[1] if len(w) > 1 else np.nan
    print("Spectral gap (lambda1 - lambda2):", gap)

    # Powers of S: show convergence of rows
    ks = [1, 2, 3, 5, 10]
    for k, Sk in powers(S, ks):
        pprint_matrix(f"S^{k}", Sk, frac=False)

    # Show an initial feature vector and how it's smoothed
    x = np.zeros((n,1))
    x[0,0] = 1.0     # spike at node 0
    x[-1,0] = -1.0   # negative spike at last node
    pprint_matrix("Initial feature x", x, frac=False)
    e0 = variation_energy(Lsym, x)
    print("Initial variation energy x^T L_sym x =", e0)

    for k in [1,2,3,5,10]:
        xk = (np.linalg.matrix_power(S, k) @ x)
        pprint_matrix(f"x_k = S^{k} x", xk, frac=False)
        ek = variation_energy(Lsym, xk)
        print(f"Variation energy after {k} steps =", ek)

def print_oversquashing_demo(A, leaf_pairs, K_list, show_frac=True):
    print("\n========== Star Graph: Oversquashing Demo ==========")
    A_tilde, D_tilde, S = gcn_norm(A)
    pprint_matrix("GCN normalization S (star)", S, frac=show_frac)

    # Influence (Jacobian) from inputs at node j to outputs at node i after K layers (linearized) is (S^K)_{ij}
    for K in K_list:
        SK = np.linalg.matrix_power(S, K)
        print(f"\nLayer depth K = {K}: selected pairwise influences (entries of S^{K})")
        for (i, j) in leaf_pairs:
            print(f"influence i<-j (node {i} from node {j}): {SK[i,j]:.6g}")
        # Sum of influences at center from all leaves (capacity crowding at bottleneck)
        infl_center_from_leaves = SK[0, 1:].sum()
        print(f"Total influence into center from all leaves at depth {K}: {infl_center_from_leaves:.6g} (crowding)")

def main():
    n = 5
    # Chain (oversmoothing): nodes 0-1-2-3-4
    A_chain = chain_graph(n)
    print_oversmoothing_demo("Chain (P5): oversmoothing", A_chain, show_frac=True)

    # Star (oversquashing): node 0 is center, 1..4 leaves
    A_star = star_graph(n)
    # Choose leaf pairs to inspect cross-leaf influence through bottleneck
    leaf_pairs = [(i,j) for i in range(1,n) for j in range(1,n) if i!=j]
    K_list = [1,2,3,5,10]
    print_oversquashing_demo(A_star, leaf_pairs[:6], K_list, show_frac=True)

if __name__ == "__main__":
    main()
