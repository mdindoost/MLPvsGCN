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

def stationary_distribution_from_degrees(D):
    deg = np.diag(D).astype(float)
    pi = deg / deg.sum()
    return pi

def left_stationary(P):
    w, V = np.linalg.eig(P.T)
    idx = np.argmin(np.abs(w - 1))
    v = np.real(V[:, idx])
    v = np.maximum(v, 0)
    if v.sum() == 0:
        v = np.abs(v)
    return v / v.sum()

def main():
    A = np.array([[0,1,0],[1,0,1],[0,1,0]], dtype=float)
    I = np.eye(3)
    D = np.diag(A.sum(axis=1))

    R = np.linalg.inv(D) @ A
    R2 = R @ R
    R3 = R2 @ R

    R_lazy = 0.5*(I + R)

    A_tilde = A + I
    D_tilde = np.diag(A_tilde.sum(axis=1))
    D_tilde_inv = np.linalg.inv(D_tilde)
    D_tilde_inv_sqrt = np.diag(1.0/np.sqrt(np.diag(D_tilde)))

    P = D_tilde_inv @ A_tilde
    S = D_tilde_inv_sqrt @ A_tilde @ D_tilde_inv_sqrt

    D_tilde_sqrt = np.diag(np.sqrt(np.diag(D_tilde)))
    lhs = S
    rhs = D_tilde_sqrt @ P @ np.linalg.inv(D_tilde_sqrt)

    pi_deg = stationary_distribution_from_degrees(D)
    pi_tilde_deg = stationary_distribution_from_degrees(D_tilde)
    pi_from_R = left_stationary(R)
    pi_from_P = left_stationary(P)

    eig_R_vals, _ = np.linalg.eig(R)
    eig_Rlazy_vals, _ = np.linalg.eig(R_lazy)
    eig_P_vals, _ = np.linalg.eig(P)
    eig_S_vals, _ = np.linalg.eig(S)

    p0 = np.array([1.0, 0.0, 0.0])
    def power_distribution(Pmat, t):
        v = p0.copy()
        for _ in range(t):
            v = v @ Pmat
        return v

    p_R_1 = power_distribution(R, 1)
    p_R_2 = power_distribution(R, 2)
    p_R_3 = power_distribution(R, 3)
    p_Rlazy_5  = power_distribution(R_lazy, 5)
    p_Rlazy_20 = power_distribution(R_lazy, 20)
    p_P_5  = power_distribution(P, 5)
    p_P_20 = power_distribution(P, 20)

    pprint_matrix("Adjacency A", A, frac=True)
    pprint_matrix("Degree D", D, frac=True)
    pprint_matrix("Random-walk R = D^{-1}A", R, frac=True)
    pprint_matrix("R^2", R2, frac=True)
    pprint_matrix("R^3", R3, frac=True)

    print("\nOscillation under R (start at node 1):")
    print("p0 =", p0)
    print("p1 = p0 R^1 =", p_R_1)
    print("p2 = p0 R^2 =", p_R_2)
    print("p3 = p0 R^3 =", p_R_3)

    pprint_matrix("Lazy random walk R_lazy = 1/2 (I + R)", R_lazy, frac=True)
    print("\nConvergence under R_lazy (start at node 1):")
    print("p5  =", p_Rlazy_5)
    print("p20 =", p_Rlazy_20)

    pprint_matrix("Self-looped adjacency A_tilde = A + I", A_tilde, frac=True)
    pprint_matrix("Self-looped degree D_tilde = D + I", D_tilde, frac=True)
    pprint_matrix("Random-walk-with-self-loops P = D_tilde^{-1} A_tilde", P, frac=True)
    pprint_matrix("GCN symmetric normalization S = D_tilde^{-1/2} A_tilde D_tilde^{-1/2}", S, frac=True)

    print("\nSimilarity check: S ?= D_tilde^{1/2} P D_tilde^{-1/2}")
    with np.printoptions(precision=6, suppress=True):
        print("max|lhs - rhs| =", np.max(np.abs(lhs - rhs)))

    print("\nStationary distributions:")
    print("Degree-proportional for R (pi_deg)          =", pi_deg)
    print("From eigen (R):                             =", pi_from_R)
    print("Degree-proportional for P with self-loops   =", pi_tilde_deg)
    print("From eigen (P):                             =", pi_from_P)

    print("\nEigenvalues:")
    with np.printoptions(precision=6, suppress=True):
        print("eig(R)      =", np.sort(np.real(eig_R_vals)))
        print("eig(R_lazy) =", np.sort(np.real(eig_Rlazy_vals)))
        print("eig(P)      =", np.sort(np.real(eig_P_vals)))
        print("eig(S)      =", np.sort(np.real(eig_S_vals)))

if __name__ == "__main__":
    main()
