# ruff: noqa: RUF002, E741

"""
This script includes the code for:

(1) Graphical lasso algorithm (MLE without Laplacian constraints), solving

    min_{Σ} trace(S Σ^-1) - log det(Σ^-1) + alpha |Σ^-1|_1, subject to PSD: Σ = Σ^T, Σ ⪰ 0

(2) MLE with Laplacian constraints, solving

    min_{Θ in L}; trace(S Θ) - log det(Θ) + alpha |Θ|_1

    where L is the set of combinatorial graph Laplacians.

(3) Smooth graph Laplacian learning algorithm, solving

    min_Θ; trace(S Θ)+ α_1 |Θ|_F,off^2 - α_2 sum_i log(Θ_ii), subject to Θ ⪰ 0, Θ1 = 0, Θ_i≠j ≤ 0

Constraining the sum of each row/column to be zero in (2) leads to a singularity for computing the determinant.
This is why we use a shifted precision matrix Θ + ν I for (2).
However (3) handles the singularity differently, using the log of diagonal elements, and the Frobenius norm of off-diagonal elements.

Note: The sparisty terms are only applied when no mask is given.
All algorithms allow for connectivity constraints in the form of a mask, or a sparsity-inducing regularization.
We do not use QP with L1 regularization but soft thresholding, since L1 penalty decomposes into separable problems,
where zeros emerge from the optimization rather than enforcing exact zeros.

TODO:
- Try out and implement Newton-like or proximal Newton approaches
- Move away from l1 regularization for sparsity
"""

# TODO: Implement algo for Pb 3

from typing import Any

import numpy as np

from ..core import (
    Matrix,
)
from .mle_utils import nonnegative_qp_solver, sherman_morrison_update, soft_threshold


def block_update(
    matrix_type: str,
    K: np.ndarray,
    C: np.ndarray,
    A_mask: Matrix | None,
    u: int,
    params: dict[str, float],
) -> tuple[np.ndarray, float]:
    """Single block update with optional sparsity/mask constraints"""
    n = K.shape[0]
    minus_u = list(range(u)) + list(range(u + 1, n))

    # Submatrix extraction
    k_u = K[minus_u, u]
    k_uu = K[u, u]
    c_u = C[minus_u, u]
    c_uu = C[u, u]

    # Shared matrix computation
    Ou_i = C[np.ix_(minus_u, minus_u)] - np.outer(c_u, c_u) / c_uu

    if matrix_type == "cgl":
        residual = k_u / k_uu + Ou_i @ np.ones(n - 1) / n

        if A_mask is not None:
            # Mask-constrained CGL with optional L1
            ind_nz = A_mask[minus_u, u] == 1
            A_nnls = Ou_i[np.ix_(ind_nz, ind_nz)] if np.any(ind_nz) else np.zeros((0, 0))
            b_nnls = residual[ind_nz]

            result = (
                nonnegative_qp_solver(A_nnls, b_nnls, params["inner_tol"])
                if A_nnls.size
                else {"xopt": np.array([])}
            )
            beta = np.zeros(n - 1)
            if np.any(ind_nz):
                beta[ind_nz] = -result["xopt"]
            o_u = beta + 1 / n
        else:
            # Sparsity-regularized CGL (no mask)
            o_u = soft_threshold(residual, params["lambda"]) + 1 / n  # preserves sign

        o_uu = 1 / k_uu + o_u.T @ Ou_i @ o_u

    elif matrix_type == "glasso":
        residual = k_u - Ou_i @ C[minus_u, u]

        # Apply mask constraints if provided
        if A_mask is not None:
            mask = A_mask[minus_u, u]
            residual *= mask  # Zero out prohibited edges
            beta = residual  # No L1 penalty when mask exists
        else:
            # Apply L1 penalty else
            beta = soft_threshold(residual, params["lambda"])

        o_u = beta
        o_uu = 1 / k_uu

    else:
        raise ValueError("Invalid matrix_type. Use 'cgl' or 'glasso'")

    return o_u, o_uu


def update_covariance(
    matrix_type: str,
    O: np.ndarray,
    C: np.ndarray,
    o_u: np.ndarray,
    o_uu: float,
    u: int,
    params: dict[str, float],
) -> tuple[np.ndarray, np.ndarray]:
    n = O.shape[0]
    minus_u = list(range(u)) + list(range(u + 1, n))

    # Update precision matrix
    O[u, u] = o_uu
    O[minus_u, u] = O[u, minus_u] = o_u

    # Sherman-Morrison update
    Ou_i = (
        C[np.ix_(minus_u, minus_u)] - np.outer(C[minus_u, u], C[minus_u, u]) / C[u, u]
    )  # Schur complement
    cu = (Ou_i @ o_u) / (o_uu - o_u.T @ Ou_i @ o_u)
    cuu = 1 / (o_uu - o_u.T @ Ou_i @ o_u)

    # Additional sparsity enforcement for CGL
    if matrix_type == "cgl" and params["lambda"] > 0:
        cu = soft_threshold(cu, params["lambda"])

    # Update covariance matrix
    C[u, u] = cuu
    C[minus_u, u] = C[u, minus_u] = -cu
    C[np.ix_(minus_u, minus_u)] = Ou_i + np.outer(cu, cu) / cuu

    return O, C


def block_descent(
    S: Matrix,
    matrix_type: str,
    A_mask: Matrix | None = None,
    params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Main function for GLASSO and graph learning algorithms.
    :input: sample covariance matrix, algorithm, optional adjacencies"""

    default_params = {
        "alpha": 0.1,  # CGL regularization
        "lambda": 0.0,  # sparsity lambda
        "prob_tol": 1e-8,
        "inner_tol": 1e-6,
        "max_cycle": 100,
        "shift_delay": 3,
        "shift_tol": 1e-8,
        "min_eig": 1e-8,
        "regularization_type": 1,
    }
    params = {**default_params, **(params or {})}

    n = S.shape[0]
    if matrix_type == "cgl" and params["lambda"] > 0 and A_mask is not None:
        raise ValueError("CGL cannot have both mask and L1 regularization")

    # Matrix initialization
    if matrix_type == "cgl":
        e_v = np.ones(n) / np.sqrt(n)
        dc_var = e_v.T @ S @ e_v
        if abs(dc_var) < params["prob_tol"]:
            S = S + 1 / n

        if params["regularization_type"] == 1:
            H_alpha = params["alpha"] * (2 * np.eye(n) - np.ones((n, n)))
        elif params["regularization_type"] == 2:
            H_alpha = params["alpha"] * (np.eye(n) - np.ones((n, n)))
        K = S + H_alpha
        O = np.diag(1 / np.diag(K))
        C = np.diag(np.diag(K))

    elif matrix_type == "glasso":
        K = S.copy()
        if A_mask is None:
            K += params["lambda"] * np.eye(n)
        O = np.linalg.inv(K)
        C = K.copy()

    frob_norms = []
    converged = False
    cycle = 0

    try:
        while not converged and cycle < params["max_cycle"]:
            O_old = O.copy()

            # Block iteration with dual constraints
            for u in range(n):
                o_u, o_uu = block_update(matrix_type, K, C, A_mask, u, params)
                O, C = update_covariance(matrix_type, O, C, o_u, o_uu, u, params)

            # Enhanced diagonal correction
            if matrix_type == "cgl" and cycle > params["shift_delay"]:
                d_shifts = O @ np.ones(n) - 1
                large_diag_idx = np.where(np.abs(d_shifts) > params["shift_tol"])[0]
                for idx in large_diag_idx:
                    delta = -d_shifts[idx]
                    O[idx, idx] += delta
                    u_vec = np.zeros(n)
                    u_vec[idx] = 1.0
                    C = sherman_morrison_update(
                        C, u_vec * delta, u_vec
                    )  # maintains inverse positivity

            # Convergence check
            frob_norm = np.linalg.norm(O - O_old, "fro") / np.linalg.norm(O_old, "fro")
            frob_norms.append(frob_norm)
            converged = (cycle > 5) and (frob_norm < params["prob_tol"])
            cycle += 1

    except Exception as e:
        print(f"Optimization stopped due to error: {e}")
        frob_norms = [np.inf]
        converged = False

    # Final adjustments
    if matrix_type == "cgl":
        O -= 1 / n
        C -= 1 / n

    return {
        "O": O,
        "C": C,
        "convergence": {
            "frob_norms": frob_norms,
            "converged": converged,
            "cycles": cycle,
        },
    }
