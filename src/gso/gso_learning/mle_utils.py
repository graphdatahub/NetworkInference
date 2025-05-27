from typing import Any

import numpy as np


def soft_threshold(x: np.ndarray, threshold: float) -> np.ndarray:
    """Soft thresholding operator for L1 regularization."""
    return np.sign(x) * np.maximum(np.abs(x) - threshold, 0.0)


def sherman_morrison_update(
    A_inv: np.ndarray, u: np.ndarray, v: np.ndarray
) -> np.ndarray:
    """Sherman-Morrison rank-1 update for matrix inverse."""
    denominator = 1.0 + v.T @ A_inv @ u
    if abs(denominator) < 1e-12:
        raise ValueError("Sherman-Morrison update failed: singular matrix")
    return A_inv - (A_inv @ np.outer(u, v) @ A_inv) / denominator


def nonnegative_qp_solver(
    A: np.ndarray, b: np.ndarray, tol: float = 1e-6, max_iter: int = 1000
) -> dict[str, Any]:
    """Coordinate descent solver for nonnegative quadratic programming."""
    n = A.shape[0]
    x = np.maximum(np.linalg.solve(A, b), 0.0)

    for _it in range(max_iter):
        x_old = x.copy()
        for i in range(n):
            residual = A[i, :] @ x - A[i, i] * x[i] - b[i]
            x[i] = max(0.0, (b[i] - residual) / A[i, i])
        if np.linalg.norm(x - x_old) < tol:
            break

    return {"xopt": x, "iterations": _it + 1, "converged": _it < max_iter - 1}


def laplacian_error_metrics(
    L_true: np.ndarray, L_est: np.ndarray, k: int | None = None
) -> dict[str, float]:
    """
    Compute Frobenius error, Spectral error, and Subspace error between two Laplacian matrices.

    Args:
        L_true (np.ndarray): True Laplacian matrix.
        L_est (np.ndarray): Estimated Laplacian matrix.
        k (int, optional): Number of eigenvectors to use for subspace error. If None, use all.

    Returns:
        dict: Dictionary with keys 'frobenius_error', 'spectral_error', 'subspace_error'.
    """
    # Frobenius norm of the difference
    frob_err = np.linalg.norm(L_true - L_est, ord="fro")

    # Spectral norm (largest singular value) of the difference
    spec_err = np.linalg.norm(L_true - L_est, ord=2)

    # Subspace error as Frobenius norm of difference of projection matrices
    eigvals_true, eigvecs_true = np.linalg.eigh(L_true)
    eigvals_est, eigvecs_est = np.linalg.eigh(L_est)
    if k is not None:
        U_true = eigvecs_true[:, :k]
        U_est = eigvecs_est[:, :k]
    else:
        U_true = eigvecs_true
        U_est = eigvecs_est
    P_true = U_true @ U_true.T
    P_est = U_est @ U_est.T
    subsp_err = np.linalg.norm(P_true - P_est, ord="fro")

    return {
        "frobenius_error": frob_err,
        "spectral_error": spec_err,
        "subspace_error": subsp_err,
    }
