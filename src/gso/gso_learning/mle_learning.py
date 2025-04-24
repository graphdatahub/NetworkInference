# mypy: ignore-errors

import numpy as np
from scipy.linalg import eigh
from scipy.optimize import minimize


def estimate_laplacian_mle(X, maxiter=1000, tol=1e-6):
    """
    Estimates graph Laplacian from signals using constrained MLE approach

    Args:
        X: Input signal matrix of shape (num_signals, num_nodes)
        maxiter: Maximum optimization iterations
        tol: Convergence tolerance

    Returns:
        L: Estimated Laplacian matrix of shape (num_nodes, num_nodes)
    """
    m, n = X.shape
    Sigma = X.T @ X / m  # Sample covariance

    # eigvals = eigh(Sigma, eigvals_only=True) # Precompute eigenvalues for pseudo-determinant calculation
    reg = 1e-6 * np.eye(n)  # Regularization for numerical stability

    def neg_log_likelihood(L_flat):
        L = L_flat.reshape(n, n)
        L = (L + L.T) / 2  # Enforce symmetry

        # Enforce Laplacian constraints
        np.fill_diagonal(L, 0)
        L -= np.diag(L.sum(axis=1))

        # Check positive semi-definiteness
        try:
            eigv = eigh(L, eigvals_only=True)
            if np.any(eigv < -1e-10):
                return np.inf
        except np.linalg.LinAlgError:
            return np.inf

        # Pseudo-determinant (product of positive eigenvalues)
        pos_eig = eigv[eigv > 1e-10]
        if len(pos_eig) == 0:
            return np.inf
        log_pdet = np.sum(np.log(pos_eig))

        # Trace term with regularization
        trace_term = np.trace(L @ (Sigma + reg))

        return -log_pdet + trace_term

    # Initialization: Nearest Laplacian approximation
    W = np.maximum(-Sigma - np.diag(np.diag(Sigma)), 0)
    np.fill_diagonal(W, 0)
    L_init = np.diag(W.sum(axis=1)) - W

    # Optimization setup
    bounds = [(0 if i == j else (None, 0)) for i in range(n) for j in range(n)]
    res = minimize(
        neg_log_likelihood,
        L_init.flatten(),
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": maxiter, "ftol": tol},
    )

    if not res.success:
        raise RuntimeError(f"Optimization failed: {res.message}")

    # Post-process to ensure Laplacian properties
    L = res.x.reshape(n, n)
    L = (L + L.T) / 2
    np.fill_diagonal(L, 0)
    L -= np.diag(L.sum(axis=1))

    return L
