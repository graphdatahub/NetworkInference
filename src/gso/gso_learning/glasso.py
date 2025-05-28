import numpy as np
from numba import njit


def _initialize_matrices(S, alpha):
    """Ensure S is symmetric positive definite, and initialize Theta, W."""
    n = S.shape[0]
    min_eig = np.min(np.linalg.eigvalsh(S))
    if min_eig < 1e-8:
        S = S + (abs(min_eig) + 1e-4) * np.eye(n)

    # DP-GLASSO diagonal initialization
    Theta_init = np.diag(1.0 / (np.diag(S) + alpha))
    W_init = np.diag(np.diag(S) + alpha)
    return S, Theta_init, W_init


@njit(fastmath=True, cache=True, nogil=True)
def _soft_threshold(x, lam):
    if x > lam:
        return x - lam
    elif x < -lam:
        return x + lam
    else:
        return 0.0


@njit(fastmath=True, cache=True, nogil=True)
def _lasso_coordinate_descent(A, b, lambda_reg, max_iter=100, tol=1e-6):
    """Solve min_beta 0.5 * ||A*beta - b||^2 + lambda_reg * ||beta||_1 using coordinate descent."""
    n, p = A.shape
    beta = np.zeros(p)
    AtA_diag = np.zeros(p)
    Atb = np.zeros(p)

    for j in range(p):
        col_j = np.ascontiguousarray(A[:, j])
        AtA_diag[j] = np.dot(col_j, col_j)
        Atb[j] = np.dot(col_j, b)

    for _ in range(max_iter):
        beta_old = beta.copy()
        for j in range(p):
            if AtA_diag[j] < 1e-12:
                continue

            col_j = np.ascontiguousarray(A[:, j])
            vec = np.ascontiguousarray(A @ beta - A[:, j] * beta[j])

            # Compute residual correlation
            r_j = Atb[j] - np.dot(col_j, vec)
            beta[j] = _soft_threshold(r_j / AtA_diag[j], lambda_reg / AtA_diag[j])
        if np.linalg.norm(beta - beta_old) < tol:
            break
    return beta


def _glasso_block_update(Theta, W, S, alpha, lasso_tol=1e-6, lasso_max_iter=100):
    n = Theta.shape[0]
    for i in range(n):
        not_i = np.array([j for j in range(n) if j != i])
        S11 = S[np.ix_(not_i, not_i)]
        s12 = S[not_i, i]
        W11 = W[np.ix_(not_i, not_i)]

        # Cholesky factorization
        try:
            L = np.linalg.cholesky(W11 + 1e-12 * np.eye(n - 1))
            A = L
            b = np.linalg.solve(L, s12)
        except np.linalg.LinAlgError:
            try:
                # Fallback: more regularization
                L = np.linalg.cholesky(W11 + 1e-6 * np.eye(n - 1))
                A = L
                b = np.linalg.solve(L, s12)
            except np.linalg.LinAlgError:
                continue  # Keep current values if decomposition fails

        beta = _lasso_coordinate_descent(
            A, b, alpha, max_iter=lasso_max_iter, tol=lasso_tol
        )
        theta12 = -beta
        theta22 = 1.0 / (S[i, i] - np.dot(s12, beta))
        Theta[i, i] = theta22
        Theta[not_i, i] = theta12
        Theta[i, not_i] = theta12

        # Sherman-Morrison-Woodbury update for W
        w12_new = -W11 @ beta * theta22
        W[i, i] = (1.0 / theta22) + np.dot(beta, w12_new)
        W[not_i, i] = w12_new
        W[i, not_i] = w12_new

    return Theta, W


def glasso_fit(S, alpha=0.1, tol=1e-4, max_iter=100, lasso_tol=1e-6, lasso_max_iter=100):
    """Graphical lasso routine"""
    S, Theta, W = _initialize_matrices(S, alpha)
    frob_norm = []
    converged = False

    for _iter in range(max_iter):
        Theta_old = Theta.copy()
        Theta, W = _glasso_block_update(
            Theta, W, S, alpha, lasso_tol=lasso_tol, lasso_max_iter=lasso_max_iter
        )

        denom = np.linalg.norm(Theta_old, "fro")
        rel_change = (
            np.inf if denom == 0 else np.linalg.norm(Theta - Theta_old, "fro") / denom
        )

        frob_norm.append(rel_change)
        if rel_change < tol:
            converged = True
            break
    try:
        Sigma = np.linalg.inv(Theta)
    except np.linalg.LinAlgError:
        Sigma = np.linalg.pinv(Theta)

    return {
        "O": Theta,
        "C": Sigma,
        "convergence": {
            "frob_norms": np.array(frob_norm),
            "converged": converged,
            "cycles": _iter,
        },
    }
