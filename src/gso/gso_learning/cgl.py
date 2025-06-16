import numpy as np


def _regularized_lstsq(A, b, reg_lambda=1e-5):
    """
    Solve (A^T A + reg_lambda * I) x = A^T b using lstsq for regularization.
    This is equivalent to Tikhonov (ridge) regularization.
    """
    n = A.shape[1]
    A_reg = np.vstack([A, np.sqrt(reg_lambda) * np.eye(n)])
    b_reg = np.concatenate([b, np.zeros(n)])
    x, residuals, rank, s = np.linalg.lstsq(A_reg, b_reg, rcond=None)
    return x


def _check_kkt_cond(grad, x):
    num = np.abs(grad) * (grad > 1e-12)
    score = num.copy()
    Pc = num < 1e-12
    if np.any(Pc):
        score[Pc] = num[Pc]
    if np.any(~Pc):
        score[~Pc] = num[~Pc] / (x[~Pc] + 1e-12)
    Pc = (num < 1e-12) | (score < 1)
    P = ~Pc
    kktP = np.max(np.abs(x[P])) if np.any(P) else 0
    kktPc = np.max(np.abs(grad[Pc])) if np.any(Pc) else 0
    N = x < -1e-12
    kktN = np.max(np.abs(x[N])) if np.any(N) else 0
    return max(kktP, kktPc, kktN)


def _nonnegative_qp_solver(A, b, inner_tol=1e-6, maxiter=200, reg_lambda=1e-5):
    p = A.shape[1]
    x = np.zeros(p)
    lambda_ = -b.copy()
    F = np.zeros(p, dtype=np.bool_)
    Lambda = ~F
    iter_ = 0
    check = _check_kkt_cond(lambda_, x)
    while check > inner_tol and iter_ < maxiter:
        fH1 = x < -1e-12
        H1 = fH1 & F
        fH2 = lambda_ < -1e-12
        H2 = fH2 & Lambda
        H1H2 = H1 | H2
        if np.sum(H1H2) == 0:
            break

        idx = np.where(H1H2)[0][-1]
        if H1[idx]:
            F[idx] = False
            Lambda[idx] = True
        else:
            F[idx] = True
            Lambda[idx] = False
        active = np.where(F)[0]

        if active.size > 0:
            # Use regularized least squares for stability
            x[active] = _regularized_lstsq(
                A[np.ix_(active, active)], b[active], reg_lambda=reg_lambda
            )

        x[~F] = 0
        lambda_[Lambda] = A[np.ix_(Lambda, F)] @ x[F] - b[Lambda]
        lambda_[F] = 0.9 * 1e-12
        iter_ += 1
        check = _check_kkt_cond(lambda_, x)
    return x


def _update_sherman_morrison_diag(O, C, shift, idx, tol=1e-10):
    O[idx, idx] += shift
    c_d = C[idx, idx]
    denom = 1 + shift * c_d
    if np.abs(denom) < tol:
        return O, C
    C -= (np.outer(C[:, idx], C[idx, :]) * shift) / denom
    return O, C


def _initialize_params(S, alpha, prob_tol, regularization_type):
    n = S.shape[0]
    e_v = np.ones(n) / np.sqrt(n)
    dc_var = e_v @ S @ e_v

    isshifting = np.abs(dc_var) < prob_tol
    if isshifting:
        # S = S + np.eye(n) / n
        S = S + (1 / n)

    if regularization_type == 1:
        H_alpha = alpha * (2 * np.eye(n) - np.ones((n, n)))
    elif regularization_type == 2:
        H_alpha = alpha * (np.eye(n) - np.ones((n, n)))
    else:
        raise ValueError("regularization_type must be 1 or 2")

    K = S + H_alpha

    return n, K, isshifting


def _initialize_matrices(K):
    O = np.diag(1.0 / np.diag(K))
    C = np.diag(np.diag(K))
    return O, C


def _block_update(O, C, K, A_mask, n, inner_tol, reg_lambda=1e-5):
    for u in range(n):
        minus_u = np.array([i for i in range(n) if i != u])
        k_u = K[minus_u, u]
        k_uu = K[u, u]
        c_u = C[minus_u, u]
        c_uu = C[u, u]
        Ou_i = C[np.ix_(minus_u, minus_u)] - np.outer(c_u, c_u) / c_uu
        beta = np.zeros(n - 1)
        ind_nz = A_mask[minus_u, u] == 1
        A_nnls = Ou_i[np.ix_(ind_nz, ind_nz)]
        b = k_u / k_uu + (1 / n) * Ou_i @ np.ones(n - 1)
        b_nnls = b[ind_nz]

        if A_nnls.shape[0] > 0:
            out_x = -_nonnegative_qp_solver(
                A_nnls, b_nnls, inner_tol, reg_lambda=reg_lambda
            )
            beta[ind_nz] = out_x

        o_u = beta + (1 / n)
        o_uu = (1 / k_uu) + o_u @ Ou_i @ o_u

        denom = o_uu - o_u @ Ou_i @ o_u
        if np.abs(denom) < 1e-10:
            denom = 1e-10

        cu = (Ou_i @ o_u) / denom
        cuu = 1.0 / denom

        O[u, u] = o_uu
        O[minus_u, u] = o_u
        O[u, minus_u] = o_u
        C[u, u] = cuu
        C[u, minus_u] = -cu
        C[minus_u, u] = -cu
        C[np.ix_(minus_u, minus_u)] = Ou_i + np.outer(cu, cu) / cuu
    return O, C


def cgl_fit(
    S,
    A_mask,
    alpha=0.1,
    prob_tol=1e-4,
    inner_tol=1e-6,
    max_cycle=20,
    regularization_type=1,
    reg_lambda=1e-5,
):
    n, K, isshifting = _initialize_params(S, alpha, prob_tol, regularization_type)
    O, C = _initialize_matrices(K)
    O_best = O.copy()
    C_best = C.copy()
    frob_norm = []
    converged = False
    cycle = 0

    while not converged and cycle < max_cycle:
        O_old = O.copy()
        O, C = _block_update(O, C, K, A_mask, n, inner_tol, reg_lambda=reg_lambda)

        _check_validity(O, name="precision matrix (post-BU)")
        _check_validity(C, name="covariance matrix (post-BU)")

        if cycle > 3:
            d_shifts = O @ np.ones(n) - 1
            large_diag_idx = np.where(np.abs(d_shifts) > 1e-12)[0]
            for idx in large_diag_idx:
                O, C = _update_sherman_morrison_diag(O, C, -d_shifts[idx], idx)

                _check_validity(O, name="precision matrix (post-SM)")
                _check_validity(C, name="covariance matrix (post-SM)")

        O_best = O.copy()
        C_best = C.copy()
        cycle += 1
        frob_norm.append(np.linalg.norm(O_old - O, "fro") / np.linalg.norm(O_old, "fro"))

        if cycle > 5 and frob_norm[-1] < prob_tol:
            converged = True
            O_best = O.copy()
            C_best = C.copy()

    if isshifting:
        # O = O_best - np.eye(n) / n
        # C = C_best - np.eye(n) / n
        O = O_best - (1 / n)
        C = C_best - (1 / n)
    else:
        O = O_best
        C = C_best

    return {
        "O": O,
        "C": C,
        "convergence": {
            "frob_norms": np.array(frob_norm),
            "converged": converged,
            "cycles": cycle,
        },
    }


def _check_validity(matrix, name="matrix", tol=1e-8):
    """Raise a RuntimeError if matrix contains NaN or Inf values."""
    has_nan = np.isnan(matrix).any()
    has_inf = np.isinf(matrix).any()
    if has_nan or has_inf:
        msg = f"Invalid {name}: contains"
        if has_nan:
            msg += " NaN"
        if has_nan and has_inf:
            msg += " and"
        if has_inf:
            msg += " Inf"
        msg += " values"
        raise RuntimeError(msg)
