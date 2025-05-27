import numpy as np
import numba
from numba import jit, prange
from numba.typed import List
import warnings

from .mle_utils import compute_sample_cov

@jit(nopython=True, fastmath=True, cache=True)
def soft_threshold(x, threshold):
    """Soft thresholding operator for LASSO"""
    if x > threshold:
        return x - threshold
    elif x < -threshold:
        return x + threshold
    else:
        return 0.0

@jit(nopython=True, fastmath=True, cache=True)
def lasso_coordinate_descent(A, b, lambda_reg, beta_init, max_iter=1000, tol=1e-6):
    """
    Solve min_beta 0.5 * ||A*beta - b||^2 + lambda_reg * ||beta||_1
    using coordinate descent
    """
    n, p = A.shape
    beta = beta_init.copy()
    
    # Precompute A^T A diagonal and A^T b
    AtA_diag = np.zeros(p)
    Atb = np.zeros(p)
    
    for j in range(p):
        AtA_diag[j] = np.dot(A[:, j], A[:, j])
        Atb[j] = np.dot(A[:, j], b)
    
    # Coordinate descent iterations
    for iteration in range(max_iter):
        beta_old = beta.copy()
        
        for j in range(p):
            if AtA_diag[j] < 1e-12:
                continue
                
            # Compute residual correlation
            r_j = Atb[j]
            for k in range(p):
                if k != j:
                    r_j -= AtA_diag[j] * beta[k] * np.dot(A[:, j], A[:, k]) / AtA_diag[j]
            
            # Apply soft thresholding
            beta[j] = soft_threshold(r_j / AtA_diag[j], lambda_reg / AtA_diag[j])
        
        # Check convergence
        if np.linalg.norm(beta - beta_old) < tol:
            break
    
    return beta

@jit(nopython=True, fastmath=True, cache=True)
def update_precision_block(W, S, j, lambda_reg, max_iter_lasso=1000, tol_lasso=1e-6):
    """
    Update j-th column/row of precision matrix using block coordinate descent
    """
    p = W.shape[0]
    
    if p == 1:
        return W
    
    # Create index arrays for partitioning
    indices_11 = np.zeros(p-1, dtype=numba.int64)
    idx = 0
    for i in range(p):
        if i != j:
            indices_11[idx] = i
            idx += 1
    
    # Extract submatrices
    W11 = np.zeros((p-1, p-1))
    s12 = np.zeros(p-1)
    
    for i in range(p-1):
        s12[i] = S[indices_11[i], j]
        for k in range(p-1):
            W11[i, k] = W[indices_11[i], indices_11[k]]
    
    w22 = W[j, j]
    
    # Solve LASSO problem: min_beta 0.5 * beta^T W11 beta + s12^T beta + lambda ||beta||_1
    # This is equivalent to: min_beta 0.5 * ||sqrt(W11) beta - sqrt(W11)^(-1) s12||^2 + lambda ||beta||_1
    
    # Use Cholesky decomposition for W11^(1/2)
    try:
        L = np.linalg.cholesky(W11 + 1e-12 * np.eye(p-1))
        W11_sqrt = L
        # Solve W11_sqrt @ x = s12 for x
        b = np.linalg.solve(L, s12)
        
        # Initialize beta
        beta_init = np.zeros(p-1)
        
        # Solve LASSO subproblem
        beta = lasso_coordinate_descent(W11_sqrt, b, lambda_reg, beta_init, max_iter_lasso, tol_lasso)
        
        # Update W
        w12 = W11 @ beta
        
        # Update the j-th column and row
        for i in range(p-1):
            W[indices_11[i], j] = w12[i]
            W[j, indices_11[i]] = w12[i]
            
    except:
        # Fallback: use regularized version
        W11_reg = W11 + 1e-6 * np.eye(p-1)
        try:
            L = np.linalg.cholesky(W11_reg)
            b = np.linalg.solve(L, s12)
            beta_init = np.zeros(p-1)
            beta = lasso_coordinate_descent(L, b, lambda_reg, beta_init, max_iter_lasso, tol_lasso)
            w12 = W11_reg @ beta
            
            for i in range(p-1):
                W[indices_11[i], j] = w12[i]
                W[j, indices_11[i]] = w12[i]
        except:
            pass  # Keep current values if decomposition fails
    
    return W

@jit(nopython=True, fastmath=True, cache=True, parallel=True)
def glasso_core(S, lambda_reg, max_iter=100, tol=1e-4, warm_start=None):
    """
    Core GLASSO algorithm using block coordinate descent
    """
    p = S.shape[0]
    
    # Initialize W (working covariance matrix)
    if warm_start is not None:
        W = warm_start.copy()
    else:
        W = S.copy()
        # Add regularization to diagonal for positive definiteness
        for i in range(p):
            W[i, i] += lambda_reg
    
    # Store previous iteration for convergence check
    W_prev = W.copy()
    
    # Block coordinate descent iterations
    for iteration in range(max_iter):
        # Cycle through all columns
        for j in range(p):
            W = update_precision_block(W, S, j, lambda_reg)
        
        # Check convergence
        diff_norm = 0.0
        for i in range(p):
            for k in range(p):
                diff_norm += (W[i, k] - W_prev[i, k]) ** 2
        diff_norm = np.sqrt(diff_norm)
        
        if diff_norm < tol:
            break
            
        W_prev = W.copy()
    
    return W

def glasso_fit(S, alpha=0.1, max_iter=100, tol=1e-4, warm_start=None):
    """
    Graphical LASSO algorithm for sparse precision matrix estimation
    
    Parameters:
    -----------
    S : numpy.ndarray
        Sample covariance matrix of shape (n_samples, n_samples)
    alpha : float
        Regularization parameter (λ)
    max_iter : int, default=100
        Maximum number of iterations
    tol : float, default=1e-4
        Convergence tolerance
    warm_start : numpy.ndarray, optional
        Initial covariance matrix for warm starting
        
    Returns:
    --------
    precision : numpy.ndarray
        Estimated precision matrix (Θ = Σ^(-1))
    covariance : numpy.ndarray
        Estimated covariance matrix (Σ)
    """
    W = glasso_core(S, alpha, max_iter=max_iter, tol=tol, warm_start=warm_start)

    try:
        precision = np.linalg.inv(W)
    except np.linalg.LinAlgError:
        W_reg = W + 1e-6 * np.eye(W.shape[0])
        precision = np.linalg.inv(W_reg)

    return precision, W
