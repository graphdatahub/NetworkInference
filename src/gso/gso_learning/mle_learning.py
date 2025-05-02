# mypy: ignore-errors

import numpy as np
from scipy.optimize import minimize
from scipy.linalg import eigh, pinv
from scipy.sparse.csgraph import laplacian
from qpsolvers import solve_qp
from sklearn.covariance import graphical_lasso

class GraphLaplacianEstimatorBis:
    def __init__(self, method='sic', alpha=1e-4, gamma=2.0, max_iter=1000, tol=1e-8, pinv_init=True, solver='L-BFGS-B'):
        self.method = method
        self.alpha = alpha  # Regularization parameter
        self.gamma = gamma
        self.max_iter = max_iter
        self.tol = tol
        self.pinv_init = pinv_init
        self.solver = solver

    def fit(self, X, A_mask=None):
        if self.method == 'sic':
            self.L = self._sic_estimate(X, A_mask)
        elif self.method == 'ggl':
            if A_mask is None:
                A_mask = np.ones((X.shape[1], X.shape[1])) - np.eye(X.shape[1])
            self.L = self._ggl_estimate(X, A_mask)
        else:
            raise ValueError("Supported methods: 'mle', 'ggl'")
        return self

    def _sic_estimate(self, X, A_mask):
        """Core Graphical Lasso implementation. Needs full-rank."""
        S = np.cov(X, rowvar=False)
        n = S.shape[0]
        
        # Regularization for ill-conditioned cases
        S += 1e-3 * np.eye(n)  # Diagonal loading
        S = (S + S.T) / 2
        
        precision, _ = graphical_lasso(
            S, alpha=self.alpha, mode='cd',
            tol=self.tol, max_iter=self.max_iter,
            verbose=False, return_costs=False
        )

        assert precision is not None, "Graphical lasso failed"
        
        if A_mask is not None:
            A_mask = np.array(A_mask, dtype=bool)
            A_mask = np.maximum(A_mask, A_mask.T)

            precision *= A_mask
        
        precision = (precision + precision.T) / 2
        return -np.fill_diagonal(precision, 0)

    def _sic_estimate_v2(self, X):
        """Vanilla sparse inverse covariance estimation (without Laplacian constraints)"""
        n = X.shape[1]
        S = np.cov(X, rowvar=False)

        W = np.copy(S)

        for _ in range(self.max_iter):
            W_prev = np.copy(W)

            for i in range(n):
                not_i = np.arange(n) != i
                W_11 = W[not_i][:, not_i]
                s_12 = S[not_i, i]
                
                # Solve lasso problem
                def objective(beta):
                    return 0.5 * beta.T @ W_11 @ beta - beta.T @ s_12 + self.alpha * np.sum(np.abs(beta))

                res = minimize(objective, 
                            x0=W[not_i, i],
                            method='L-BFGS-B',
                            bounds=[(0, None) if i < j else (None, None) for _ in range(n-1)])
                
                W[not_i, i] = res.x
                W[i, not_i] = res.x
            
            if self._converged(W, W_prev):
                break
        else:
            print("WARNING: Did not reach the tolerance level")
        
        # Compute final precision matrix using block inversion formula
        Theta = np.zeros_like(W)
        for i in range(n):
            not_i = np.arange(n) != i
            W_11 = W[not_i][:, not_i]
            w_12 = W[not_i, i]
            theta_22 = 1 / (W[i,i] - w_12.T @ np.linalg.solve(W_11, w_12))
            theta_12 = -theta_22 * np.linalg.solve(W_11, w_12)
            Theta[i,i] = theta_22
            Theta[not_i,i] = theta_12
            Theta[i,not_i] = theta_12

        return Theta
    
        # if self.pinv_init:
        #     L = pinv(S)
        # else:
        #     L = np.diag(1 / np.diag(S))
        
        # # Graphical Lasso objective
        # def objective(L_flat):
        #     L = L_flat.reshape(n, n)
        #     try:
        #         logdet = np.linalg.slogdet(L)[1]
        #     except:
        #         return np.inf
        #     return np.trace(L @ S) - logdet + self.alpha * np.abs(L).sum()
        
        # # Optimization without Laplacian constraints
        # res = minimize(objective, L.flatten(), method='L-BFGS-B')

        # if not res.success:
        #     raise RuntimeError(f"Optimization failed: {res.message}")
        
        # return res.x.reshape(n, n)

    def _ggl_estimate(self, X, A_mask):
        """Generalized graph Laplacian estimation (with structural constraints)"""
        n = X.shape[1]
        S = np.cov(X, rowvar=False)

        # Ensures K is full-rank even if S is rank-deficient
        K = S + self.alpha * (2*np.eye(n) - np.ones((n,n)))
        
        # Validate and symmetrize mask
        A_mask = np.array(A_mask, dtype=bool)
        A_mask = np.maximum(A_mask, A_mask.T) 
        
        # Initialize with pseudo-inverse
        if self.pinv_init:
            L = pinv(K)
            C = pinv(L)
        else:
            L = np.diag(1 / np.diag(K))
            C = np.diag(np.diag(K))
        
        for _ in range(self.max_iter):
            L_prev = L.copy()
            for u in range(n):
                # Get connectivity constraints for current node
                mask = A_mask[:,u].copy()
                mask[u] = False
                L, C = self._update_row_column(L, C, K, u, mask)

            if self._converged(L, L_prev):
                break
        else:
            print("WARNING: Did not reach the tolerance level")

        return L

    def _update_row_column(self, L, C, K, u, mask):
        """Block coordinate descent update for GGL"""
        n = L.shape[0]
        not_u = np.arange(n) != u

        # Skip update if no variables to optimize
        if not np.any(mask):
            return L, C
        
        # Extract submatrices
        L_uu_full = L[np.ix_(not_u, not_u)]
        k_u_full = K[not_u, u].reshape(-1, 1)

        # Extract active variables
        active = np.where(mask[not_u])[0]
        L_uu = L_uu_full[np.ix_(active, active)]
        k_u = k_u_full[active]
        x0 = L[not_u, u][active]

        # Ensure positive definiteness
        # L_uu += 1e-8 * np.eye(L_uu.shape[0])

        # Solve non-negative QP subproblem with connectivity constraints
        if self.solver == 'L-BFGS-B':
            res = minimize(self._qp_objective, 
                        x0=x0,
                        args=(L_uu, k_u),
                        method=self.solver,
                        bounds=[(0, None)]*len(active))
            
            if not res.success:
                raise RuntimeError(f"Optimization failed: {res.message}")
            beta_opt = res.x
        
        elif self.solver == 'osqp':
            beta_opt = solve_qp(
                P=L_uu, 
                q=-k_u,
                lb=np.zeros(len(active)),
                solver=self.solver,
                # max_iter=10_000,
                # eps_abs=1e-6,
                # eps_rel=1e-6,
                verbose=False
            )
            if beta_opt is None:
                return L, C

        # Update matrix blocks
        beta_full = np.zeros(n-1)
        beta_full[active] = beta_opt
        L[not_u, u] = beta_full
        L[u, not_u] = beta_full
        
        # Sherman-Morrison update for inverse
        v = np.zeros(n)
        v[not_u] = beta_full
        C = self._sherman_morrison_update(C, v, u, K, not_u)
        
        return L, C

    def _qp_objective(self, x, L_uu, k_u):
        """Quadratic programming objective for row updates"""
        return 0.5 * x.T @ L_uu @ x + k_u.T @ x

    def _sherman_morrison_update(self, C, v, u, K, not_u):
        """Rank-1 update for matrix inverse"""
        c = C @ v
        denominator = 1 + v @ c
        C = C - np.outer(c, c) / denominator
        C_uu = C[np.ix_(not_u, not_u)]
        C_uu += np.outer(c[not_u], c[not_u]) / (denominator * K[u,u])
        C[np.ix_(not_u, not_u)] = C_uu
        return C

    def _converged(self, sol, sol_prev):
        delta = np.linalg.norm(sol - sol_prev, 'fro') / max(1, np.linalg.norm(sol_prev, 'fro'))
        return delta < self.tol
        
    def _post_process(self, L):
        """Final projection to Laplacian space"""
        L = (L + L.T) / 2
        return np.fill_diagonal(L, 0)

    @property
    def laplacian(self):
        # return self._post_process(self.L)
        return self.L




# ----------------------------
# Previous Versions
# ----------------------------


class GraphLaplacianEstimator:
    def __init__(self, method='mle', alpha=1e-4, gamma=2.0, max_iter=1000, tol=1e-6, ridge=1e-8):
        """
        Graph Laplacian estimator with multiple regularization options
        
        Parameters:
        method : str ('mle', 'frobenius', 'mcp')
        alpha : regularization strength (for frobenius/mcp)
        gamma : MCP concavity parameter (γ > 1)
        """
        self.method = method
        self.alpha = alpha
        self.gamma = gamma
        self.max_iter = max_iter
        self.tol = tol
        self.ridge = ridge

    def fit(self, X):
        """Estimate Laplacian from data matrix X (n_samples, n_nodes)"""
        self.n = X.shape[1]
        self.Sigma = self._regularized_covariance(X)

        L_init = self._nearest_laplacian(pinv(self.Sigma))
        
        bounds = [(0,0) if i==j else (None,0) for i in range(self.n) for j in range(self.n)]
        
        # Optimize using L-BFGS-B
        res = minimize(self._objective, L_init.flatten(), method='L-BFGS-B',
                       bounds=bounds, options={'maxiter': self.max_iter, 'ftol': self.tol})
        
        if not res.success:
            raise RuntimeError(f"Optimization failed: {res.message}")
        
        self.L = self._post_process(res.x)
        return self
    
    def _regularized_covariance(self, X):
        """Compute regularized sample covariance (Eq. 3.2)"""
        m = X.shape[0]
        Sigma = X.T @ X / m
        return Sigma + self.ridge * np.trace(Sigma) * np.eye(self.n) / self.n
    
    def _objective(self, L_flat):
        L = L_flat.reshape(self.n, self.n)
        L = self._enforce_laplacian_properties(L)
        
        if np.any(np.isnan(L)) or np.any(np.isinf(L)):
            return np.inf
    
        try:
            eigvals = eigh(L, eigvals_only=True, check_finite=False)
        except np.linalg.LinAlgError:
            return np.inf
        
        # Handle rank deficiency using pseudo-determinant
        pos_eig = np.clip(eigvals[1:], 1e-12, None)  # Exclude λ=0
        if len(pos_eig) < self.n-1:  # Disconnected graph penalty
            return np.inf
        
        log_pdet = np.sum(np.log(pos_eig)) if len(pos_eig) > 0 else 0
        trace_term = np.trace(L @ self.Sigma)
        
        # Base MLE objective
        obj = -log_pdet + trace_term
        
        # Add regularization based on method
        if self.method == 'frobenius':
            obj += self.alpha * np.linalg.norm(L, 'fro')**2
        elif self.method == 'mcp':
            obj += self._mcp_penalty(L)
            
        return obj
    
    def _mcp_penalty(self, L):
        """Minimax Concave Penalty (MCP) for off-diagonal elements"""
        penalty = 0
        for i in range(self.n):
            for j in range(i+1, self.n):
                x = abs(L[i,j])
                if x <= self.alpha * self.gamma:
                    penalty += self.alpha*x - x**2/(2*self.gamma)
                else:
                    penalty += 0.5 * self.alpha**2 * self.gamma
        return penalty
    
    def _enforce_laplacian_properties(self, L):
        """Ensure matrix satisfies Laplacian constraints"""
        L = 0.5 * (L + L.T)  # Symmetry
        np.fill_diagonal(L, 0)
        L -= np.diag(L.sum(axis=1))
        return np.clip(L, -1e12, 0)  # Numerical stability
    
    def _post_process(self, L_flat):
        """Final projection to Laplacian space"""
        L = L_flat.reshape(self.n, self.n)
        L = (L + L.T) / 2
        return np.fill_diagonal(L, 0)
        # return self._nearest_laplacian(L)
    
    def _nearest_laplacian(self, A):
        """Project matrix to Laplacian space (Sato & Suzuki 2024)"""
        W = np.maximum(-A - np.diag(np.diag(A)), 0)
        np.fill_diagonal(W, 0)
        return np.diag(W.sum(axis=1)) - W

    @property
    def laplacian(self):
        return self.L


# -------------------------------
# Other Implementations
# -------------------------------

def estimate_laplacian_mle(X, maxiter=1000, tol=1e-6, ridge=1e-6):
    """
    Estimates graph Laplacian from signals using constrained MLE approach.
    This method assumes that the covariance matrix if full-rank (num_signals >= num_nodes).

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
    reg = ridge * np.eye(n)  # Regularization for numerical stability

    def neg_log_likelihood(L_flat):
        L = L_flat.reshape(n, n)

        # Strict numerical checks
        if np.any(np.isnan(L)) or np.any(np.isinf(L)):
            return np.inf

        # Enforce symmetry
        L = (L + L.T) / 2

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
    bounds = [(0, 0) if i == j else (None, 0) for i in range(n) for j in range(n)]
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

def relaxed_mle_laplacian(X, alpha=1e-4, beta=1e-6, max_iter=1000, tol=1e-8):
    """
    Implements regularized likelihood:
    L = -log det†(L) + tr(LΣ) + α|L|_F^2 + γϕ(L)
    where ϕ(L) penalizes disconnected graphs.

    Args:
        X: (m, n) signal matrix (m samples, n nodes)
        alpha: Frobenius regularization strength
        beta: Covariance regularization strength
        
    Returns:
        L_est: Estimated Laplacian matrix
    """
    m, n = X.shape
    X_centered = X - np.mean(X, axis=0)
    
    # Regularized covariance estimation
    Sigma = X_centered.T @ X_centered / m
    Sigma_reg = Sigma + beta * np.trace(Sigma) * np.eye(n) / n
    
    # Spectral initialization
    try:
        L_init = nearest_laplacian(pinv(Sigma_reg))
    except np.linalg.LinAlgError:
        L_init = nearest_laplacian(-Sigma_reg)
    
    # Optimization bounds
    bounds = [(0, 0) if i == j else (None, 0) 
              for i in range(n) for j in range(n)]
    
    def objective(L_flat):
        L = L_flat.reshape(n, n)
        
        # Enforce symmetry and Laplacian constraints
        L = 0.5 * (L + L.T)
        np.fill_diagonal(L, 0)
        L -= np.diag(L.sum(axis=1))
        
        # Eigenvalue regularization
        try:
            eigvals = eigh(L, eigvals_only=True, check_finite=False)
            eigvals = np.clip(eigvals, 1e-10, None)  # λ_i ≥ ε
        except np.linalg.LinAlgError:
            return np.inf
        
        # Pseudo-determinant and penalty terms
        log_pdet = np.sum(np.log(eigvals[1:]))  # Exclude λ_1=0
        trace_term = np.trace(L @ Sigma_reg)
        frob_penalty = alpha * np.linalg.norm(L, 'fro')**2
        
        # Connected components penalty
        cc_penalty = connected_components_penalty(L)
        
        return -log_pdet + trace_term + frob_penalty + cc_penalty
    
    res = minimize(objective, L_init.flatten(), method='L-BFGS-B',
                   bounds=bounds, options={'maxiter': max_iter, 'ftol': tol})
    L_est = nearest_laplacian(res.x.reshape(n, n))
    
    return L_est

def nearest_laplacian(A):
    """Projects matrix to Laplacian space (Algorithm 4)"""
    W = np.maximum(-A - np.diag(np.diag(A)), 0)
    np.fill_diagonal(W, 0)
    return np.diag(W.sum(axis=1)) - W

def connected_components_penalty(L, penalty=1e6):
    """Penalizes disconnected graphs (Eq. 3.5)"""
    eigvals = eigh(L, eigvals_only=True, subset_by_index=[1,1])[0]
    return penalty * np.exp(-1e3 * eigvals) if eigvals < 1e-8 else 0