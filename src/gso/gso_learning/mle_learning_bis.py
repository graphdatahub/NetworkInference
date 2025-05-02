import numpy as np
from scipy.optimize import minimize
from scipy.linalg import pinv, eigh

class GraphLaplacianEstimatorBis:
    def __init__(self, method='mle', alpha=1e-4, gamma=2.0, max_iter=1000, tol=1e-6):
        self.method = method
        self.alpha = alpha
        self.gamma = gamma
        self.max_iter = max_iter
        self.tol = tol

    def fit(self, X, A_mask=None):
        if self.method == 'mle':
            self.L = self._mle_estimate(X)
        elif self.method == 'ggl':
            self.L = self._ggl_estimate(X, A_mask)
        else:
            raise ValueError("Supported methods: 'mle', 'ggl'")
        return self

    def _mle_estimate(self, X):
        """Vanilla sparse inverse covariance estimation (without Laplacian constraints)"""
        n = X.shape[1]
        S = np.cov(X, rowvar=False)
        
        # Graphical Lasso objective
        def objective(L_flat):
            L = L_flat.reshape(n, n)
            try:
                logdet = np.linalg.slogdet(L)[1]
            except:
                return np.inf
            return np.trace(L @ S) - logdet + self.alpha * np.abs(L).sum()
        
        # Optimization without Laplacian constraints
        res = minimize(objective, pinv(S).flatten(), method='L-BFGS-B')

        if not res.success:
            raise RuntimeError(f"Optimization failed: {res.message}")
        
        return res.x.reshape(n, n)

    def _ggl_estimate(self, X, A_mask):
        """Generalized graph Laplacian estimation (with structural constraints)"""
        n = X.shape[1]
        S = np.cov(X, rowvar=False)
        K = S + self.alpha * (2*np.eye(n) - np.ones((n,n)))
        
        # Initialize with pseudo-inverse
        L = pinv(K)
        C = pinv(L)
        
        for cycle in range(self.max_iter):
            L_prev = L.copy()
            for u in range(n):
                # Get connectivity constraints for current node
                mask = np.concatenate([A_mask[:u,u], A_mask[u+1:,u]])
                L, C = self._update_row_column(L, C, K, u, mask)
                
            # Check convergence
            delta = np.linalg.norm(L - L_prev, 'fro') / np.linalg.norm(L_prev, 'fro')
            if delta < self.tol:
                break
                
        return L

    def _update_row_column(self, L, C, K, u, mask):
        """Block coordinate descent update for GGL"""
        n = L.shape[0]
        not_u = np.arange(n) != u
        
        # Extract submatrices
        L_uu = L[not_u, not_u]
        k_u = K[not_u, u]
        
        # Solve non-negative QP subproblem with connectivity constraints
        res = minimize(self._qp_objective, 
                      x0=L[not_u,u], 
                      args=(L_uu, k_u, mask),
                      method='L-BFGS-B',
                      bounds=[(0, None) if m else (0, 0) for m in mask])
        
        if not res.success:
            raise RuntimeError(f"Optimization failed: {res.message}")
        
        # Update matrix blocks
        L[not_u, u] = res.x
        L[u, not_u] = res.x
        
        # Sherman-Morrison update for inverse
        v = np.zeros(n)
        v[not_u] = res.x
        C = self._sherman_morrison_update(C, v, u)
        
        return L, C

    def _qp_objective(self, x, L_uu, k_u, mask):
        """Quadratic programming objective for row updates"""
        return 0.5 * x.T @ L_uu @ x + k_u.T @ x

    def _sherman_morrison_update(self, C, v, u):
        """Rank-1 update for matrix inverse"""
        c = C @ v
        return C - np.outer(c, c) / (1 + v @ c)

    @property
    def laplacian(self):
        return self.L
