import numpy as np
from numba import njit, prange
from qpsolvers import solve_qp

@njit(fastmath=True, cache=True)
def update_sherman_morrison_diag(O, C, delta, idx):
    v = np.zeros(O.shape[0])
    v[idx] = 1.0
    Cv = C @ v
    denom = 1.0 + delta * Cv[idx]
    C_new = C - np.outer(Cv, Cv) * delta / denom
    O_new = O.copy()
    O_new[idx, idx] += delta
    return O_new, C_new

@njit(fastmath=True, cache=True)
def process_node(u, n, A_mask, K, O, C):
    not_u = np.delete(np.arange(n), u)
    mask = A_mask[not_u, u]
    
    if not np.any(mask):
        # Return empty arrays with proper types and dimensions
        empty_Q = np.empty((0,0), dtype=np.float64)
        empty_c = np.empty(0, dtype=np.float64)
        empty_mask = np.empty(0, dtype=np.bool_)
        return empty_Q, empty_c, not_u, empty_mask, 0.0, np.empty((0,0), dtype=np.float64)
    
    K_uu = K[u,u]
    K_u = K[not_u,u]
    C_uu = C[u,u]
    C_u = C[not_u,u]
    O_uu_inv = C[not_u, :][:, not_u] - np.outer(C_u, C_u)/C_uu
    
    # Solve QP subproblem setup
    Q = O_uu_inv[mask, :][:, mask].copy()
    c = (-K_u[mask]/K_uu - O_uu_inv[mask, :] @ np.ones(len(not_u))/n).copy()
    
    return Q, c, not_u, mask, K_uu, O_uu_inv

def solve_node_qp(Q, c):
    return solve_qp(Q, c, solver='osqp', lb=np.zeros_like(c))

@njit(fastmath=True, parallel=True, cache=True)
def update_matrices(O, C, u, not_u, mask, x, K_uu, O_uu_inv, n):
    beta = np.zeros(len(not_u))
    beta[mask] = x
    o_u = beta + 1/n
    o_uu = 1/K_uu + o_u.T @ O_uu_inv @ o_u
    
    # Update O matrix
    O[u,u] = o_uu
    O[not_u,u] = o_u
    O[u,not_u] = o_u
    
    # Update C matrix using Sherman-Morrison
    Cu = (O_uu_inv @ o_u) / (o_uu - o_u.T @ O_uu_inv @ o_u)
    Cuu = 1 / (o_uu - o_u.T @ O_uu_inv @ o_u)
    
    C[u,u] = Cuu
    C[not_u,u] = -Cu
    C[u,not_u] = -Cu
    C[not_u, not_u] = O_uu_inv + np.outer(Cu, Cu)/Cuu
    
    return O, C

def cgl_fit(S, A_mask=None, alpha=0.1, max_iter=100, tol=1e-4):
    n = S.shape[0]
    A_mask = np.ones((n,n), dtype=np.bool_) if A_mask is None else A_mask
    
    # Initialize matrices
    O = np.diag(1/(np.diag(S) + alpha))
    C = np.diag(1/np.diag(O))
    e = np.ones(n)
    K = S + alpha * (2*np.eye(n) - np.outer(e, e))
    
    for cycle in range(max_iter):
        O_prev = O.copy()
        
        # Parallel loop over nodes
        for u in prange(n):
            Q, c, not_u, mask, K_uu, O_uu_inv = process_node(u, n, A_mask, K, O, C)
            
            if mask.size == 0:
                continue
                
            x = solve_node_qp(Q, c)  # Python solver call
            
            # Thread-safe matrix updates
            with numba.objmode(O='float64[:,:]', C='float64[:,:]'):
                O, C = update_matrices(O.copy(), C.copy(), u, not_u, mask, x, 
                                     K_uu, O_uu_inv, n)

        if np.linalg.norm(O - O_prev, 'fro') / np.linalg.norm(O_prev, 'fro') < tol:
            break

    # Final Laplacian constraints
    O = (O + O.T) / 2
    np.fill_diagonal(O, 0)
    np.fill_diagonal(O, -np.sum(O, axis=1))
    
    return O, C
