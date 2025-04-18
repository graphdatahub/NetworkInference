"""
TODO: Optimize by not computing distances for zero-edges. 
"""

import numpy as np
from scipy.sparse import csr_matrix, coo_matrix, dok_matrix
from scipy.sparse.csgraph import laplacian as sparse_laplacian
from typing import Optional, Tuple
from ..manifolds.base import Manifold

class GraphSampler:
    """
    Constructs graph Laplacians with optional graph structure.
    """

    def __init__(self, 
                 epsilon_scale_factor: float = 1.0,
                 epsilon_scale_power_denom_const: float = 4.0,
                 min_epsilon_factor: float = 1e-4):
        self.epsilon_scale_factor = epsilon_scale_factor
        self.epsilon_scale_power_denom_const = epsilon_scale_power_denom_const
        self.min_epsilon_factor = min_epsilon_factor

    def _calculate_epsilon(self, n_points: int, avg_degree: float, manifold_dim: int) -> float:
        log_n = np.log(max(2, n_points))

        if avg_degree < log_n:
            # Use avg_degree as lower-bound stabilizer
            scaling_base = max(log_n/n_points, avg_degree/n_points)
        else:
            scaling_base = log_n/n_points
        
        scaling_base = max(0.0, log_n/n_points) + self.min_epsilon_factor
        exponent = 2/(manifold_dim + self.epsilon_scale_power_denom_const)
        return self.epsilon_scale_factor * (scaling_base**exponent)

    def _generate_knn_structure(self, geo_dists: np.ndarray) -> csr_matrix:
        """Generates symmetric kNN graph with k ~ log(n) for sparsity"""
        n_nodes = geo_dists.shape[0]
        k = max(1, int(np.log(n_nodes)))
        
        # Find k nearest neighbors (excluding self)
        knn_indices = np.argpartition(geo_dists, k+1, axis=1)[:, 1:k+1]
        
        # Build symmetric adjacency matrix
        rows = np.repeat(np.arange(n_nodes), k)
        cols = knn_indices.flatten()
        data = np.ones_like(rows)
        adj = coo_matrix((data, (rows, cols)), shape=(n_nodes, n_nodes)).tocsr()
        return adj.maximum(adj.T)  # Symmetrize

    def create_weighted_graph(self,
                            manifold: Manifold,
                            n_nodes: Optional[int] = None,
                            structure_matrix: Optional[csr_matrix] = None
                            ) -> Tuple[csr_matrix, csr_matrix, np.ndarray, float]:
        """
        Mathematical workflow:
        1. Sample X = {x_i} ⊂ M
        2. Compute geodesic distances d_M(x_i, x_j)
        3. Generate kNN structure if none provided (k ~ log n)
        4. Calculate optimal ε(n,d)
        5. Build W_ij = exp(-d_M(x_i,x_j)^2/ε) for (i,j) in structure
        6. Construct L = D - W
        """
        if structure_matrix is not None:
            if structure_matrix.shape[0] != structure_matrix.shape[1]:
                raise ValueError("Structure matrix must be square")
            n_nodes: int = structure_matrix.shape[0]
        elif n_nodes is None:
            raise ValueError("Must provide either n_nodes or structure_matrix")
        
        # 1. Sample points and compute geodesics
        point_cloud = manifold.sample(n_nodes)
        geo_dists = manifold.compute_geodesics(point_cloud)

        # 2. Generate structure if not provided
        if structure_matrix is None:
            structure_matrix = self._generate_knn_structure(geo_dists)
        else:
            structure_matrix = structure_matrix.tocsr()

        # 3. Calculate convergence-optimal bandwidth
        d = manifold.intrinsic_dim
        epsilon = self._calculate_epsilon(n_nodes, structure_matrix.mean(), d)

        # 4. Build weighted adjacency matrix
        rows, cols = structure_matrix.nonzero()
        weights = np.exp(-geo_dists[rows, cols]**2 / epsilon)
        W = csr_matrix((weights, (rows, cols)), shape=(n_nodes, n_nodes))

        # 5. Symmetrize weights if needed
        W = (W + W.T)/2  # Ensures symmetric weights

        # 6. Construct combinatorial Laplacian
        L = sparse_laplacian(W, normed=False)

        return L, W, point_cloud, epsilon

    def create_weighted_graph_from_structure(self,
                            manifold: Manifold,
                            structure_matrix: csr_matrix = None, 
                            to_symmetric: bool = True,
                            normalized: bool = False
                            ) -> Tuple[csr_matrix, csr_matrix, np.ndarray, float]:
        structure_matrix = structure_matrix.tocsr()
        n_nodes: int = structure_matrix.shape[0]
        
        point_cloud = manifold.sample(n_nodes)
        
        d = manifold.intrinsic_dim
        epsilon = self._calculate_epsilon(n_nodes, structure_matrix.mean(), d)

        rows, cols = structure_matrix.nonzero()
        weights = []
        for i, j in zip(rows, cols):
            dist = manifold.geodesic(point_cloud[i], point_cloud[j])
            weights.append(np.exp(-dist**2 / epsilon))

        W = csr_matrix(
            (np.array(weights), (rows, cols)), 
            shape=(n_nodes, n_nodes)
        )

        if to_symmetric:
            W = (W + W.T)/2

        L = sparse_laplacian(W, normed=normalized)

        return L, W, point_cloud, epsilon
