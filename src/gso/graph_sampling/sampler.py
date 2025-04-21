# TODO:
# - integrate knn case to sample directly
# - add possible extra args to sample, for future manifolds with other args (or adding mcmc call)
# - check geodesics to construct knn (and what it implies, also for cases with input structure)

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.csgraph import laplacian as sparse_laplacian

from ..core import Matrix, PointCloud, SparseMatrix
from ..manifolds import Manifold


class GraphSampler:
    """
    Constructs graph Laplacians with optional graph structure using
    a Gaussian kernel scaled for Laplace-Beltrami convergence.
    """

    def __init__(
        self,
        epsilon_scale_factor: float = 1.0,
        epsilon_scale_power_denom_const: float = 4.0,
        min_epsilon_factor: float = 1e-4,
    ):
        """
        Args:
            epsilon_scale_factor: Multiplicative factor C_M in epsilon scaling.
                                  Needs tuning based on manifold geometry/scale.
            epsilon_scale_power_denom_const: Constant C added to d in the
                                     scaling exponent denominator 2/(d+C).
                                     Common values are 4 or 6.
            min_epsilon_factor: A small additive term to the (log n / n) part
                                to stabilize epsilon for small n.
        """
        if epsilon_scale_factor <= 0:
            raise ValueError("epsilon_scale_factor must be positive.")
        if epsilon_scale_power_denom_const <= 0:
            raise ValueError("epsilon_scale_power_denom_const must be positive.")
        if min_epsilon_factor < 0:
            raise ValueError("min_epsilon_factor cannot be negative.")

        self.epsilon_scale_factor = epsilon_scale_factor
        self.epsilon_scale_power_denom_const = epsilon_scale_power_denom_const
        self.min_epsilon_factor = min_epsilon_factor

    def _calculate_epsilon(
        self, n_points: int, avg_degree: float, manifold_dim: int
    ) -> float:
        """Calculates the Gaussian kernel bandwidth epsilon."""
        log_n = np.log(max(2, n_points))  # avoid limit case n=1
        log_n_over_n = log_n / n_points

        if avg_degree < log_n:
            # Use avg_degree as lower-bound stabilizer
            scaling_base = max(log_n_over_n, avg_degree / n_points)
        else:
            scaling_base = log_n_over_n

        scaling_base += self.min_epsilon_factor
        exponent = 2 / (manifold_dim + self.epsilon_scale_power_denom_const)
        eps: float = self.epsilon_scale_factor * (scaling_base**exponent)
        return eps

    def _generate_knn_structure(self, geo_dists: Matrix) -> SparseMatrix:
        """Generates symmetric kNN graph with k ~ log(n) for sparsity"""
        n_nodes = geo_dists.shape[0]
        k = max(1, int(np.log(n_nodes)))

        # Find k nearest neighbors (excluding self)
        knn_indices = np.argpartition(geo_dists, k + 1, axis=1)[:, 1 : k + 1]

        # Build symmetric adjacency matrix
        rows = np.repeat(np.arange(n_nodes), k)
        cols = knn_indices.flatten()
        data = np.ones_like(rows)
        adj = coo_matrix((data, (rows, cols)), shape=(n_nodes, n_nodes)).tocsr()
        return adj.maximum(adj.T)  # symmetrize

    def create_weighted_graph(
        self,
        manifold: Manifold,
        n_nodes: int | None = None,
        structure_matrix: SparseMatrix | None = None,
    ) -> tuple[SparseMatrix, SparseMatrix, PointCloud, float]:
        """
        Generates a weighted graph Laplacian based on manifold samples and a
        fixed connectivity structure (input or kNN model).

        Workflow:
        1. Sample X = {x_i} ⊂ M
        2. Compute geodesic distances d_M(x_i, x_j)
        3. Generate kNN structure if none provided (k ~ log n)
        4. Calculate optimal ε(n,d)
        5. Build W_ij = exp(-d_M(x_i,x_j)^2/ε) for (i,j) in structure
        6. Construct L = D - W

        Args:
            manifold: The manifold object to sample from.
            n_nodes: The number of nodes (points) to sample.
            structure_matrix: A sparse matrix (n_nodes x n_nodes) containing
                              only 0s and 1s, where 1 indicates an edge whose
                              weight should be computed. Must be symmetric for
                              an undirected graph Laplacian.

        Returns:
            A tuple containing:
            - L (Matrix): The computed graph Laplacian.
            - W (Matrix): The weighted sparse adjacency matrix.
            - points (PointCloud): The sampled points used for weighting.
            - epsilon (float): The computed kernel bandwidth.
        """
        if structure_matrix is not None:
            if structure_matrix.shape[0] != structure_matrix.shape[1]:
                raise ValueError("Structure matrix must be square")
            n_nodes: int = structure_matrix.shape[0]  # type: ignore[no-redef]
        elif n_nodes is None:
            raise ValueError("Must provide either n_nodes or structure_matrix")

        assert n_nodes is not None
        assert structure_matrix is not None

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
        weights = np.exp(-(geo_dists[rows, cols] ** 2) / epsilon)
        W = csr_matrix((weights, (rows, cols)), shape=(n_nodes, n_nodes))

        # 5. Symmetrize weights if needed
        W = (W + W.T) / 2

        # 6. Construct combinatorial Laplacian
        L = sparse_laplacian(W, normed=False)

        return L, W, point_cloud, epsilon

    def create_weighted_graph_from_structure(
        self,
        manifold: Manifold,
        structure_matrix: SparseMatrix = None,
        to_symmetric: bool = True,
        normalized: bool = False,
    ) -> tuple[SparseMatrix, SparseMatrix, PointCloud, float]:
        structure_matrix = structure_matrix.tocsr()
        n_nodes: int = structure_matrix.shape[0]

        point_cloud = manifold.sample(n_nodes)

        d = manifold.intrinsic_dim
        epsilon = self._calculate_epsilon(n_nodes, structure_matrix.mean(), d)

        # Weight all edges based on geodesic distance
        rows, cols = structure_matrix.nonzero()
        weights = []
        for i, j in zip(rows, cols, strict=False):
            dist = manifold.geodesic(point_cloud[i], point_cloud[j])
            weights.append(np.exp(-(dist**2) / epsilon))

        W = csr_matrix((np.array(weights), (rows, cols)), shape=(n_nodes, n_nodes))

        if to_symmetric:
            W = (W + W.T) / 2

        # Compute Laplacian: large weight = important commute time
        L = sparse_laplacian(W, normed=normalized, return_diag=False)

        return L, W, point_cloud, epsilon
