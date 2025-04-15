# graph_sampling/sampler.py
import numpy as np
from scipy.sparse import csr_matrix, diags
from scipy.sparse.csgraph import laplacian as sparse_laplacian
from scipy.spatial.distance import cdist

from core.types import PointCloud, SparseMatrix, Matrix
from manifolds.base import Manifold

class GraphSampler:
    """
    Samples points from a manifold and assigns weights to a fixed graph
    structure using a Gaussian kernel scaled for Laplace-Beltrami convergence.
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

    def _calculate_epsilon(self, n_points: int, manifold_dim: int) -> float:
        """Calculates the Gaussian kernel bandwidth epsilon."""
        if n_points < 2:
             # Cannot determine scale from a single point, return default scale factor
             # Or raise error? Returning scale factor seems slightly more robust for edge cases.
             # Warning: This deviates from convergence theory if n=1.
             print(f"Warning: n_points={n_points}. Cannot apply scaling theory. Using epsilon={self.epsilon_scale_factor}")
             return self.epsilon_scale_factor
             # raise ValueError("Need at least 2 points for epsilon scaling.")
        if manifold_dim <= 0:
            raise ValueError("Manifold dimension must be positive.")

        log_n = np.log(n_points)
        scaling_base = max(0.0, log_n / n_points) + self.min_epsilon_factor
        exponent = 2.0 / (manifold_dim + self.epsilon_scale_power_denom_const)
        epsilon = self.epsilon_scale_factor * (scaling_base ** exponent)

        if epsilon <= 1e-15: # Check for effectively zero epsilon
            raise ValueError(
                f"Calculated epsilon={epsilon} is too close to zero. "
                f"Check parameters (scale_factor={self.epsilon_scale_factor}), "
                f"increase n_points (currently {n_points}), or increase min_epsilon_factor."
             )
        return epsilon

    def create_weighted_graph(
        self,
        manifold: Manifold,
        n_nodes: int,
        structure_matrix: SparseMatrix,
        sampling_seed: int | None = None,
    ) -> tuple[Matrix, Matrix, PointCloud, float]:
        """
        Generates a weighted graph Laplacian based on manifold samples and a
        fixed connectivity structure.

        Args:
            manifold: The manifold object to sample from.
            n_nodes: The number of nodes (points) to sample.
            structure_matrix: A sparse matrix (n_nodes x n_nodes) containing
                              only 0s and 1s, where 1 indicates an edge whose
                              weight should be computed. Must be symmetric for
                              an undirected graph Laplacian.
            sampling_seed: Optional random seed for point sampling.

        Returns:
            A tuple containing:
            - L (Matrix): The computed combinatorial graph Laplacian (L = D - W).
            - W (Matrix): The weighted sparse adjacency matrix.
            - points (PointCloud): The sampled points used for weighting.
            - epsilon (float): The computed kernel bandwidth epsilon used.
        """
        if not isinstance(structure_matrix, csr_matrix):
            try:
                structure_matrix = structure_matrix.tocsr()
            except AttributeError:
                 raise TypeError("structure_matrix must be a SciPy sparse matrix, preferably CSR.")
        if structure_matrix.shape != (n_nodes, n_nodes):
            raise ValueError(f"structure_matrix shape {structure_matrix.shape} "
                             f"does not match n_nodes ({n_nodes}).")
        # Optional check for symmetry:
        # if np.any(structure_matrix != structure_matrix.T):
        #     print("Warning: structure_matrix is not symmetric. Resulting Laplacian might not be either.")
        # Check if structure_matrix contains only 0s and 1s (can be slow for large matrices)
        # if not np.all(np.isin(structure_matrix.data, [0, 1])):
        #     raise ValueError("structure_matrix data should only contain 0s and 1s.")


        # 1. Sample points from the manifold
        points: PointCloud = manifold.sample(n_nodes, seed=sampling_seed)

        # 2. Calculate epsilon
        epsilon = self._calculate_epsilon(n_nodes, manifold.intrinsic_dim)

        # 3. Compute weights for existing edges
        rows, cols = structure_matrix.nonzero()
        weights = np.zeros(len(rows), dtype=np.float64)

        # Efficiently calculate distances only for pairs defined by non-zero entries
        # Consider only upper triangle pairs to avoid redundant distance calc?
        # If structure is symmetric, rows[i],cols[i] covers both (i,j) and (j,i) eventually.
        # We need weights for all non-zero entries defined in structure_matrix.

        processed_pairs = set() # Keep track if we handle upper/lower triangle separately
        valid_indices_mask = np.ones(len(rows), dtype=bool) # Mask for weights array

        # It's simpler to compute weights for all given non-zero indices (rows[i], cols[i])
        # even if structure_matrix isn't strictly upper/lower triangular
        unique_indices_for_dist = np.array(list(set(np.concatenate([rows, cols]))))
        subset_points = points[unique_indices_for_dist]
        sq_dists_subset = squareform(pdist(subset_points, metric='sqeuclidean'))

        # Create mapping from original index to subset index
        idx_map = {orig_idx: subset_idx for subset_idx, orig_idx in enumerate(unique_indices_for_dist)}

        for k in range(len(rows)):
            i, j = rows[k], cols[k]
            if i == j: # Skip self-loops for weight calculation if desired
                 weights[k] = 0.0 # Or handle as per convention (often 0)
                 continue

            # Find indices in the subset distance matrix
            try:
                 sub_i, sub_j = idx_map[i], idx_map[j]
                 dist_sq = sq_dists_subset[sub_i, sub_j]
                 weights[k] = np.exp(-dist_sq / epsilon)
            except KeyError:
                 # This shouldn't happen if idx_map is built correctly
                 print(f"Warning: Index mapping error for pair ({i}, {j}). Setting weight to 0.")
                 weights[k] = 0.0


        # 4. Construct weighted adjacency matrix W
        W = csr_matrix((weights, (rows, cols)), shape=(n_nodes, n_nodes))

        # Ensure symmetry if the input structure was meant to be symmetric
        # W = (W + W.T) / 2.0 # Can uncomment if strict symmetry is needed


        # 5. Compute Combinatorial Laplacian L = D - W
        L = sparse_laplacian(W, normed=False, return_diag=False)

        # Ensure L is symmetric (sparse_laplacian should preserve if W is symmetric)
        # L = (L + L.T) / 2.0 # Can uncomment if strict symmetry is needed

        return L, W, points, epsilon

