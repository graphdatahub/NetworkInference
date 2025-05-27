from collections.abc import Callable

import numpy as np
from scipy.linalg import eigh


class DiffusionMap:
    """
    A class for Diffusion Maps with functionality to reconstruct the Laplacian matrix.
    """

    def __init__(
        self,
        n_components: int = 2,
        t: float = 1.0,
        alpha: float = 0.0,
        epsilon: float = 1.0,
        kernel: Callable[[np.ndarray, np.ndarray], float] | None = None,
    ):
        self.n_components = n_components
        self.t = t
        self.alpha = alpha
        self.epsilon = epsilon
        self.kernel = kernel if kernel else self._default_gaussian_kernel
        self.input_dim: int | None = None
        self.lambdas: np.ndarray | None = None
        self.K: np.ndarray | None = None
        self.embedding: np.ndarray | None = None
        self.Laplacian: np.ndarray | None = None

    def _default_gaussian_kernel(self, x: np.ndarray, y: np.ndarray) -> float:
        """
        Default Gaussian kernel for computing similarity between two points.
        """
        return float(np.exp(-np.sum((x - y) ** 2) / self.epsilon))

    def _compute_kernel_matrix(self, X: np.ndarray) -> np.ndarray:
        """
        Compute the kernel matrix for the input data.
        """
        n_samples = X.shape[0]
        K = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j in range(i, n_samples):
                K[i, j] = self.kernel(X[i], X[j])
                K[j, i] = K[i, j]
        return K

    def _normalize_kernel(self, K: np.ndarray) -> np.ndarray:
        """
        Normalize the kernel matrix based on the alpha parameter to form Markov matrix.
        """
        if self.alpha > 0:
            D = np.sum(K, axis=1)
            D_alpha_inv = np.diag(D ** (-self.alpha))
            K = D_alpha_inv @ K @ D_alpha_inv
            D_new = np.sum(K, axis=1)
            D_new_inv = np.diag(1.0 / D_new)
            K = D_new_inv @ K
        else:
            D = np.sum(K, axis=1)
            D_inv = np.diag(1.0 / D)
            K = D_inv @ K
        return K

    def _decompose(self, K: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Perform eigendecomposition on the normalized kernel matrix.
        """
        lambdas, V = eigh(K, eigvals=(K.shape[0] - self.n_components - 1, K.shape[0] - 1))
        lambdas = lambdas[::-1]
        V = V[:, ::-1]
        return lambdas, V

    def reconstruct_laplacian(
        self, X: np.ndarray, kernel_matrix: np.ndarray | None = None
    ) -> np.ndarray:
        """
        Reconstruct the Laplacian matrix from the input data or kernel matrix.

        Args:
            X (np.ndarray): Input data matrix of shape (n_samples, n_features).
            kernel_matrix (Optional[np.ndarray]): Precomputed kernel matrix. If None, compute from X.

        Returns:
            np.ndarray: Reconstructed Laplacian matrix.
        """
        if kernel_matrix is None:
            K = self._compute_kernel_matrix(X)
        else:
            K = kernel_matrix.copy()

        # Compute degree matrix
        D = np.diag(np.sum(K, axis=1))

        # Compute unnormalized Laplacian L = D - K
        L = D - K

        # Optionally normalize the Laplacian (symmetric normalization)
        if self.alpha > 0:
            D_inv_sqrt = np.diag(1.0 / np.sqrt(np.diag(D)))
            L = D_inv_sqrt @ L @ D_inv_sqrt

        self.Laplacian = L
        return L

    def fit(self, X: np.ndarray) -> "DiffusionMap":
        """
        Fit the Diffusion Map model to the input data and reconstruct Laplacian.
        """
        if len(X.shape) != 2:
            raise ValueError("Input data must be a 2D array.")

        self.input_dim = X.shape[1]
        self.K = self._compute_kernel_matrix(X)

        # Reconstruct Laplacian before normalization
        self.reconstruct_laplacian(X, kernel_matrix=self.K)

        # Normalize kernel matrix to form Markov matrix
        self.K = self._normalize_kernel(self.K)

        # Perform eigendecomposition
        self.lambdas, V = self._decompose(self.K)

        # Compute the diffusion map embedding
        if self.lambdas is None:
            raise RuntimeError("Eigendecomposition failed to set eigenvalues.")
        self.embedding = (self.lambdas**self.t) * V.T

        return self

    def transform(self, X: np.ndarray | None = None) -> np.ndarray:
        """
        Transform the data into the reduced space using the fitted diffusion map.
        """
        if self.embedding is None:
            raise ValueError("Model must be fitted before transformation.")
        # Return training embedding if no new data is provided
        if X is None:
            return self.embedding.T
        raise NotImplementedError(
            "Out-of-sample extension for new data is not implemented."
        )

    def get_laplacian(self) -> np.ndarray:
        """
        Get the reconstructed Laplacian matrix.

        Returns:
            np.ndarray: Laplacian matrix.
        """
        if self.Laplacian is None:
            raise ValueError("Model must be fitted before accessing Laplacian matrix.")
        return self.Laplacian
