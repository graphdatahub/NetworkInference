import numpy as np
import numpy.typing as npt
from ..core.types import (
    GraphSignals,
    Matrix,
    PointCloud,  # For optional initial phases based on coords
)
from scipy.linalg import pinvh
from scipy.sparse import csr_matrix, issparse


class SignalGenerator:
    """Generates graph signals using different models."""

    def generate_gaussian_signal(
        self,
        laplacian: Matrix,
        n_samples: int,
        mean: npt.NDArray[np.float64] | None = None,
        noise_sigma: float = 0.0,
        seed: int | None = None,
    ) -> GraphSignals:
        """
        Generates signals from a Gaussian Markov Random Field (GMRF) defined
        by the graph Laplacian L (as precision matrix).

        Assumes x ~ N(mean, (L + delta*I)^-1) or N(mean, L^+) where L^+ is pseudoinverse.
        Using pseudoinverse L^+ relates directly to intrinsic variations on the graph.

        Args:
            laplacian: The graph Laplacian matrix (n_nodes x n_nodes).
            n_samples: Number of signal samples (m).
            mean: Optional mean vector (n_nodes,). Defaults to zero vector.
            noise_sigma: Standard deviation of optional additive Gaussian noise
                         applied after sampling from the GMRF. Defaults to 0.
            seed: Optional random seed.

        Returns:
            GraphSignals array of shape (n_samples, n_nodes).
        """
        if issparse(laplacian):
            L_dense = laplacian.toarray()
        else:
            L_dense = np.asarray(laplacian)  # Ensure it's a numpy array

        n_nodes = L_dense.shape[0]
        if L_dense.shape != (n_nodes, n_nodes):
            raise ValueError("Laplacian must be a square matrix.")

        if mean is None:
            mean_vec = np.zeros(n_nodes)
        else:
            mean_vec = np.asarray(mean)
            if mean_vec.shape != (n_nodes,):
                raise ValueError(
                    f"Mean vector shape {mean_vec.shape} incompatible "
                    f"with n_nodes ({n_nodes})."
                )

        # Use pseudo-inverse of Laplacian to define covariance
        # Add small ridge to improve condition number before pinvh if needed
        ridge = 1e-10
        try:
            covariance = pinvh(L_dense + ridge * np.identity(n_nodes))
            # Ensure symmetry (pinvh should return symmetric, but for safety)
            covariance = (covariance + covariance.T) / 2.0
        except np.linalg.LinAlgError:
            raise RuntimeError("Failed to compute pseudo-inverse of the Laplacian.")

        rng = np.random.default_rng(seed)

        # Sample from multivariate normal
        signals: GraphSignals = rng.multivariate_normal(
            mean=mean_vec, cov=covariance, size=n_samples, check_valid="warn"
        )

        # Add optional observation noise
        if noise_sigma > 0:
            noise = rng.normal(scale=noise_sigma, size=signals.shape)
            signals += noise

        return signals

    def generate_kuramoto_signal(
        self,
        W: Matrix,
        n_timesteps: int,
        coupling_K: float,
        dt: float = 0.1,
        natural_frequencies: npt.NDArray[np.float64] | None = None,
        initial_phases: npt.NDArray[np.float64] | PointCloud | None = None,
        seed: int | None = None,
    ) -> GraphSignals:
        """
        Simulates Kuramoto dynamics on a graph with weighted adjacency W.

        Args:
            W: Weighted adjacency matrix (n_nodes x n_nodes). Can be sparse or dense.
            n_timesteps: Number of time steps (m).
            coupling_K: Coupling strength.
            dt: Integration time step.
            natural_frequencies: Natural frequencies (n_nodes,). Defaults to random uniform(-0.5, 0.5).
            initial_phases: Initial phases (n_nodes,). Defaults to random uniform(0, 2*pi).
                            Can also be a PointCloud, where e.g. the first coordinate is used.
            seed: Optional random seed for frequency/phase initialization.

        Returns:
            GraphSignals array of phases, shape (n_timesteps, n_nodes).
        """
        if issparse(W):
            W_sparse = W.tocsr()  # Ensure CSR for efficient row slicing if needed
        else:
            W_sparse = csr_matrix(W)  # Convert dense to sparse

        n_nodes = W_sparse.shape[0]
        if W_sparse.shape != (n_nodes, n_nodes):
            raise ValueError("Adjacency matrix W must be square.")
        if n_timesteps <= 0:
            raise ValueError("n_timesteps must be positive.")
        if dt <= 0:
            raise ValueError("Time step dt must be positive.")

        rng = np.random.default_rng(seed)

        # Initialize natural frequencies (omega)
        if natural_frequencies is None:
            omegas = rng.uniform(-0.5, 0.5, n_nodes)
        else:
            omegas = np.asarray(natural_frequencies)
            if omegas.shape != (n_nodes,):
                raise ValueError(
                    f"natural_frequencies shape {omegas.shape} "
                    f"incompatible with n_nodes ({n_nodes})."
                )

        # Initialize phases (theta)
        if initial_phases is None:
            phases = rng.uniform(0, 2 * np.pi, n_nodes)
        else:
            # Allow initialization from point coordinates (e.g., first coordinate)
            init_ph = np.asarray(initial_phases)
            if init_ph.ndim == 2 and init_ph.shape[0] == n_nodes:  # Looks like PointCloud
                print(
                    "Warning: Initializing Kuramoto phases from first coordinate of input."
                )
                phases = init_ph[:, 0]
                # Optionally scale phases to [0, 2*pi]
                min_ph, max_ph = np.min(phases), np.max(phases)
                if max_ph > min_ph:
                    phases = 2 * np.pi * (phases - min_ph) / (max_ph - min_ph)
                else:
                    phases = np.zeros(n_nodes)  # All same coord -> zero phase
            elif init_ph.shape == (n_nodes,):  # Looks like phase vector
                phases = init_ph
            else:
                raise ValueError(
                    f"initial_phases shape {init_ph.shape} incompatible "
                    f"with n_nodes ({n_nodes})."
                )

        # Store graph signals (phases over time)
        graph_signals: GraphSignals = np.zeros((n_timesteps, n_nodes))

        # Simulation loop (Euler method)
        rows, cols = W_sparse.nonzero()
        weights = W_sparse.data

        for t in range(n_timesteps):
            graph_signals[t, :] = phases

            # Calculate phase differences efficiently using sparse structure
            # sin(theta_j - theta_i)
            phase_diffs = phases[cols] - phases[rows]
            sin_phase_diffs = np.sin(phase_diffs)

            # Interaction term for each node i: sum_j W_ij * sin(theta_j - theta_i)
            # Need to aggregate contributions based on 'rows' index
            interaction = np.zeros(n_nodes)
            np.add.at(interaction, rows, weights * sin_phase_diffs)  # Efficient summation

            # Update phases
            dtheta_dt = omegas + coupling_K * interaction
            phases = (phases + dtheta_dt * dt) % (2 * np.pi)  # Keep in [0, 2*pi]

        return graph_signals
