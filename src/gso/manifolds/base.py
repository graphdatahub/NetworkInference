# TODO: 
# - Debug uniform_sampling
# - Accelerate with numba
# - Write mcmc_sample
# - Check Jacobian and _is_smooth_v2

import abc
import numpy as np
from numba import njit, prange
import sympy as sp
from typing import Optional
from scipy.optimize import fsolve, root
from dataclasses import dataclass, field
from ..core.types import PointCloud


class Manifold(abc.ABC):
    """Abstract base class for manifolds."""

    @property
    @abc.abstractmethod
    def ambient_dim(self) -> int:
        """Returns the dimension of the ambient space R^N."""
        pass

    @property
    @abc.abstractmethod
    def intrinsic_dim(self) -> int:
        """Returns the intrinsic dimension 'd' of the manifold."""
        pass

    @abc.abstractmethod
    def sample(self, n_points: int, seed: int | None = None) -> PointCloud:
        """
        Samples points (approximately) uniformly from the manifold.

        Args:
            n_points: Number of points to sample.
            seed: Optional random seed for reproducibility.

        Returns:
            A PointCloud array of shape (n_points, ambient_dim).
        """
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(ambient_dim={self.ambient_dim}, intrinsic_dim={self.intrinsic_dim})"


@dataclass
class AlgebraicManifold(Manifold):
    """Represents a continuous algebraic manifold defined by polynomial equations."""
    num_equations: int                 # Number of polynomial equations (k)
    num_variables: int                 # Number of ambient space variables (n)
    max_degree: int                    # Maximum polynomial degree
    random_seed: Optional[int] = None  # Seed for reproducibility
    
    # Symbolic components
    variables: list[sp.Symbol] = field(init=False, repr=False)
    equations: list[sp.Expr] = field(init=False, repr=False)
    jacobian: sp.Matrix = field(init=False, repr=False)
    
    def __post_init__(self):
        if self.num_equations >= self.num_variables:
            raise ValueError("Requires k < n for continuous manifolds")

        np.random.seed(self.random_seed)
        self.variables = [sp.Symbol(f'x{i}') for i in range(self.num_variables)]
        self.equations = self._generate_random_polynomials()
        self.jacobian = self._compute_jacobian()

        # Compile symbolic metric tensor into a fast NumPy function
        metric_symbolic = self.jacobian.T * self.jacobian
        self.metric_func = sp.lambdify(self.variables, metric_symbolic, 'numpy')
    
    @property
    def ambient_dim(self) -> int:
        return self.num_variables
    
    @property
    def intrinsic_dim(self) -> int:
        return self.num_variables - self.num_equations
    
    def _generate_random_polynomials(self) -> list[sp.Expr]:
        """Generates random polynomials with controlled structure."""
        return [self._create_polynomial() for _ in range(self.num_equations)]
    
    def _create_polynomial(self) -> sp.Expr:
        """Constructs a single random polynomial."""
        return sum(
            np.random.uniform(-1, 1) * var**np.random.randint(1, self.max_degree+1)
            for var in self.variables
        )
    
    def _compile_functions(self):
        """Compile symbolic expressions to numeric functions."""
        # Jacobian for smoothness checks
        jacobian = sp.Matrix([[sp.diff(eq, var) for var in self.variables] 
                            for eq in self.equations])
        self.jacobian_func = sp.lambdify(self.variables, jacobian, 'numpy')
        
        # Metric tensor for geodesics
        metric = jacobian.T * jacobian
        self.metric_func = sp.lambdify(self.variables, metric, 'numpy')

    def _compute_jacobian(self) -> sp.Matrix:
        """Computes symbolic Jacobian matrix."""
        return sp.Matrix([[sp.diff(eq, var) for var in self.variables] for eq in self.equations])
    
    def uniform_sampling(self, n_points: int, domain: tuple[float, float] = (-2.0, 2.0), tol: float = 1e-6) -> PointCloud:
        """Uniform sampling via adaptive slicing (Dufresne et al., 2018)"""
        points = []
        d = self.intrinsic_dim
        
        while len(points) < n_points:
            # 1. Random affine slice
            slice_coeffs = np.random.randn(d, self.ambient_dim)
            slice_intercepts = np.random.uniform(*domain, size=d)
            
            # 2. Solve augmented system: F(x)=0 ∧ A·x = b
            def equations(x):
                main_eqs = [eq.subs(dict(zip(self.variables, x))) for eq in self.equations]
                slice_eqs = slice_coeffs @ x - slice_intercepts
                return np.concatenate([main_eqs, slice_eqs])
                
            # 3. Homotopy continuation with random initialization
            x0 = np.random.uniform(*domain, self.ambient_dim)
            sol = root(equations, x0, method='lm', tol=tol)
            
            # 4. Rejection sampling for uniformity
            if sol.success and self._is_smooth(sol.x):
                points.append(sol.x)
                
        return np.array(points)

    def sample(self, n_points: int, scale: float = 1e2, max_attempts: int = 10) -> tuple[PointCloud, int]:
        """Samples points on the manifold using numerical root finding."""
        points = []
        attempt = 0
        d = self.num_variables - self.num_equations  # Degrees of freedom
        
        while len(points) < n_points and attempt < max_attempts:
            new_points = self._sample_batch(n_points - len(points), d, scale)
            valid_points = [p for p in new_points if self._is_smooth(p)]
            points.extend(valid_points)
            attempt += 1
            
        return np.array(points[:n_points])
    
    def _sample_batch(self, batch_size: int, d: int, scale: float) -> PointCloud:
        """Generate a batch of candidate points."""
        points = []
        for _ in range(batch_size):
            fixed_vars = scale * np.random.randn(d)
            success, point = self._solve_with_fixed_vars(fixed_vars)
            if success:
                points.append(point)
        return points
    
    def mcmc_sample(self, n_points: int, step_size: float = 0.1, burn_in: int = 1000) -> PointCloud:
        """Manifold-adjusted Langevin MCMC (Cheng et al. 2022)"""
        pass
        
    def _solve_with_fixed_vars(self, fixed_vars: np.ndarray) -> tuple[bool, np.ndarray]:
        """Numerical solver with fixed variables."""
        def residual(remaining_vars):
            full_vars = np.concatenate([fixed_vars, remaining_vars])
            subs = dict(zip(self.variables, full_vars))
            return [float(eq.subs(subs)) for eq in self.equations]

        # Multi-start optimization with different initial guesses
        for _ in range(3):  # Try up to 3 different initial guesses
            sol, info, success = fsolve(residual, np.random.randn(self.num_equations), 
                                      full_output=True)[:3]
            if success and self._is_valid_solution(sol, fixed_vars):
                full_point = np.concatenate([fixed_vars, sol])
                return True, full_point
        return False, np.zeros(self.num_variables)
    
    def _is_valid_solution(self, sol: np.ndarray, fixed_vars: np.ndarray) -> bool:
        """Check solution validity and smoothness."""
        try:
            full_point = np.concatenate([fixed_vars, sol])
            return self._is_smooth(full_point)
        except (ValueError, TypeError):
            return False
    
    def _is_smooth(self, point: np.ndarray, tol: float = 1e-6) -> bool:
        """Checks if the Jacobian has full rank at the point."""
        subs = {var: val for var, val in zip(self.variables, point)}
        numeric_jacobian = np.array(self.jacobian.subs(subs).evalf(), dtype=float)
        return np.linalg.matrix_rank(numeric_jacobian, tol=tol) == self.num_equations

    def _is_smooth_v2(self, point: np.ndarray) -> bool:
        """Check Jacobian rank meets regular value theorem (Tu 2011)"""
        subs = dict(zip(self.variables, point))
        J = np.array([[float(sp.diff(eq, var).subs(subs)) 
                    for var in self.variables] 
                    for eq in self.equations], dtype=float)
        return np.linalg.matrix_rank(J) == self.num_equations
    
    def compute_geodesics(self, points: np.ndarray) -> np.ndarray:
        """Compute all pairwise geodesic distances using Numba."""
        n = len(points)
        metrics = np.array([self.metric_func(*p) for p in points])
        distances = np.zeros((n, n))
        
        # Precompute metrics as 3D array for Numba
        metric_tensors = np.stack([m.reshape(self.num_variables, self.num_variables) 
                                 for m in metrics])
        
        # Parallel computation over upper triangle
        _compute_geodesic_matrix(points, metric_tensors, distances)
        return distances + distances.T  # Symmetrize
    
    def geodesic(p1, p2) -> float:
        """Compute geodesic distance between two points with metric-aware optimization."""
        ...

    def get_equations(self):
        return self.equations

    def get_jacobian(self):
        return self.jacobian
    
    def get_seed(self):
        return self.random_seed

@njit(parallel=True)
def _compute_geodesic_matrix(points: np.ndarray, 
                           metrics: np.ndarray,
                           distances: np.ndarray):
    """Numba-accelerated geodesic computation."""
    n = len(points)
    for i in prange(n):
        for j in range(i+1, n):
            distances[i,j] = _geodesic_length(points[i], points[j], 
                                            metrics[i], metrics[j])

@njit
def _geodesic_length(p1: np.ndarray, p2: np.ndarray, 
                   metric1: np.ndarray, metric2: np.ndarray) -> float:
    """Compute geodesic length between two points with metric-aware optimization."""
    # Manually create linear interpolation path
    path = np.empty((10, p1.shape[0]))
    for k in range(10):
        t = k / 9.0  # 10 points from 0.0 to 1.0
        path[k] = p1 * (1 - t) + p2 * t
    
    energy = 0.0
    
    for k in range(len(path)-1):
        dx = path[k+1] - path[k]
        # Use average metric along the segment
        M = 0.5*(metric1 + metric2) if k < 5 else metric2
        energy += np.sqrt(np.abs(dx.T @ M @ dx))  # Absolute value for numerical stability
        
    return energy