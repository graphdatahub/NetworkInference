# TODO:
# - Accelerate with numba
# - Write mcmc_sample

from dataclasses import dataclass, field

import numpy as np
import sympy as sp
from scipy.optimize import root

from ..core import Matrix, Point, PointCloud
from .base import Manifold


@dataclass
class AlgebraicManifold(Manifold):
    """Represents a continuous algebraic manifold defined by polynomial equations."""

    num_equations: int
    num_variables: int
    max_degree: int
    random_seed: int | None = None

    # Symbolic components
    variables: list[sp.Symbol] = field(init=False, repr=False)
    equations: list[sp.Expr] = field(init=False, repr=False)
    jacobian: sp.Matrix = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.num_equations >= self.num_variables:
            raise ValueError("Requires k < n for continuous manifolds")

        np.random.seed(self.random_seed)
        self.variables = [sp.Symbol(f"x{i}") for i in range(self.num_variables)]
        self.equations = self._generate_random_polynomials()
        self.jacobian = self._compute_jacobian()
        self._compile_functions()

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(num_equations={self.num_equations}, num_variables={self.num_variables}, max_degree={self.max_degree}, random_seed={self.random_seed})"

    @property
    def ambient_dim(self) -> int:
        return self.num_variables

    @property
    def intrinsic_dim(self) -> int:
        return self.num_variables - self.num_equations

    # -----------------------------
    # Public functions
    # -----------------------------
    def get_equations(self) -> list[sp.Expr]:
        return self.equations

    def get_jacobian(self) -> sp.Matrix:
        return self.jacobian

    def get_seed(self) -> int | None:
        return self.random_seed

    def mcmc_sample(
        self, n_points: int, step_size: float = 0.1, burn_in: int = 1000
    ) -> PointCloud:
        """Manifold-adjusted Langevin MCMC (Cheng et al. 2022)"""
        pass

    def sample(
        self,
        n_points: int,
        domain: tuple[float, float] = (-2.0, 2.0),
        tol: float = 1e-6,
        max_attempts: int = 1000,
    ) -> PointCloud:
        """Uniform sampling via adaptive slicing (Dufresne et al., 2018)"""
        points: list[Point] = []
        d = self.intrinsic_dim

        while len(points) < n_points and max_attempts > 0:
            # Create random slice matching manifold dimensions
            slice_coeffs = np.random.randn(d, self.ambient_dim)
            slice_intercepts = np.random.uniform(*domain, size=d)

            # Initial guess using pseudoinverse
            try:
                x0 = np.linalg.pinv(slice_coeffs) @ slice_intercepts
            except np.linalg.LinAlgError:
                x0 = np.random.uniform(*domain, self.ambient_dim)

            # Solve augmented system
            sol = root(
                lambda x, sc=slice_coeffs, si=slice_intercepts: np.concatenate(
                    [self.eq_func(*x), sc @ x - si]
                ),
                x0,
                method="lm",
                tol=tol,
            )

            if sol.success:
                point = sol.x
                if self._is_valid_point(point, domain, tol):
                    points.append(point)

            max_attempts -= 1

        return np.array(points[:n_points])

    def geodesic(self, p1: Point, p2: Point) -> float:
        """Compute geodesic distance between two points using midpoint metric approximation."""
        midpoint: Point = (p1 + p2) / 2
        G: Matrix = self.metric_func(
            *midpoint
        )  # Get metric tensor at midpoint, shape: (ambient_dim, ambient_dim)
        delta: Point = p2 - p1
        distance_squared: float = delta @ G @ delta  # Riemannian norm squared
        return distance_squared

    # -----------------------------
    # Private functions
    # -----------------------------
    def _generate_random_polynomials(self) -> list[sp.Expr]:
        """Generates random polynomials with controlled structure."""
        return [self._create_polynomial() for _ in range(self.num_equations)]

    def _create_polynomial(self) -> sp.Expr:
        """Constructs a single random polynomial."""
        return sum(
            np.random.uniform(-1, 1) * var ** np.random.randint(1, self.max_degree + 1)
            for var in self.variables
        )

    def _compute_jacobian(self) -> sp.Matrix:
        """Computes symbolic Jacobian matrix."""
        return sp.Matrix(
            [[sp.diff(eq, var) for var in self.variables] for eq in self.equations]
        )

    def _compile_functions(self) -> None:
        """Compile symbolic expressions to numeric functions."""
        self.eq_func = sp.lambdify(self.variables, self.equations, "numpy")
        self.jacobian_func = sp.lambdify(self.variables, self.jacobian, "numpy")
        metric_symbolic = self.jacobian.T * self.jacobian
        self.metric_func = sp.lambdify(self.variables, metric_symbolic, "numpy")

    def _is_valid_point(self, x: Point, domain: tuple[float, float], tol: float) -> bool:
        """Complete validity check."""
        return (
            self._in_domain(x, domain)
            and self._equations_satisfied(x, tol)
            and self._is_smooth(x, tol)
        )

    def _in_domain(self, x: Point, domain: tuple[float, float]) -> bool:
        """Dmain check."""
        return bool(np.all((x >= domain[0]) & (x <= domain[1])))

    def _equations_satisfied(self, x: Point, tol: float) -> bool:
        """Check equation residuals."""
        return bool(np.max(np.abs(self.eq_func(*x))) < tol)

    def _is_smooth(self, x: Point, tol: float) -> bool:
        """Jacobian rank check."""
        J = self.jacobian_func(*x)
        return bool(np.linalg.matrix_rank(J, tol=tol) == self.num_equations)
