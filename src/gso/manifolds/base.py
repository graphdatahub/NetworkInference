from abc import ABC, abstractmethod

from numpy import zeros

from ..core import Matrix, Point, PointCloud


class Manifold(ABC):
    """Abstract base class for manifolds.

    Any concrete subclass requires:
      - intrinsic_dim
      - sample(n_points)
      - geodesic(p1, p2)
    for the GraphSampler to work.
    """

    @property
    @abstractmethod
    def ambient_dim(self) -> int:
        """Returns the number of coordinates."""
        pass

    @property
    @abstractmethod
    def intrinsic_dim(self) -> int:
        """Returns the intrinsic dimension of the manifold."""
        pass

    @abstractmethod
    def sample(
        self,
        n_points: int,
        domain: tuple[float, float] = (-2.0, 2.0),
        tol: float = 1e-6,
        max_attempts: int = 1000,
    ) -> PointCloud:
        """
        Samples points (approximately) uniformly from the manifold.

        Args:
            n_points: Number of points to sample.
            seed: Optional random seed for reproducibility.

        Returns:
            A PointCloud array of shape (n_points, ambient_dim).
        """
        pass

    @abstractmethod
    def geodesic(self, p1: Point, p2: Point) -> float:
        """
        Compute geodesic distance between two points on the manifold.

        Args:
            p1 and p2: Two points of shape (ambient_dim,).
        """
        pass

    def compute_geodesics(self, points: PointCloud) -> Matrix:
        """Compute all pairwise geodesic distances."""
        n = len(points)
        distances = zeros((n, n))

        # Calculate upper triangle
        for i in range(n):
            for j in range(i + 1, n):
                distances[i, j] = self.geodesic(points[i], points[j])

        return distances + distances.T

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(ambient_dim={self.ambient_dim}, intrinsic_dim={self.intrinsic_dim})"
