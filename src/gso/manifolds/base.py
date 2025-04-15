import abc
import numpy as np
from core.types import PointCloud

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
