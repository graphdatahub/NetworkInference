import numpy as np
import pytest
from numpy.testing import assert_allclose

from gso.core import Point, PointCloud
from gso.manifolds import Manifold


class DummyManifold(Manifold):
    """Concrete implementation for testing base class functionality"""
    @property
    def ambient_dim(self) -> int:
        return 3

    @property
    def intrinsic_dim(self) -> int:
        return 2

    def sample(self, n_points: int, **kwargs) -> PointCloud:
        return np.array([
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0]
        ])[:n_points]

    def geodesic(self, p1: Point, p2: Point) -> float:
        return np.linalg.norm(p1 - p2)

@pytest.fixture
def manifold():
    """Fixture providing instantiable abstract class"""
    DummyManifold.__abstractmethods__ = set()
    return DummyManifold()

def test_properties(manifold):
    assert manifold.ambient_dim == 3
    assert manifold.intrinsic_dim == 2

def test_compute_geodesics(manifold):
    points = np.array([
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0]
    ])

    result = manifold.compute_geodesics(points)

    # Check matrix properties
    assert result.shape == (3, 3)
    assert np.allclose(result, result.T), "Matrix should be symmetric"
    assert np.all(np.diag(result) == 0), "Diagonal should be zero"

def test_repr(manifold):
    assert "ambient_dim=3" in repr(manifold)
    assert "intrinsic_dim=2" in repr(manifold)

def test_sample_default_behavior(manifold):
    points = manifold.sample(2)
    assert points.shape == (2, 3)
    assert_allclose(points, [[0, 0, 0], [1, 0, 0]])
