from .base import (
    Manifold,
    AlgebraicManifold
)
from .algebraic import AlgebraicManifold as AlgebraicManifoldBis
from .visualize import plot_manifold

__all__ = [
    'Manifold',
    'AlgebraicManifold',
    'AlgebraicManifoldBis',
    'plot_manifold'
]
