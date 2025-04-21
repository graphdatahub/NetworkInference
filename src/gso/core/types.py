# TODO:
# - Add shape features for Pyright, using TypeVarTuple and Unpack
#   for example, something along those lines:
#
#       from typing import TypeVarTuple, Unpack
#       import numpy as np
#
#       Shape = TypeVarTuple("Shape")
#       DType = TypeVar("DType", bound=np.generic)
#
#       class NDArray(np.ndarray[Unpack[Shape], DType]): ...
#
#       Vector = NDArray[Tuple[T1], np.float64]
#
#       Matrix = NDArray[Tuple[T1, T2], np.float64]

from collections.abc import Sequence
from typing import Annotated, TypeAlias

from numpy import float64
from numpy.typing import NDArray
from scipy.sparse import csr_matrix, spmatrix

Point: TypeAlias = Annotated[NDArray[float64], "Shape: (ambient_dim,)"]
PointCloud: TypeAlias = Annotated[NDArray[float64], "Shape: (n_points, ambient_dim)"]
Signal: TypeAlias = Annotated[NDArray[float64], "Shape: (n_nodes,)"]
Collection: TypeAlias = Annotated[NDArray[float64], "Shape: (n_signals, n_nodes)"]
Vector: TypeAlias = Annotated[NDArray[float64], "Dimension: 1"]
Matrix: TypeAlias = Annotated[NDArray[float64] | spmatrix, "Dimension: 2"]
SparseMatrix: TypeAlias = csr_matrix
EdgeIndices: TypeAlias = Sequence[tuple[int, int]]
