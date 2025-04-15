# core/types.py
from typing import TypeAlias, Union, Sequence
import numpy as np
import numpy.typing as npt
from scipy.sparse import csr_matrix, spmatrix

PointCloud: TypeAlias = npt.NDArray[np.float64]  # graph signals: rows are samples/time, columns are nodes
SparseMatrix: TypeAlias = csr_matrix
Matrix: TypeAlias = Union[npt.NDArray[np.float64], spmatrix]
EdgeIndices: TypeAlias = Sequence[tuple[int, int]]
