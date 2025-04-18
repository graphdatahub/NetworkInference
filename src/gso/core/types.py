from collections.abc import Sequence
from typing import TypeAlias

import numpy as np
import numpy.typing as npt
from scipy.sparse import csr_matrix, spmatrix

PointCloud: TypeAlias = npt.NDArray[
    np.float64
]
GraphSignals: TypeAlias = npt.NDArray[
    np.float64
]
SparseMatrix: TypeAlias = csr_matrix
Matrix: TypeAlias = npt.NDArray[np.float64] | spmatrix
EdgeIndices: TypeAlias = Sequence[tuple[int, int]]
