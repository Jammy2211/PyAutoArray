import pytest

import numpy as np

from autoarray.structures.triangles.coordinate_array import JAXCoordinateArrayTriangles as CoordinateArrayTriangles


@pytest.fixture
def one_triangle():
    return CoordinateArrayTriangles(
        coordinates=np.array([[0, 0]]),
        side_length=1.0,
    )


@pytest.fixture
def two_triangles():
    return CoordinateArrayTriangles(
        coordinates=np.array([[0, 0], [1, 0]]),
        side_length=1.0,
    )
