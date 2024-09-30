from autoarray.numpy_wrapper import numpy as np
from autoarray.structures.triangles.array import ArrayTriangles


import pytest


@pytest.fixture
def compare_with_nans():
    def compare_with_nans_(arr1, arr2):
        nan_mask1 = np.isnan(arr1)
        nan_mask2 = np.isnan(arr2)

        arr1 = arr1[~nan_mask1]
        arr2 = arr2[~nan_mask2]

        return np.all(arr1 == arr2)

    return compare_with_nans_


@pytest.fixture
def triangles():
    return ArrayTriangles(
        indices=np.array(
            [
                [0, 1, 2],
                [1, 2, 3],
            ]
        ),
        vertices=np.array(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
                [1.0, 1.0],
            ]
        ),
    )
