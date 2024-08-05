from autoarray.numpy_wrapper import numpy as np
import pytest


@pytest.fixture
def compare_with_nans():
    def compare_with_nans_(arr1, arr2):
        nan_mask1 = np.isnan(arr1)
        nan_mask2 = np.isnan(arr2)

        equal_elements = np.where(
            nan_mask1 & nan_mask2,
            True,
            arr1 == arr2,
        )

        return np.all(equal_elements)

    return compare_with_nans_
