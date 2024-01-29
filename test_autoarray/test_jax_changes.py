import autoarray as aa
import pytest


@pytest.fixture(name="array")
def make_array():
    return aa.Array2D.no_mask(
        [[1.0, 2.0], [3.0, 4.0]],
        pixel_scales=1.0,
    )


def test_copy(array):
    copied_array = array.copy()

    array[0] = 5.0

    assert array[0] == 5.0
    assert copied_array[0] == 1.0


def test_in_place_multiply(array):
    array[0] *= 2.0

    assert array[0] == 2.0
