import autoarray as aa


def test_repr():
    array = aa.Array2D.no_mask([[1, 2], [3, 4]], pixel_scales=1)
    assert repr(array) == "Array2D([1., 2., 3., 4.])"
