import autoarray as aa


def test_copy():
    array = aa.Array2D.no_mask(
        [[1.0, 2.0], [3.0, 4.0]],
        pixel_scales=1.0,
    )
    copied_array = array.copy()

    array[0] = 5.0

    assert array[0] == 5.0
    assert copied_array[0] == 1.0
