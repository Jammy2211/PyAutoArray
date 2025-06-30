import numpy as np
import pytest
import autoarray as aa


@pytest.mark.parametrize(
    "mask, expected_centre, expected_offset_pixels, expected_shape_native",
    [
        (
            aa.Mask2D.all_false(shape_native=(4, 6), pixel_scales=(1.0, 1.0)),
            (1.5, 2.5),
            (0, 0),
            (6, 6),
        ),
        (
            aa.Mask2D.all_false(shape_native=(6, 4), pixel_scales=(1.0, 1.0)),
            (2.5, 1.5),
            (0, 0),
            (6, 6),
        ),
        (
            aa.Mask2D(
                mask=np.array(
                    [[True, True, True], [True, True, False], [True, True, True]]
                ),
                pixel_scales=(1.0, 1.0),
            ),
            (1.0, 2.0),
            (0, 1),
            (1, 1),
        ),
        (
            aa.Mask2D(
                mask=np.array(
                    [[True, True, True], [True, True, True], [True, False, True]]
                ),
                pixel_scales=(1.0, 1.0),
            ),
            (2.0, 1.0),
            (1, 0),
            (1, 1),
        ),
        (
            aa.Mask2D(
                mask=np.array(
                    [[False, True, False], [True, True, True], [True, True, True]]
                ),
                pixel_scales=(1.0, 1.0),
            ),
            (0.0, 1.0),
            (-1, 0),
            (3, 3),
        ),
        (
            aa.Mask2D(
                mask=np.array(
                    [[False, False, True], [True, True, True], [True, True, True]]
                ),
                pixel_scales=(1.0, 1.0),
            ),
            (0.0, 0.5),
            (-1, -0.5),
            (1, 2),
        ),
        (
            aa.Mask2D(
                mask=np.array(
                    [
                        [True, True, True, True, True, True, True],
                        [True, True, True, True, True, True, True],
                        [True, True, True, True, True, True, False],
                    ]
                ),
                pixel_scales=(1.0, 1.0),
            ),
            (2.0, 6.0),
            (1, 3),
            (1, 1),
        ),
        (
            aa.Mask2D(
                mask=np.array(
                    [
                        [True, True, True],
                        [True, True, True],
                        [True, True, True],
                        [True, True, True],
                        [True, True, False],
                    ]
                ),
                pixel_scales=(1.0, 1.0),
            ),
            (4.0, 2.0),
            (2, 1),
            (1, 1),
        ),
        (
            aa.Mask2D(
                mask=np.array(
                    [
                        [True, True, True],
                        [True, True, True],
                        [True, True, True],
                        [True, True, True],
                        [True, True, True],
                        [True, True, True],
                        [True, True, False],
                    ]
                ),
                pixel_scales=(1.0, 1.0),
            ),
            (6.0, 2.0),
            (3, 1),
            (1, 1),
        ),
    ],
)
def test_zoom_quantities(
    mask, expected_centre, expected_offset_pixels, expected_shape_native
):
    zoom = aa.Zoom2D(mask=mask)
    assert zoom.centre == expected_centre
    assert zoom.offset_pixels == expected_offset_pixels
    assert zoom.shape_native == expected_shape_native


@pytest.mark.parametrize(
    "mask_array, values, expected_native",
    [
        (
            [
                [True, True, True, True],
                [True, False, False, True],
                [True, False, False, True],
                [True, True, True, True],
            ],
            [
                [1.0, 2.0, 3.0, 4.0],
                [5.0, 6.0, 7.0, 8.0],
                [9.0, 10.0, 11.0, 12.0],
                [13.0, 14.0, 15.0, 16.0],
            ],
            [[6.0, 7.0], [10.0, 11.0]],
        ),
        (
            [
                [True, True, True, True],
                [True, False, False, True],
                [False, False, False, True],
                [True, True, True, True],
            ],
            [
                [1.0, 2.0, 3.0, 4.0],
                [5.0, 6.0, 7.0, 8.0],
                [9.0, 10.0, 11.0, 12.0],
                [13.0, 14.0, 15.0, 16.0],
            ],
            [[0.0, 6.0, 7.0], [9.0, 10.0, 11.0]],
        ),
        (
            [
                [True, False, True, True],
                [True, False, False, True],
                [True, False, False, True],
                [True, True, True, True],
            ],
            [
                [1.0, 2.0, 3.0, 4.0],
                [5.0, 6.0, 7.0, 8.0],
                [9.0, 10.0, 11.0, 12.0],
                [13.0, 14.0, 15.0, 16.0],
            ],
            [[2.0, 0.0], [6.0, 7.0], [10.0, 11.0]],
        ),
    ],
)
def test_zoom_array_native(mask_array, values, expected_native):
    mask = aa.Mask2D(mask=mask_array, pixel_scales=(1.0, 1.0))
    arr = aa.Array2D(values=values, mask=mask)
    zoom = aa.Zoom2D(mask=mask)
    arr_zoomed = zoom.array_2d_from(array=arr, buffer=0)
    assert np.array_equal(arr_zoomed.native, np.array(expected_native))


@pytest.mark.parametrize(
    "mask_array, shape, expected_origin",
    [
        (
            [
                [True, True, True, True],
                [True, False, False, True],
                [True, False, False, True],
                [True, True, True, True],
            ],
            (4, 4),
            (0.0, 0.0),
        ),
        (
            [
                [True, True, True, True, True, True],
                [True, True, True, True, True, True],
                [True, True, True, False, False, True],
                [True, True, True, False, False, True],
                [True, True, True, True, True, True],
                [True, True, True, True, True, True],
            ],
            (6, 6),
            (0.0, 1.0),
        ),
    ],
)
def test_zoom_array_mask_origin(mask_array, shape, expected_origin):
    mask = aa.Mask2D(mask=mask_array, pixel_scales=(1.0, 1.0))
    arr = aa.Array2D(values=np.ones(shape=shape), mask=mask)
    zoom = aa.Zoom2D(mask=mask)
    arr_zoomed = zoom.array_2d_from(array=arr, buffer=0)
    assert arr_zoomed.mask.origin == expected_origin
