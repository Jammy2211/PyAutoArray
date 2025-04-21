import numpy as np

import autoarray as aa


def test__quantities():

    mask = aa.Mask2D.all_false(shape_native=(4, 6), pixel_scales=(1.0, 1.0))
    zoom = aa.Zoom2D(mask=mask)
    
    assert zoom.centre == (1.5, 2.5)
    assert zoom.offset_pixels == (0, 0)
    assert zoom.shape_native == (6, 6)

    mask = aa.Mask2D.all_false(shape_native=(6, 4), pixel_scales=(1.0, 1.0))
    zoom = aa.Zoom2D(mask=mask)
    
    assert zoom.centre == (2.5, 1.5)
    assert zoom.offset_pixels == (0, 0)
    assert zoom.shape_native == (6, 6)
  
    mask = aa.Mask2D(
        mask=np.array([[True, True, True], [True, True, False], [True, True, True]]),
        pixel_scales=(1.0, 1.0),
    )
    zoom = aa.Zoom2D(mask=mask)
    
    assert zoom.centre == (1, 2)
    assert zoom.offset_pixels == (0, 1)
    assert zoom.shape_native == (1, 1)

    mask = aa.Mask2D(
        mask=np.array([[True, True, True], [True, True, True], [True, False, True]]),
        pixel_scales=(1.0, 1.0),
    )
    zoom = aa.Zoom2D(mask=mask)
    
    assert zoom.centre == (2, 1)
    assert zoom.offset_pixels == (1, 0)
    assert zoom.shape_native == (1, 1)

    mask = aa.Mask2D(
        mask=np.array([[False, True, False], [True, True, True], [True, True, True]]),
        pixel_scales=(1.0, 1.0),
    )
    zoom = aa.Zoom2D(mask=mask)
    
    assert zoom.centre == (0, 1)
    assert zoom.offset_pixels == (-1, 0)
    assert zoom.shape_native == (3, 3)

    mask = aa.Mask2D(
        mask=np.array([[False, False, True], [True, True, True], [True, True, True]]),
        pixel_scales=(1.0, 1.0),
    )
    zoom = aa.Zoom2D(mask=mask)
    
    assert zoom.centre == (0, 0.5)
    assert zoom.offset_pixels == (-1, -0.5)
    assert zoom.shape_native == (1, 2)

    mask = aa.Mask2D(
        mask=np.array(
            [
                [True, True, True, True, True, True, True],
                [True, True, True, True, True, True, True],
                [True, True, True, True, True, True, False],
            ]
        ),
        pixel_scales=(1.0, 1.0),
    )
    zoom = aa.Zoom2D(mask=mask)

    assert zoom.centre == (2, 6)
    assert zoom.offset_pixels == (1, 3)

    mask = aa.Mask2D(
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
    )
    zoom = aa.Zoom2D(mask=mask)

    assert zoom.centre == (4, 2)
    assert zoom.offset_pixels == (2, 1)

    mask = aa.Mask2D(
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
    )
    zoom = aa.Zoom2D(mask=mask)

    assert zoom.centre == (6, 2)
    assert zoom.offset_pixels == (3, 1)
