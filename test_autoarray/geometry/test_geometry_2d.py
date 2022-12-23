import os
from os import path
import numpy as np
import pytest
import shutil

import autoarray as aa
from autoarray import exc


def test__central_pixel_coordinates():

    mask = aa.Mask2D.unmasked(shape_native=(3, 3), pixel_scales=(0.1, 0.1))
    geometry = aa.Geometry2D(mask=mask)

    central_pixel_coordinates_util = aa.util.geometry.central_pixel_coordinates_2d_from(
        shape_native=(3, 3)
    )

    assert geometry.central_pixel_coordinates == central_pixel_coordinates_util

    mask = aa.Mask2D.unmasked(
        shape_native=(5, 3), pixel_scales=(2.0, 1.0), origin=(1.0, 2.0)
    )
    geometry = aa.Geometry2D(mask=mask)

    central_pixel_coordinates_util = aa.util.geometry.central_pixel_coordinates_2d_from(
        shape_native=(5, 3)
    )

    assert geometry.central_pixel_coordinates == central_pixel_coordinates_util