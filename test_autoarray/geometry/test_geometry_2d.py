import os
from os import path
import numpy as np
import pytest
import shutil

import autoarray as aa
from autoarray import exc


def test__central_pixel_coordinates():

    geometry = aa.Geometry2D(shape_native=(3, 3), pixel_scales=(0.1, 0.1))

    central_pixel_coordinates_util = aa.util.geometry.central_pixel_coordinates_2d_from(
        shape_native=(3, 3)
    )

    assert geometry.central_pixel_coordinates == central_pixel_coordinates_util

    geometry = aa.Geometry2D(shape_native=(5, 3), pixel_scales=(2.0, 1.0))

    central_pixel_coordinates_util = aa.util.geometry.central_pixel_coordinates_2d_from(
        shape_native=(5, 3)
    )

    assert geometry.central_pixel_coordinates == central_pixel_coordinates_util


def test__pixel_coordinates_2d_from():

    geometry = aa.Geometry2D(
        shape_native=(6, 7), pixel_scales=(2.4, 1.8), origin=(1.0, 1.5)
    )

    pixel_coordinates_util = aa.util.geometry.pixel_coordinates_2d_from(
        scaled_coordinates_2d=(2.3, 1.2),
        shape_native=(6, 7),
        pixel_scales=(2.4, 1.8),
        origins=(1.0, 1.5),
    )

    assert (
        geometry.pixel_coordinates_2d_from(scaled_coordinates_2d=(2.3, 1.2))
        == pixel_coordinates_util
    )


def test__scaled_coordinates_2d_from():

    geometry = aa.Geometry2D(
        shape_native=(6, 7), pixel_scales=(2.4, 1.8), origin=(1.0, 1.5)
    )

    pixel_coordinates_util = aa.util.geometry.scaled_coordinates_2d_from(
        pixel_coordinates_2d=(5, 4),
        shape_native=(6, 7),
        pixel_scales=(2.4, 1.8),
        origins=(1.0, 1.5),
    )

    assert (
        geometry.scaled_coordinates_2d_from(pixel_coordinates_2d=(5, 4))
        == pixel_coordinates_util
    )


def test__grid_pixels_2d_slim_from():

    geometry = aa.Geometry2D(shape_native=(2, 2), pixel_scales=(2.0, 4.0))

    grid_scaled_2d = aa.Grid2D.manual_native(
        grid=[[[1.0, -2.0], [1.0, 2.0]], [[-1.0, -2.0], [-1.0, 2.0]]],
        pixel_scales=geometry.pixel_scales,
    )

    grid_pixels_util = aa.util.geometry.grid_pixels_2d_slim_from(
        grid_scaled_2d_slim=grid_scaled_2d,
        shape_native=(2, 2),
        pixel_scales=geometry.pixel_scales,
    )
    grid_pixels = geometry.grid_pixels_2d_from(grid_scaled_2d=grid_scaled_2d)

    assert (grid_pixels == grid_pixels_util).all()
    assert (grid_pixels.slim == grid_pixels_util).all()


def test__grid_pixel_centres_2d_from():

    geometry = aa.Geometry2D(shape_native=(2, 2), pixel_scales=(7.0, 2.0))

    grid_scaled_2d = aa.Grid2D.manual_native(
        grid=[[[1.0, -2.0], [1.0, 2.0]], [[-1.0, -2.0], [-1.0, 2.0]]],
        pixel_scales=geometry.pixel_scales,
    )

    grid_pixels_util = aa.util.geometry.grid_pixel_centres_2d_slim_from(
        grid_scaled_2d_slim=grid_scaled_2d,
        shape_native=(2, 2),
        pixel_scales=(7.0, 2.0),
    )

    grid_pixels = geometry.grid_pixel_centres_2d_from(grid_scaled_2d=grid_scaled_2d)

    assert (grid_pixels == grid_pixels_util).all()
