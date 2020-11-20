import numpy as np
import pytest

import autoarray as aa


class TestCoordinates:
    def test__central_pixel__depends_on_shape_pixel_scale_and_origin(self):

        central_pixel_coordinates = aa.util.geometry.central_pixel_coordinates_from(
            shape=(3,)
        )

        assert central_pixel_coordinates == (1,)

        central_pixel_coordinates = aa.util.geometry.central_pixel_coordinates_from(
            shape=(3, 3)
        )

        assert central_pixel_coordinates == (1, 1)

        central_pixel_coordinates = aa.util.geometry.central_pixel_coordinates_from(
            shape=(3, 3)
        )

        assert central_pixel_coordinates == (1, 1)

        central_pixel_coordinates = aa.util.geometry.central_pixel_coordinates_from(
            shape=(4, 4)
        )

        assert central_pixel_coordinates == (1.5, 1.5)

        central_pixel_coordinates = aa.util.geometry.central_pixel_coordinates_from(
            shape=(4, 4)
        )

        assert central_pixel_coordinates == (1.5, 1.5)

    def test__pixel_coordinates_2d_from(self):

        pixel_coordinates = aa.util.geometry.pixel_coordinates_2d_from(
            scaled_coordinates_2d=(1.0, -1.0), shape_2d=(2, 2), pixel_scales=(2.0, 2.0)
        )

        assert pixel_coordinates == (0, 0)

        pixel_coordinates = aa.util.geometry.pixel_coordinates_2d_from(
            scaled_coordinates_2d=(1.0, 1.0), shape_2d=(2, 2), pixel_scales=(2.0, 2.0)
        )

        assert pixel_coordinates == (0, 1)

        pixel_coordinates = aa.util.geometry.pixel_coordinates_2d_from(
            scaled_coordinates_2d=(-1.0, -1.0), shape_2d=(2, 2), pixel_scales=(2.0, 2.0)
        )

        assert pixel_coordinates == (1, 0)

        pixel_coordinates = aa.util.geometry.pixel_coordinates_2d_from(
            scaled_coordinates_2d=(-1.0, 1.0), shape_2d=(2, 2), pixel_scales=(2.0, 2.0)
        )

        assert pixel_coordinates == (1, 1)

        pixel_coordinates = aa.util.geometry.pixel_coordinates_2d_from(
            scaled_coordinates_2d=(3.0, -3.0), shape_2d=(3, 3), pixel_scales=(3.0, 3.0)
        )

        assert pixel_coordinates == (0, 0)

        pixel_coordinates = aa.util.geometry.pixel_coordinates_2d_from(
            scaled_coordinates_2d=(3.0, 0.0), shape_2d=(3, 3), pixel_scales=(3.0, 3.0)
        )

        assert pixel_coordinates == (0, 1)

        pixel_coordinates = aa.util.geometry.pixel_coordinates_2d_from(
            scaled_coordinates_2d=(3.0, 3.0), shape_2d=(3, 3), pixel_scales=(3.0, 3.0)
        )

        assert pixel_coordinates == (0, 2)

        pixel_coordinates = aa.util.geometry.pixel_coordinates_2d_from(
            scaled_coordinates_2d=(0.0, -3.0), shape_2d=(3, 3), pixel_scales=(3.0, 3.0)
        )

        assert pixel_coordinates == (1, 0)

        pixel_coordinates = aa.util.geometry.pixel_coordinates_2d_from(
            scaled_coordinates_2d=(0.0, 0.0), shape_2d=(3, 3), pixel_scales=(3.0, 3.0)
        )

        assert pixel_coordinates == (1, 1)

        pixel_coordinates = aa.util.geometry.pixel_coordinates_2d_from(
            scaled_coordinates_2d=(0.0, 3.0), shape_2d=(3, 3), pixel_scales=(3.0, 3.0)
        )

        assert pixel_coordinates == (1, 2)

        pixel_coordinates = aa.util.geometry.pixel_coordinates_2d_from(
            scaled_coordinates_2d=(-3.0, -3.0), shape_2d=(3, 3), pixel_scales=(3.0, 3.0)
        )

        assert pixel_coordinates == (2, 0)

        pixel_coordinates = aa.util.geometry.pixel_coordinates_2d_from(
            scaled_coordinates_2d=(-3.0, 0.0), shape_2d=(3, 3), pixel_scales=(3.0, 3.0)
        )

        assert pixel_coordinates == (2, 1)

        pixel_coordinates = aa.util.geometry.pixel_coordinates_2d_from(
            scaled_coordinates_2d=(-3.0, 3.0), shape_2d=(3, 3), pixel_scales=(3.0, 3.0)
        )

        assert pixel_coordinates == (2, 2)

    def test__pixel_coordinates_2d_from__scaled_are_pixel_corners(self):

        pixel_coordinates = aa.util.geometry.pixel_coordinates_2d_from(
            scaled_coordinates_2d=(1.99, -1.99),
            shape_2d=(2, 2),
            pixel_scales=(2.0, 2.0),
        )

        assert pixel_coordinates == (0, 0)

        pixel_coordinates = aa.util.geometry.pixel_coordinates_2d_from(
            scaled_coordinates_2d=(1.99, -0.01),
            shape_2d=(2, 2),
            pixel_scales=(2.0, 2.0),
        )

        assert pixel_coordinates == (0, 0)

        pixel_coordinates = aa.util.geometry.pixel_coordinates_2d_from(
            scaled_coordinates_2d=(0.01, -1.99),
            shape_2d=(2, 2),
            pixel_scales=(2.0, 2.0),
        )

        assert pixel_coordinates == (0, 0)

        pixel_coordinates = aa.util.geometry.pixel_coordinates_2d_from(
            scaled_coordinates_2d=(0.01, -0.01),
            shape_2d=(2, 2),
            pixel_scales=(2.0, 2.0),
        )

        assert pixel_coordinates == (0, 0)

        pixel_coordinates = aa.util.geometry.pixel_coordinates_2d_from(
            scaled_coordinates_2d=(2.01, 0.01), shape_2d=(2, 2), pixel_scales=(2.0, 2.0)
        )

        assert pixel_coordinates == (0, 1)

        pixel_coordinates = aa.util.geometry.pixel_coordinates_2d_from(
            scaled_coordinates_2d=(2.01, 1.99), shape_2d=(2, 2), pixel_scales=(2.0, 2.0)
        )

        assert pixel_coordinates == (0, 1)

        pixel_coordinates = aa.util.geometry.pixel_coordinates_2d_from(
            scaled_coordinates_2d=(0.01, 0.01), shape_2d=(2, 2), pixel_scales=(2.0, 2.0)
        )

        assert pixel_coordinates == (0, 1)

        pixel_coordinates = aa.util.geometry.pixel_coordinates_2d_from(
            scaled_coordinates_2d=(0.01, 1.99), shape_2d=(2, 2), pixel_scales=(2.0, 2.0)
        )

        assert pixel_coordinates == (0, 1)

        pixel_coordinates = aa.util.geometry.pixel_coordinates_2d_from(
            scaled_coordinates_2d=(-0.01, -1.99),
            shape_2d=(2, 2),
            pixel_scales=(2.0, 2.0),
        )

        assert pixel_coordinates == (1, 0)

        pixel_coordinates = aa.util.geometry.pixel_coordinates_2d_from(
            scaled_coordinates_2d=(-0.01, -0.01),
            shape_2d=(2, 2),
            pixel_scales=(2.0, 2.0),
        )

        assert pixel_coordinates == (1, 0)
        pixel_coordinates = aa.util.geometry.pixel_coordinates_2d_from(
            scaled_coordinates_2d=(-1.99, -1.99),
            shape_2d=(2, 2),
            pixel_scales=(2.0, 2.0),
        )

        assert pixel_coordinates == (1, 0)

        pixel_coordinates = aa.util.geometry.pixel_coordinates_2d_from(
            scaled_coordinates_2d=(-1.99, -0.01),
            shape_2d=(2, 2),
            pixel_scales=(2.0, 2.0),
        )

        assert pixel_coordinates == (1, 0)

        pixel_coordinates = aa.util.geometry.pixel_coordinates_2d_from(
            scaled_coordinates_2d=(-0.01, 0.01),
            shape_2d=(2, 2),
            pixel_scales=(2.0, 2.0),
        )

        assert pixel_coordinates == (1, 1)

        pixel_coordinates = aa.util.geometry.pixel_coordinates_2d_from(
            scaled_coordinates_2d=(-0.01, 1.99),
            shape_2d=(2, 2),
            pixel_scales=(2.0, 2.0),
        )

        assert pixel_coordinates == (1, 1)

        pixel_coordinates = aa.util.geometry.pixel_coordinates_2d_from(
            scaled_coordinates_2d=(-1.99, 0.01),
            shape_2d=(2, 2),
            pixel_scales=(2.0, 2.0),
        )

        assert pixel_coordinates == (1, 1)

        pixel_coordinates = aa.util.geometry.pixel_coordinates_2d_from(
            scaled_coordinates_2d=(-1.99, 1.99),
            shape_2d=(2, 2),
            pixel_scales=(2.0, 2.0),
        )

        assert pixel_coordinates == (1, 1)

    def test__pixel_coordinates_2d_from___scaled_are_pixel_centres__nonzero_centre(
        self,
    ):

        pixel_coordinates = aa.util.geometry.pixel_coordinates_2d_from(
            scaled_coordinates_2d=(2.0, 0.0),
            shape_2d=(2, 2),
            pixel_scales=(2.0, 2.0),
            origins=(1.0, 1.0),
        )

        assert pixel_coordinates == (0, 0)

        pixel_coordinates = aa.util.geometry.pixel_coordinates_2d_from(
            scaled_coordinates_2d=(2.0, 2.0),
            shape_2d=(2, 2),
            pixel_scales=(2.0, 2.0),
            origins=(1.0, 1.0),
        )

        assert pixel_coordinates == (0, 1)

        pixel_coordinates = aa.util.geometry.pixel_coordinates_2d_from(
            scaled_coordinates_2d=(0.0, 0.0),
            shape_2d=(2, 2),
            pixel_scales=(2.0, 2.0),
            origins=(1.0, 1.0),
        )

        assert pixel_coordinates == (1, 0)

        pixel_coordinates = aa.util.geometry.pixel_coordinates_2d_from(
            scaled_coordinates_2d=(0.0, 2.0),
            shape_2d=(2, 2),
            pixel_scales=(2.0, 2.0),
            origins=(1.0, 1.0),
        )

        assert pixel_coordinates == (1, 1)

        pixel_coordinates = aa.util.geometry.pixel_coordinates_2d_from(
            scaled_coordinates_2d=(6.0, 0.0),
            shape_2d=(3, 3),
            pixel_scales=(3.0, 3.0),
            origins=(3.0, 3.0),
        )

        assert pixel_coordinates == (0, 0)

        pixel_coordinates = aa.util.geometry.pixel_coordinates_2d_from(
            scaled_coordinates_2d=(6.0, 3.0),
            shape_2d=(3, 3),
            pixel_scales=(3.0, 3.0),
            origins=(3.0, 3.0),
        )

        assert pixel_coordinates == (0, 1)

        pixel_coordinates = aa.util.geometry.pixel_coordinates_2d_from(
            scaled_coordinates_2d=(6.0, 6.0),
            shape_2d=(3, 3),
            pixel_scales=(3.0, 3.0),
            origins=(3.0, 3.0),
        )

        assert pixel_coordinates == (0, 2)

        pixel_coordinates = aa.util.geometry.pixel_coordinates_2d_from(
            scaled_coordinates_2d=(3.0, 0.0),
            shape_2d=(3, 3),
            pixel_scales=(3.0, 3.0),
            origins=(3.0, 3.0),
        )

        assert pixel_coordinates == (1, 0)

        pixel_coordinates = aa.util.geometry.pixel_coordinates_2d_from(
            scaled_coordinates_2d=(3.0, 3.0),
            shape_2d=(3, 3),
            pixel_scales=(3.0, 3.0),
            origins=(3.0, 3.0),
        )

        assert pixel_coordinates == (1, 1)

        pixel_coordinates = aa.util.geometry.pixel_coordinates_2d_from(
            scaled_coordinates_2d=(3.0, 6.0),
            shape_2d=(3, 3),
            pixel_scales=(3.0, 3.0),
            origins=(3.0, 3.0),
        )

        assert pixel_coordinates == (1, 2)

        pixel_coordinates = aa.util.geometry.pixel_coordinates_2d_from(
            scaled_coordinates_2d=(0.0, 0.0),
            shape_2d=(3, 3),
            pixel_scales=(3.0, 3.0),
            origins=(3.0, 3.0),
        )

        assert pixel_coordinates == (2, 0)

        pixel_coordinates = aa.util.geometry.pixel_coordinates_2d_from(
            scaled_coordinates_2d=(0.0, 3.0),
            shape_2d=(3, 3),
            pixel_scales=(3.0, 3.0),
            origins=(3.0, 3.0),
        )

        assert pixel_coordinates == (2, 1)

        pixel_coordinates = aa.util.geometry.pixel_coordinates_2d_from(
            scaled_coordinates_2d=(0.0, 6.0),
            shape_2d=(3, 3),
            pixel_scales=(3.0, 3.0),
            origins=(3.0, 3.0),
        )

        assert pixel_coordinates == (2, 2)

    def test__pixel_coordinates_2d_from__scaled_are_pixel_corners__nonzero_centre(
        self,
    ):

        pixel_coordinates = aa.util.geometry.pixel_coordinates_2d_from(
            scaled_coordinates_2d=(2.99, -0.99),
            shape_2d=(2, 2),
            pixel_scales=(2.0, 2.0),
            origins=(1.0, 1.0),
        )

        assert pixel_coordinates == (0, 0)

        pixel_coordinates = aa.util.geometry.pixel_coordinates_2d_from(
            scaled_coordinates_2d=(2.99, 0.99),
            shape_2d=(2, 2),
            pixel_scales=(2.0, 2.0),
            origins=(1.0, 1.0),
        )

        assert pixel_coordinates == (0, 0)

        pixel_coordinates = aa.util.geometry.pixel_coordinates_2d_from(
            scaled_coordinates_2d=(1.01, -0.99),
            shape_2d=(2, 2),
            pixel_scales=(2.0, 2.0),
            origins=(1.0, 1.0),
        )

        assert pixel_coordinates == (0, 0)

        pixel_coordinates = aa.util.geometry.pixel_coordinates_2d_from(
            scaled_coordinates_2d=(1.01, 0.99),
            shape_2d=(2, 2),
            pixel_scales=(2.0, 2.0),
            origins=(1.0, 1.0),
        )

        assert pixel_coordinates == (0, 0)

        pixel_coordinates = aa.util.geometry.pixel_coordinates_2d_from(
            scaled_coordinates_2d=(3.01, 1.01),
            shape_2d=(2, 2),
            pixel_scales=(2.0, 2.0),
            origins=(1.0, 1.0),
        )

        assert pixel_coordinates == (0, 1)

        pixel_coordinates = aa.util.geometry.pixel_coordinates_2d_from(
            scaled_coordinates_2d=(3.01, 2.99),
            shape_2d=(2, 2),
            pixel_scales=(2.0, 2.0),
            origins=(1.0, 1.0),
        )

        assert pixel_coordinates == (0, 1)

        pixel_coordinates = aa.util.geometry.pixel_coordinates_2d_from(
            scaled_coordinates_2d=(1.01, 1.01),
            shape_2d=(2, 2),
            pixel_scales=(2.0, 2.0),
            origins=(1.0, 1.0),
        )

        assert pixel_coordinates == (0, 1)

        pixel_coordinates = aa.util.geometry.pixel_coordinates_2d_from(
            scaled_coordinates_2d=(1.01, 2.99),
            shape_2d=(2, 2),
            pixel_scales=(2.0, 2.0),
            origins=(1.0, 1.0),
        )

        assert pixel_coordinates == (0, 1)

        pixel_coordinates = aa.util.geometry.pixel_coordinates_2d_from(
            scaled_coordinates_2d=(0.99, -0.99),
            shape_2d=(2, 2),
            pixel_scales=(2.0, 2.0),
            origins=(1.0, 1.0),
        )

        assert pixel_coordinates == (1, 0)

        pixel_coordinates = aa.util.geometry.pixel_coordinates_2d_from(
            scaled_coordinates_2d=(0.99, 0.99),
            shape_2d=(2, 2),
            pixel_scales=(2.0, 2.0),
            origins=(1.0, 1.0),
        )

        assert pixel_coordinates == (1, 0)

        pixel_coordinates = aa.util.geometry.pixel_coordinates_2d_from(
            scaled_coordinates_2d=(-0.99, -0.99),
            shape_2d=(2, 2),
            pixel_scales=(2.0, 2.0),
            origins=(1.0, 1.0),
        )

        assert pixel_coordinates == (1, 0)

        pixel_coordinates = aa.util.geometry.pixel_coordinates_2d_from(
            scaled_coordinates_2d=(-0.99, 0.99),
            shape_2d=(2, 2),
            pixel_scales=(2.0, 2.0),
            origins=(1.0, 1.0),
        )

        assert pixel_coordinates == (1, 0)

        pixel_coordinates = aa.util.geometry.pixel_coordinates_2d_from(
            scaled_coordinates_2d=(0.99, 1.01),
            shape_2d=(2, 2),
            pixel_scales=(2.0, 2.0),
            origins=(1.0, 1.0),
        )

        assert pixel_coordinates == (1, 1)

        pixel_coordinates = aa.util.geometry.pixel_coordinates_2d_from(
            scaled_coordinates_2d=(0.99, 2.99),
            shape_2d=(2, 2),
            pixel_scales=(2.0, 2.0),
            origins=(1.0, 1.0),
        )

        assert pixel_coordinates == (1, 1)

        pixel_coordinates = aa.util.geometry.pixel_coordinates_2d_from(
            scaled_coordinates_2d=(-0.99, 1.01),
            shape_2d=(2, 2),
            pixel_scales=(2.0, 2.0),
            origins=(1.0, 1.0),
        )

        assert pixel_coordinates == (1, 1)

        pixel_coordinates = aa.util.geometry.pixel_coordinates_2d_from(
            scaled_coordinates_2d=(-0.99, 2.99),
            shape_2d=(2, 2),
            pixel_scales=(2.0, 2.0),
            origins=(1.0, 1.0),
        )

        assert pixel_coordinates == (1, 1)

    def test__scaled_coordinates_2d_from___scaled_are_pixel_centres__nonzero_centre(
        self,
    ):

        scaled_coordinates = aa.util.geometry.scaled_coordinates_2d_from(
            pixel_coordinates_2d=(0, 0), shape_2d=(3, 3), pixel_scales=(3.0, 3.0)
        )

        assert scaled_coordinates == (3.0, -3.0)

        scaled_coordinates = aa.util.geometry.scaled_coordinates_2d_from(
            pixel_coordinates_2d=(0, 1), shape_2d=(3, 3), pixel_scales=(3.0, 3.0)
        )

        assert scaled_coordinates == (3.0, 0.0)

        scaled_coordinates = aa.util.geometry.scaled_coordinates_2d_from(
            pixel_coordinates_2d=(0, 2), shape_2d=(3, 3), pixel_scales=(3.0, 3.0)
        )

        assert scaled_coordinates == (3.0, 3.0)

        scaled_coordinates = aa.util.geometry.scaled_coordinates_2d_from(
            pixel_coordinates_2d=(1, 0), shape_2d=(3, 3), pixel_scales=(3.0, 3.0)
        )

        assert scaled_coordinates == (0.0, -3.0)

        scaled_coordinates = aa.util.geometry.scaled_coordinates_2d_from(
            pixel_coordinates_2d=(1, 1), shape_2d=(3, 3), pixel_scales=(3.0, 3.0)
        )

        assert scaled_coordinates == (0.0, 0.0)

        scaled_coordinates = aa.util.geometry.scaled_coordinates_2d_from(
            pixel_coordinates_2d=(1, 2), shape_2d=(3, 3), pixel_scales=(3.0, 3.0)
        )

        assert scaled_coordinates == (0.0, 3.0)

        scaled_coordinates = aa.util.geometry.scaled_coordinates_2d_from(
            pixel_coordinates_2d=(2, 0), shape_2d=(3, 3), pixel_scales=(3.0, 3.0)
        )

        assert scaled_coordinates == (-3.0, -3.0)

        scaled_coordinates = aa.util.geometry.scaled_coordinates_2d_from(
            pixel_coordinates_2d=(2, 1), shape_2d=(3, 3), pixel_scales=(3.0, 3.0)
        )

        assert scaled_coordinates == (-3.0, 0.0)

        scaled_coordinates = aa.util.geometry.scaled_coordinates_2d_from(
            pixel_coordinates_2d=(2, 2), shape_2d=(3, 3), pixel_scales=(3.0, 3.0)
        )

        assert scaled_coordinates == (-3.0, 3.0)

        scaled_coordinates = aa.util.geometry.scaled_coordinates_2d_from(
            pixel_coordinates_2d=(0, 0),
            shape_2d=(2, 2),
            pixel_scales=(2.0, 2.0),
            origins=(1.0, 1.0),
        )

        assert scaled_coordinates == (2.0, 0.0)

        scaled_coordinates = aa.util.geometry.scaled_coordinates_2d_from(
            pixel_coordinates_2d=(0, 1),
            shape_2d=(2, 2),
            pixel_scales=(2.0, 2.0),
            origins=(1.0, 1.0),
        )

        assert scaled_coordinates == (2.0, 2.0)

        scaled_coordinates = aa.util.geometry.scaled_coordinates_2d_from(
            pixel_coordinates_2d=(1, 0),
            shape_2d=(2, 2),
            pixel_scales=(2.0, 2.0),
            origins=(1.0, 1.0),
        )

        assert scaled_coordinates == (0.0, 0.0)

        scaled_coordinates = aa.util.geometry.scaled_coordinates_2d_from(
            pixel_coordinates_2d=(1, 1),
            shape_2d=(2, 2),
            pixel_scales=(2.0, 2.0),
            origins=(1.0, 1.0),
        )

        assert scaled_coordinates == (0.0, 2.0)

        scaled_coordinates = aa.util.geometry.scaled_coordinates_2d_from(
            pixel_coordinates_2d=(0, 0),
            shape_2d=(3, 3),
            pixel_scales=(3.0, 3.0),
            origins=(3.0, 3.0),
        )

        assert scaled_coordinates == (6.0, 0.0)

        scaled_coordinates = aa.util.geometry.scaled_coordinates_2d_from(
            pixel_coordinates_2d=(0, 1),
            shape_2d=(3, 3),
            pixel_scales=(3.0, 3.0),
            origins=(3.0, 3.0),
        )

        assert scaled_coordinates == (6.0, 3.0)

        scaled_coordinates = aa.util.geometry.scaled_coordinates_2d_from(
            pixel_coordinates_2d=(0, 2),
            shape_2d=(3, 3),
            pixel_scales=(3.0, 3.0),
            origins=(3.0, 3.0),
        )

        assert scaled_coordinates == (6.0, 6.0)

        scaled_coordinates = aa.util.geometry.scaled_coordinates_2d_from(
            pixel_coordinates_2d=(1, 0),
            shape_2d=(3, 3),
            pixel_scales=(3.0, 3.0),
            origins=(3.0, 3.0),
        )

        assert scaled_coordinates == (3.0, 0.0)

        scaled_coordinates = aa.util.geometry.scaled_coordinates_2d_from(
            pixel_coordinates_2d=(1, 1),
            shape_2d=(3, 3),
            pixel_scales=(3.0, 3.0),
            origins=(3.0, 3.0),
        )

        assert scaled_coordinates == (3.0, 3.0)

        scaled_coordinates = aa.util.geometry.scaled_coordinates_2d_from(
            pixel_coordinates_2d=(1, 2),
            shape_2d=(3, 3),
            pixel_scales=(3.0, 3.0),
            origins=(3.0, 3.0),
        )

        assert scaled_coordinates == (3.0, 6.0)

        scaled_coordinates = aa.util.geometry.scaled_coordinates_2d_from(
            pixel_coordinates_2d=(2, 0),
            shape_2d=(3, 3),
            pixel_scales=(3.0, 3.0),
            origins=(3.0, 3.0),
        )

        assert scaled_coordinates == (0.0, 0.0)

        scaled_coordinates = aa.util.geometry.scaled_coordinates_2d_from(
            pixel_coordinates_2d=(2, 1),
            shape_2d=(3, 3),
            pixel_scales=(3.0, 3.0),
            origins=(3.0, 3.0),
        )

        assert scaled_coordinates == (0.0, 3.0)

        scaled_coordinates = aa.util.geometry.scaled_coordinates_2d_from(
            pixel_coordinates_2d=(2, 2),
            shape_2d=(3, 3),
            pixel_scales=(3.0, 3.0),
            origins=(3.0, 3.0),
        )

        assert scaled_coordinates == (0.0, 6.0)
