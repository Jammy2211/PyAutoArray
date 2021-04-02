import autoarray as aa
import numpy as np
import pytest


class TestCoordinates1D:
    def test__central_pixel__depends_on_shape_pixel_scale_and_origin(self):

        central_pixel_coordinates = aa.util.geometry.central_pixel_coordinates_1d_from(
            shape_slim=(3,)
        )

        assert central_pixel_coordinates == (1,)

        central_pixel_coordinates = aa.util.geometry.central_pixel_coordinates_1d_from(
            shape_slim=(4,)
        )

        assert central_pixel_coordinates == (1.5,)

    def test__pixel_coordinates_1d_from(self):

        pixel_coordinates = aa.util.geometry.pixel_coordinates_1d_from(
            scaled_coordinates_1d=(1.0,), shape_slim=(2,), pixel_scales=(2.0,)
        )

        assert pixel_coordinates == (1,)

        pixel_coordinates = aa.util.geometry.pixel_coordinates_1d_from(
            scaled_coordinates_1d=(1.0,), shape_slim=(2,), pixel_scales=(2.0,)
        )

        assert pixel_coordinates == (1,)

        pixel_coordinates = aa.util.geometry.pixel_coordinates_1d_from(
            scaled_coordinates_1d=(-1.0,), shape_slim=(2,), pixel_scales=(2.0,)
        )

        assert pixel_coordinates == (0,)

        pixel_coordinates = aa.util.geometry.pixel_coordinates_1d_from(
            scaled_coordinates_1d=(0.0,), shape_slim=(3,), pixel_scales=(3.0,)
        )

        assert pixel_coordinates == (1,)

        pixel_coordinates = aa.util.geometry.pixel_coordinates_1d_from(
            scaled_coordinates_1d=(3.0,), shape_slim=(3,), pixel_scales=(3.0,)
        )

        assert pixel_coordinates == (2,)

        pixel_coordinates = aa.util.geometry.pixel_coordinates_1d_from(
            scaled_coordinates_1d=(-3.0,), shape_slim=(3,), pixel_scales=(3.0,)
        )

        assert pixel_coordinates == (0,)

        pixel_coordinates = aa.util.geometry.pixel_coordinates_1d_from(
            scaled_coordinates_1d=(3.0,), shape_slim=(3,), pixel_scales=(3.0,)
        )

        assert pixel_coordinates == (2,)

    def test__pixel_coordinates_1d_from__scaled_are_pixel_corners(self):

        pixel_coordinates = aa.util.geometry.pixel_coordinates_1d_from(
            scaled_coordinates_1d=(-1.99,), shape_slim=(2,), pixel_scales=(2.0,)
        )

        assert pixel_coordinates == (0,)

        pixel_coordinates = aa.util.geometry.pixel_coordinates_1d_from(
            scaled_coordinates_1d=(-0.01,), shape_slim=(2,), pixel_scales=(2.0,)
        )

        assert pixel_coordinates == (0,)

        pixel_coordinates = aa.util.geometry.pixel_coordinates_1d_from(
            scaled_coordinates_1d=(0.01,), shape_slim=(2,), pixel_scales=(2.0,)
        )

        assert pixel_coordinates == (1,)

        pixel_coordinates = aa.util.geometry.pixel_coordinates_1d_from(
            scaled_coordinates_1d=(-1.99,), shape_slim=(2,), pixel_scales=(2.0,)
        )

        assert pixel_coordinates == (0,)

        pixel_coordinates = aa.util.geometry.pixel_coordinates_1d_from(
            scaled_coordinates_1d=(-0.01,), shape_slim=(2,), pixel_scales=(2.0,)
        )

        assert pixel_coordinates == (0,)

        pixel_coordinates = aa.util.geometry.pixel_coordinates_1d_from(
            scaled_coordinates_1d=(0.01,), shape_slim=(2,), pixel_scales=(2.0,)
        )

        assert pixel_coordinates == (1,)

        pixel_coordinates = aa.util.geometry.pixel_coordinates_1d_from(
            scaled_coordinates_1d=(1.99,), shape_slim=(2,), pixel_scales=(2.0,)
        )

        assert pixel_coordinates == (1,)

    def test__pixel_coordinates_1d_from___scaled_are_pixel_centres__nonzero_centre(
        self,
    ):
        pixel_coordinates = aa.util.geometry.pixel_coordinates_1d_from(
            scaled_coordinates_1d=(0.0,),
            shape_slim=(2,),
            pixel_scales=(2.0,),
            origins=(1.0,),
        )

        assert pixel_coordinates == (0,)

        pixel_coordinates = aa.util.geometry.pixel_coordinates_1d_from(
            scaled_coordinates_1d=(2.0,),
            shape_slim=(2,),
            pixel_scales=(2.0,),
            origins=(1.0,),
        )

        assert pixel_coordinates == (1,)

        pixel_coordinates = aa.util.geometry.pixel_coordinates_1d_from(
            scaled_coordinates_1d=(0.0,),
            shape_slim=(3,),
            pixel_scales=(3.0,),
            origins=(3.0,),
        )

        assert pixel_coordinates == (0,)

        pixel_coordinates = aa.util.geometry.pixel_coordinates_1d_from(
            scaled_coordinates_1d=(3.0,),
            shape_slim=(3,),
            pixel_scales=(3.0,),
            origins=(3.0,),
        )

        assert pixel_coordinates == (1,)

        pixel_coordinates = aa.util.geometry.pixel_coordinates_1d_from(
            scaled_coordinates_1d=(6.0,),
            shape_slim=(3,),
            pixel_scales=(3.0,),
            origins=(3.0,),
        )

        assert pixel_coordinates == (2,)

    def test__pixel_coordinates_1d_from__scaled_are_pixel_corners__nonzero_centre(
        self,
    ):
        pixel_coordinates = aa.util.geometry.pixel_coordinates_1d_from(
            scaled_coordinates_1d=(-0.99,),
            shape_slim=(2,),
            pixel_scales=(2.0,),
            origins=(1.0,),
        )

        assert pixel_coordinates == (0,)

        pixel_coordinates = aa.util.geometry.pixel_coordinates_1d_from(
            scaled_coordinates_1d=(0.99,),
            shape_slim=(2,),
            pixel_scales=(2.0,),
            origins=(1.0,),
        )

        assert pixel_coordinates == (0,)

        pixel_coordinates = aa.util.geometry.pixel_coordinates_1d_from(
            scaled_coordinates_1d=(1.01,),
            shape_slim=(2,),
            pixel_scales=(2.0,),
            origins=(1.0,),
        )

        assert pixel_coordinates == (1,)

        pixel_coordinates = aa.util.geometry.pixel_coordinates_1d_from(
            scaled_coordinates_1d=(2.99,),
            shape_slim=(2,),
            pixel_scales=(2.0,),
            origins=(1.0,),
        )

        assert pixel_coordinates == (1,)

        pixel_coordinates = aa.util.geometry.pixel_coordinates_1d_from(
            scaled_coordinates_1d=(1.01,),
            shape_slim=(2,),
            pixel_scales=(2.0,),
            origins=(1.0,),
        )

        assert pixel_coordinates == (1,)

        pixel_coordinates = aa.util.geometry.pixel_coordinates_1d_from(
            scaled_coordinates_1d=(2.99,),
            shape_slim=(2,),
            pixel_scales=(2.0,),
            origins=(1.0,),
        )

        assert pixel_coordinates == (1,)

    def test__scaled_coordinates_1d_from___scaled_are_pixel_centres__nonzero_centre(
        self,
    ):
        scaled_coordinates = aa.util.geometry.scaled_coordinates_1d_from(
            pixel_coordinates_1d=(0,), shape_slim=(3,), pixel_scales=(3.0,)
        )

        assert scaled_coordinates == (-3.0,)

        scaled_coordinates = aa.util.geometry.scaled_coordinates_1d_from(
            pixel_coordinates_1d=(1,), shape_slim=(3,), pixel_scales=(3.0,)
        )

        assert scaled_coordinates == (0.0,)

        scaled_coordinates = aa.util.geometry.scaled_coordinates_1d_from(
            pixel_coordinates_1d=(2,), shape_slim=(3,), pixel_scales=(3.0,)
        )

        assert scaled_coordinates == (3.0,)

        scaled_coordinates = aa.util.geometry.scaled_coordinates_1d_from(
            pixel_coordinates_1d=(0,),
            shape_slim=(2,),
            pixel_scales=(2.0,),
            origins=(1.0,),
        )

        assert scaled_coordinates == (0.0,)

        scaled_coordinates = aa.util.geometry.scaled_coordinates_1d_from(
            pixel_coordinates_1d=(1,),
            shape_slim=(2,),
            pixel_scales=(2.0,),
            origins=(1.0,),
        )

        assert scaled_coordinates == (2.0,)

        scaled_coordinates = aa.util.geometry.scaled_coordinates_1d_from(
            pixel_coordinates_1d=(0,),
            shape_slim=(3,),
            pixel_scales=(3.0,),
            origins=(3.0,),
        )

        assert scaled_coordinates == (0.0,)

        scaled_coordinates = aa.util.geometry.scaled_coordinates_1d_from(
            pixel_coordinates_1d=(1,),
            shape_slim=(3,),
            pixel_scales=(3.0,),
            origins=(3.0,),
        )

        assert scaled_coordinates == (3.0,)

        scaled_coordinates = aa.util.geometry.scaled_coordinates_1d_from(
            pixel_coordinates_1d=(2,),
            shape_slim=(3,),
            pixel_scales=(3.0,),
            origins=(3.0,),
        )

        assert scaled_coordinates == (6.0,)


class TestCoordinates2D:
    def test__central_pixel__depends_on_shape_pixel_scale_and_origin(self):

        central_pixel_coordinates = aa.util.geometry.central_pixel_coordinates_2d_from(
            shape_native=(3, 3)
        )

        assert central_pixel_coordinates == (1, 1)

        central_pixel_coordinates = aa.util.geometry.central_pixel_coordinates_2d_from(
            shape_native=(3, 3)
        )

        assert central_pixel_coordinates == (1, 1)

        central_pixel_coordinates = aa.util.geometry.central_pixel_coordinates_2d_from(
            shape_native=(4, 4)
        )

        assert central_pixel_coordinates == (1.5, 1.5)

        central_pixel_coordinates = aa.util.geometry.central_pixel_coordinates_2d_from(
            shape_native=(4, 4)
        )

        assert central_pixel_coordinates == (1.5, 1.5)

    def test__pixel_coordinates_2d_from(self):

        pixel_coordinates = aa.util.geometry.pixel_coordinates_2d_from(
            scaled_coordinates_2d=(1.0, -1.0),
            shape_native=(2, 2),
            pixel_scales=(2.0, 2.0),
        )

        assert pixel_coordinates == (0, 0)

        pixel_coordinates = aa.util.geometry.pixel_coordinates_2d_from(
            scaled_coordinates_2d=(1.0, 1.0),
            shape_native=(2, 2),
            pixel_scales=(2.0, 2.0),
        )

        assert pixel_coordinates == (0, 1)

        pixel_coordinates = aa.util.geometry.pixel_coordinates_2d_from(
            scaled_coordinates_2d=(-1.0, -1.0),
            shape_native=(2, 2),
            pixel_scales=(2.0, 2.0),
        )

        assert pixel_coordinates == (1, 0)

        pixel_coordinates = aa.util.geometry.pixel_coordinates_2d_from(
            scaled_coordinates_2d=(-1.0, 1.0),
            shape_native=(2, 2),
            pixel_scales=(2.0, 2.0),
        )

        assert pixel_coordinates == (1, 1)

        pixel_coordinates = aa.util.geometry.pixel_coordinates_2d_from(
            scaled_coordinates_2d=(3.0, -3.0),
            shape_native=(3, 3),
            pixel_scales=(3.0, 3.0),
        )

        assert pixel_coordinates == (0, 0)

        pixel_coordinates = aa.util.geometry.pixel_coordinates_2d_from(
            scaled_coordinates_2d=(3.0, 0.0),
            shape_native=(3, 3),
            pixel_scales=(3.0, 3.0),
        )

        assert pixel_coordinates == (0, 1)

        pixel_coordinates = aa.util.geometry.pixel_coordinates_2d_from(
            scaled_coordinates_2d=(3.0, 3.0),
            shape_native=(3, 3),
            pixel_scales=(3.0, 3.0),
        )

        assert pixel_coordinates == (0, 2)

        pixel_coordinates = aa.util.geometry.pixel_coordinates_2d_from(
            scaled_coordinates_2d=(0.0, -3.0),
            shape_native=(3, 3),
            pixel_scales=(3.0, 3.0),
        )

        assert pixel_coordinates == (1, 0)

        pixel_coordinates = aa.util.geometry.pixel_coordinates_2d_from(
            scaled_coordinates_2d=(0.0, 0.0),
            shape_native=(3, 3),
            pixel_scales=(3.0, 3.0),
        )

        assert pixel_coordinates == (1, 1)

        pixel_coordinates = aa.util.geometry.pixel_coordinates_2d_from(
            scaled_coordinates_2d=(0.0, 3.0),
            shape_native=(3, 3),
            pixel_scales=(3.0, 3.0),
        )

        assert pixel_coordinates == (1, 2)

        pixel_coordinates = aa.util.geometry.pixel_coordinates_2d_from(
            scaled_coordinates_2d=(-3.0, -3.0),
            shape_native=(3, 3),
            pixel_scales=(3.0, 3.0),
        )

        assert pixel_coordinates == (2, 0)

        pixel_coordinates = aa.util.geometry.pixel_coordinates_2d_from(
            scaled_coordinates_2d=(-3.0, 0.0),
            shape_native=(3, 3),
            pixel_scales=(3.0, 3.0),
        )

        assert pixel_coordinates == (2, 1)

        pixel_coordinates = aa.util.geometry.pixel_coordinates_2d_from(
            scaled_coordinates_2d=(-3.0, 3.0),
            shape_native=(3, 3),
            pixel_scales=(3.0, 3.0),
        )

        assert pixel_coordinates == (2, 2)

    def test__pixel_coordinates_2d_from__scaled_are_pixel_corners(self):

        pixel_coordinates = aa.util.geometry.pixel_coordinates_2d_from(
            scaled_coordinates_2d=(1.99, -1.99),
            shape_native=(2, 2),
            pixel_scales=(2.0, 2.0),
        )

        assert pixel_coordinates == (0, 0)

        pixel_coordinates = aa.util.geometry.pixel_coordinates_2d_from(
            scaled_coordinates_2d=(1.99, -0.01),
            shape_native=(2, 2),
            pixel_scales=(2.0, 2.0),
        )

        assert pixel_coordinates == (0, 0)

        pixel_coordinates = aa.util.geometry.pixel_coordinates_2d_from(
            scaled_coordinates_2d=(0.01, -1.99),
            shape_native=(2, 2),
            pixel_scales=(2.0, 2.0),
        )

        assert pixel_coordinates == (0, 0)

        pixel_coordinates = aa.util.geometry.pixel_coordinates_2d_from(
            scaled_coordinates_2d=(0.01, -0.01),
            shape_native=(2, 2),
            pixel_scales=(2.0, 2.0),
        )

        assert pixel_coordinates == (0, 0)

        pixel_coordinates = aa.util.geometry.pixel_coordinates_2d_from(
            scaled_coordinates_2d=(2.01, 0.01),
            shape_native=(2, 2),
            pixel_scales=(2.0, 2.0),
        )

        assert pixel_coordinates == (0, 1)

        pixel_coordinates = aa.util.geometry.pixel_coordinates_2d_from(
            scaled_coordinates_2d=(2.01, 1.99),
            shape_native=(2, 2),
            pixel_scales=(2.0, 2.0),
        )

        assert pixel_coordinates == (0, 1)

        pixel_coordinates = aa.util.geometry.pixel_coordinates_2d_from(
            scaled_coordinates_2d=(0.01, 0.01),
            shape_native=(2, 2),
            pixel_scales=(2.0, 2.0),
        )

        assert pixel_coordinates == (0, 1)

        pixel_coordinates = aa.util.geometry.pixel_coordinates_2d_from(
            scaled_coordinates_2d=(0.01, 1.99),
            shape_native=(2, 2),
            pixel_scales=(2.0, 2.0),
        )

        assert pixel_coordinates == (0, 1)

        pixel_coordinates = aa.util.geometry.pixel_coordinates_2d_from(
            scaled_coordinates_2d=(-0.01, -1.99),
            shape_native=(2, 2),
            pixel_scales=(2.0, 2.0),
        )

        assert pixel_coordinates == (1, 0)

        pixel_coordinates = aa.util.geometry.pixel_coordinates_2d_from(
            scaled_coordinates_2d=(-0.01, -0.01),
            shape_native=(2, 2),
            pixel_scales=(2.0, 2.0),
        )

        assert pixel_coordinates == (1, 0)
        pixel_coordinates = aa.util.geometry.pixel_coordinates_2d_from(
            scaled_coordinates_2d=(-1.99, -1.99),
            shape_native=(2, 2),
            pixel_scales=(2.0, 2.0),
        )

        assert pixel_coordinates == (1, 0)

        pixel_coordinates = aa.util.geometry.pixel_coordinates_2d_from(
            scaled_coordinates_2d=(-1.99, -0.01),
            shape_native=(2, 2),
            pixel_scales=(2.0, 2.0),
        )

        assert pixel_coordinates == (1, 0)

        pixel_coordinates = aa.util.geometry.pixel_coordinates_2d_from(
            scaled_coordinates_2d=(-0.01, 0.01),
            shape_native=(2, 2),
            pixel_scales=(2.0, 2.0),
        )

        assert pixel_coordinates == (1, 1)

        pixel_coordinates = aa.util.geometry.pixel_coordinates_2d_from(
            scaled_coordinates_2d=(-0.01, 1.99),
            shape_native=(2, 2),
            pixel_scales=(2.0, 2.0),
        )

        assert pixel_coordinates == (1, 1)

        pixel_coordinates = aa.util.geometry.pixel_coordinates_2d_from(
            scaled_coordinates_2d=(-1.99, 0.01),
            shape_native=(2, 2),
            pixel_scales=(2.0, 2.0),
        )

        assert pixel_coordinates == (1, 1)

        pixel_coordinates = aa.util.geometry.pixel_coordinates_2d_from(
            scaled_coordinates_2d=(-1.99, 1.99),
            shape_native=(2, 2),
            pixel_scales=(2.0, 2.0),
        )

        assert pixel_coordinates == (1, 1)

    def test__pixel_coordinates_2d_from___scaled_are_pixel_centres__nonzero_centre(
        self,
    ):

        pixel_coordinates = aa.util.geometry.pixel_coordinates_2d_from(
            scaled_coordinates_2d=(2.0, 0.0),
            shape_native=(2, 2),
            pixel_scales=(2.0, 2.0),
            origins=(1.0, 1.0),
        )

        assert pixel_coordinates == (0, 0)

        pixel_coordinates = aa.util.geometry.pixel_coordinates_2d_from(
            scaled_coordinates_2d=(2.0, 2.0),
            shape_native=(2, 2),
            pixel_scales=(2.0, 2.0),
            origins=(1.0, 1.0),
        )

        assert pixel_coordinates == (0, 1)

        pixel_coordinates = aa.util.geometry.pixel_coordinates_2d_from(
            scaled_coordinates_2d=(0.0, 0.0),
            shape_native=(2, 2),
            pixel_scales=(2.0, 2.0),
            origins=(1.0, 1.0),
        )

        assert pixel_coordinates == (1, 0)

        pixel_coordinates = aa.util.geometry.pixel_coordinates_2d_from(
            scaled_coordinates_2d=(0.0, 2.0),
            shape_native=(2, 2),
            pixel_scales=(2.0, 2.0),
            origins=(1.0, 1.0),
        )

        assert pixel_coordinates == (1, 1)

        pixel_coordinates = aa.util.geometry.pixel_coordinates_2d_from(
            scaled_coordinates_2d=(6.0, 0.0),
            shape_native=(3, 3),
            pixel_scales=(3.0, 3.0),
            origins=(3.0, 3.0),
        )

        assert pixel_coordinates == (0, 0)

        pixel_coordinates = aa.util.geometry.pixel_coordinates_2d_from(
            scaled_coordinates_2d=(6.0, 3.0),
            shape_native=(3, 3),
            pixel_scales=(3.0, 3.0),
            origins=(3.0, 3.0),
        )

        assert pixel_coordinates == (0, 1)

        pixel_coordinates = aa.util.geometry.pixel_coordinates_2d_from(
            scaled_coordinates_2d=(6.0, 6.0),
            shape_native=(3, 3),
            pixel_scales=(3.0, 3.0),
            origins=(3.0, 3.0),
        )

        assert pixel_coordinates == (0, 2)

        pixel_coordinates = aa.util.geometry.pixel_coordinates_2d_from(
            scaled_coordinates_2d=(3.0, 0.0),
            shape_native=(3, 3),
            pixel_scales=(3.0, 3.0),
            origins=(3.0, 3.0),
        )

        assert pixel_coordinates == (1, 0)

        pixel_coordinates = aa.util.geometry.pixel_coordinates_2d_from(
            scaled_coordinates_2d=(3.0, 3.0),
            shape_native=(3, 3),
            pixel_scales=(3.0, 3.0),
            origins=(3.0, 3.0),
        )

        assert pixel_coordinates == (1, 1)

        pixel_coordinates = aa.util.geometry.pixel_coordinates_2d_from(
            scaled_coordinates_2d=(3.0, 6.0),
            shape_native=(3, 3),
            pixel_scales=(3.0, 3.0),
            origins=(3.0, 3.0),
        )

        assert pixel_coordinates == (1, 2)

        pixel_coordinates = aa.util.geometry.pixel_coordinates_2d_from(
            scaled_coordinates_2d=(0.0, 0.0),
            shape_native=(3, 3),
            pixel_scales=(3.0, 3.0),
            origins=(3.0, 3.0),
        )

        assert pixel_coordinates == (2, 0)

        pixel_coordinates = aa.util.geometry.pixel_coordinates_2d_from(
            scaled_coordinates_2d=(0.0, 3.0),
            shape_native=(3, 3),
            pixel_scales=(3.0, 3.0),
            origins=(3.0, 3.0),
        )

        assert pixel_coordinates == (2, 1)

        pixel_coordinates = aa.util.geometry.pixel_coordinates_2d_from(
            scaled_coordinates_2d=(0.0, 6.0),
            shape_native=(3, 3),
            pixel_scales=(3.0, 3.0),
            origins=(3.0, 3.0),
        )

        assert pixel_coordinates == (2, 2)

    def test__pixel_coordinates_2d_from__scaled_are_pixel_corners__nonzero_centre(
        self,
    ):

        pixel_coordinates = aa.util.geometry.pixel_coordinates_2d_from(
            scaled_coordinates_2d=(2.99, -0.99),
            shape_native=(2, 2),
            pixel_scales=(2.0, 2.0),
            origins=(1.0, 1.0),
        )

        assert pixel_coordinates == (0, 0)

        pixel_coordinates = aa.util.geometry.pixel_coordinates_2d_from(
            scaled_coordinates_2d=(2.99, 0.99),
            shape_native=(2, 2),
            pixel_scales=(2.0, 2.0),
            origins=(1.0, 1.0),
        )

        assert pixel_coordinates == (0, 0)

        pixel_coordinates = aa.util.geometry.pixel_coordinates_2d_from(
            scaled_coordinates_2d=(1.01, -0.99),
            shape_native=(2, 2),
            pixel_scales=(2.0, 2.0),
            origins=(1.0, 1.0),
        )

        assert pixel_coordinates == (0, 0)

        pixel_coordinates = aa.util.geometry.pixel_coordinates_2d_from(
            scaled_coordinates_2d=(1.01, 0.99),
            shape_native=(2, 2),
            pixel_scales=(2.0, 2.0),
            origins=(1.0, 1.0),
        )

        assert pixel_coordinates == (0, 0)

        pixel_coordinates = aa.util.geometry.pixel_coordinates_2d_from(
            scaled_coordinates_2d=(3.01, 1.01),
            shape_native=(2, 2),
            pixel_scales=(2.0, 2.0),
            origins=(1.0, 1.0),
        )

        assert pixel_coordinates == (0, 1)

        pixel_coordinates = aa.util.geometry.pixel_coordinates_2d_from(
            scaled_coordinates_2d=(3.01, 2.99),
            shape_native=(2, 2),
            pixel_scales=(2.0, 2.0),
            origins=(1.0, 1.0),
        )

        assert pixel_coordinates == (0, 1)

        pixel_coordinates = aa.util.geometry.pixel_coordinates_2d_from(
            scaled_coordinates_2d=(1.01, 1.01),
            shape_native=(2, 2),
            pixel_scales=(2.0, 2.0),
            origins=(1.0, 1.0),
        )

        assert pixel_coordinates == (0, 1)

        pixel_coordinates = aa.util.geometry.pixel_coordinates_2d_from(
            scaled_coordinates_2d=(1.01, 2.99),
            shape_native=(2, 2),
            pixel_scales=(2.0, 2.0),
            origins=(1.0, 1.0),
        )

        assert pixel_coordinates == (0, 1)

        pixel_coordinates = aa.util.geometry.pixel_coordinates_2d_from(
            scaled_coordinates_2d=(0.99, -0.99),
            shape_native=(2, 2),
            pixel_scales=(2.0, 2.0),
            origins=(1.0, 1.0),
        )

        assert pixel_coordinates == (1, 0)

        pixel_coordinates = aa.util.geometry.pixel_coordinates_2d_from(
            scaled_coordinates_2d=(0.99, 0.99),
            shape_native=(2, 2),
            pixel_scales=(2.0, 2.0),
            origins=(1.0, 1.0),
        )

        assert pixel_coordinates == (1, 0)

        pixel_coordinates = aa.util.geometry.pixel_coordinates_2d_from(
            scaled_coordinates_2d=(-0.99, -0.99),
            shape_native=(2, 2),
            pixel_scales=(2.0, 2.0),
            origins=(1.0, 1.0),
        )

        assert pixel_coordinates == (1, 0)

        pixel_coordinates = aa.util.geometry.pixel_coordinates_2d_from(
            scaled_coordinates_2d=(-0.99, 0.99),
            shape_native=(2, 2),
            pixel_scales=(2.0, 2.0),
            origins=(1.0, 1.0),
        )

        assert pixel_coordinates == (1, 0)

        pixel_coordinates = aa.util.geometry.pixel_coordinates_2d_from(
            scaled_coordinates_2d=(0.99, 1.01),
            shape_native=(2, 2),
            pixel_scales=(2.0, 2.0),
            origins=(1.0, 1.0),
        )

        assert pixel_coordinates == (1, 1)

        pixel_coordinates = aa.util.geometry.pixel_coordinates_2d_from(
            scaled_coordinates_2d=(0.99, 2.99),
            shape_native=(2, 2),
            pixel_scales=(2.0, 2.0),
            origins=(1.0, 1.0),
        )

        assert pixel_coordinates == (1, 1)

        pixel_coordinates = aa.util.geometry.pixel_coordinates_2d_from(
            scaled_coordinates_2d=(-0.99, 1.01),
            shape_native=(2, 2),
            pixel_scales=(2.0, 2.0),
            origins=(1.0, 1.0),
        )

        assert pixel_coordinates == (1, 1)

        pixel_coordinates = aa.util.geometry.pixel_coordinates_2d_from(
            scaled_coordinates_2d=(-0.99, 2.99),
            shape_native=(2, 2),
            pixel_scales=(2.0, 2.0),
            origins=(1.0, 1.0),
        )

        assert pixel_coordinates == (1, 1)

    def test__scaled_coordinates_2d_from___scaled_are_pixel_centres__nonzero_centre(
        self,
    ):

        scaled_coordinates = aa.util.geometry.scaled_coordinates_2d_from(
            pixel_coordinates_2d=(0, 0), shape_native=(3, 3), pixel_scales=(3.0, 3.0)
        )

        assert scaled_coordinates == (3.0, -3.0)

        scaled_coordinates = aa.util.geometry.scaled_coordinates_2d_from(
            pixel_coordinates_2d=(0, 1), shape_native=(3, 3), pixel_scales=(3.0, 3.0)
        )

        assert scaled_coordinates == (3.0, 0.0)

        scaled_coordinates = aa.util.geometry.scaled_coordinates_2d_from(
            pixel_coordinates_2d=(0, 2), shape_native=(3, 3), pixel_scales=(3.0, 3.0)
        )

        assert scaled_coordinates == (3.0, 3.0)

        scaled_coordinates = aa.util.geometry.scaled_coordinates_2d_from(
            pixel_coordinates_2d=(1, 0), shape_native=(3, 3), pixel_scales=(3.0, 3.0)
        )

        assert scaled_coordinates == (0.0, -3.0)

        scaled_coordinates = aa.util.geometry.scaled_coordinates_2d_from(
            pixel_coordinates_2d=(1, 1), shape_native=(3, 3), pixel_scales=(3.0, 3.0)
        )

        assert scaled_coordinates == (0.0, 0.0)

        scaled_coordinates = aa.util.geometry.scaled_coordinates_2d_from(
            pixel_coordinates_2d=(1, 2), shape_native=(3, 3), pixel_scales=(3.0, 3.0)
        )

        assert scaled_coordinates == (0.0, 3.0)

        scaled_coordinates = aa.util.geometry.scaled_coordinates_2d_from(
            pixel_coordinates_2d=(2, 0), shape_native=(3, 3), pixel_scales=(3.0, 3.0)
        )

        assert scaled_coordinates == (-3.0, -3.0)

        scaled_coordinates = aa.util.geometry.scaled_coordinates_2d_from(
            pixel_coordinates_2d=(2, 1), shape_native=(3, 3), pixel_scales=(3.0, 3.0)
        )

        assert scaled_coordinates == (-3.0, 0.0)

        scaled_coordinates = aa.util.geometry.scaled_coordinates_2d_from(
            pixel_coordinates_2d=(2, 2), shape_native=(3, 3), pixel_scales=(3.0, 3.0)
        )

        assert scaled_coordinates == (-3.0, 3.0)

        scaled_coordinates = aa.util.geometry.scaled_coordinates_2d_from(
            pixel_coordinates_2d=(0, 0),
            shape_native=(2, 2),
            pixel_scales=(2.0, 2.0),
            origins=(1.0, 1.0),
        )

        assert scaled_coordinates == (2.0, 0.0)

        scaled_coordinates = aa.util.geometry.scaled_coordinates_2d_from(
            pixel_coordinates_2d=(0, 1),
            shape_native=(2, 2),
            pixel_scales=(2.0, 2.0),
            origins=(1.0, 1.0),
        )

        assert scaled_coordinates == (2.0, 2.0)

        scaled_coordinates = aa.util.geometry.scaled_coordinates_2d_from(
            pixel_coordinates_2d=(1, 0),
            shape_native=(2, 2),
            pixel_scales=(2.0, 2.0),
            origins=(1.0, 1.0),
        )

        assert scaled_coordinates == (0.0, 0.0)

        scaled_coordinates = aa.util.geometry.scaled_coordinates_2d_from(
            pixel_coordinates_2d=(1, 1),
            shape_native=(2, 2),
            pixel_scales=(2.0, 2.0),
            origins=(1.0, 1.0),
        )

        assert scaled_coordinates == (0.0, 2.0)

        scaled_coordinates = aa.util.geometry.scaled_coordinates_2d_from(
            pixel_coordinates_2d=(0, 0),
            shape_native=(3, 3),
            pixel_scales=(3.0, 3.0),
            origins=(3.0, 3.0),
        )

        assert scaled_coordinates == (6.0, 0.0)

        scaled_coordinates = aa.util.geometry.scaled_coordinates_2d_from(
            pixel_coordinates_2d=(0, 1),
            shape_native=(3, 3),
            pixel_scales=(3.0, 3.0),
            origins=(3.0, 3.0),
        )

        assert scaled_coordinates == (6.0, 3.0)

        scaled_coordinates = aa.util.geometry.scaled_coordinates_2d_from(
            pixel_coordinates_2d=(0, 2),
            shape_native=(3, 3),
            pixel_scales=(3.0, 3.0),
            origins=(3.0, 3.0),
        )

        assert scaled_coordinates == (6.0, 6.0)

        scaled_coordinates = aa.util.geometry.scaled_coordinates_2d_from(
            pixel_coordinates_2d=(1, 0),
            shape_native=(3, 3),
            pixel_scales=(3.0, 3.0),
            origins=(3.0, 3.0),
        )

        assert scaled_coordinates == (3.0, 0.0)

        scaled_coordinates = aa.util.geometry.scaled_coordinates_2d_from(
            pixel_coordinates_2d=(1, 1),
            shape_native=(3, 3),
            pixel_scales=(3.0, 3.0),
            origins=(3.0, 3.0),
        )

        assert scaled_coordinates == (3.0, 3.0)

        scaled_coordinates = aa.util.geometry.scaled_coordinates_2d_from(
            pixel_coordinates_2d=(1, 2),
            shape_native=(3, 3),
            pixel_scales=(3.0, 3.0),
            origins=(3.0, 3.0),
        )

        assert scaled_coordinates == (3.0, 6.0)

        scaled_coordinates = aa.util.geometry.scaled_coordinates_2d_from(
            pixel_coordinates_2d=(2, 0),
            shape_native=(3, 3),
            pixel_scales=(3.0, 3.0),
            origins=(3.0, 3.0),
        )

        assert scaled_coordinates == (0.0, 0.0)

        scaled_coordinates = aa.util.geometry.scaled_coordinates_2d_from(
            pixel_coordinates_2d=(2, 1),
            shape_native=(3, 3),
            pixel_scales=(3.0, 3.0),
            origins=(3.0, 3.0),
        )

        assert scaled_coordinates == (0.0, 3.0)

        scaled_coordinates = aa.util.geometry.scaled_coordinates_2d_from(
            pixel_coordinates_2d=(2, 2),
            shape_native=(3, 3),
            pixel_scales=(3.0, 3.0),
            origins=(3.0, 3.0),
        )

        assert scaled_coordinates == (0.0, 6.0)


class TestTransforms:
    def test__transform_2d_grid_to_reference_frame(self):

        grid_2d = np.array([[0.0, 1.0], [1.0, 1.0], [1.0, 0.0]])

        transformed_grid_2d = aa.util.geometry.transform_grid_2d_to_reference_frame(
            grid_2d=grid_2d, centre=(0.0, 0.0), angle=0.0
        )

        assert transformed_grid_2d == pytest.approx(
            np.array([[0.0, 1.0], [1.0, 1.0], [1.0, 0.0]])
        )

        transformed_grid_2d = aa.util.geometry.transform_grid_2d_to_reference_frame(
            grid_2d=grid_2d, centre=(0.0, 0.0), angle=45.0
        )

        assert transformed_grid_2d == pytest.approx(
            np.array(
                [
                    [-np.sqrt(2) / 2.0, np.sqrt(2) / 2.0],
                    [0.0, np.sqrt(2)],
                    [np.sqrt(2) / 2.0, np.sqrt(2) / 2.0],
                ]
            )
        )

        transformed_grid_2d = aa.util.geometry.transform_grid_2d_to_reference_frame(
            grid_2d=grid_2d, centre=(0.0, 0.0), angle=90.0
        )

        assert transformed_grid_2d == pytest.approx(
            np.array([[-1.0, 0.0], [-1.0, 1.0], [0.0, 1.0]])
        )

        transformed_grid_2d = aa.util.geometry.transform_grid_2d_to_reference_frame(
            grid_2d=grid_2d, centre=(0.0, 0.0), angle=180.0
        )

        assert transformed_grid_2d == pytest.approx(
            np.array([[0.0, -1.0], [-1.0, -1.0], [-1.0, 0.0]])
        )

        transformed_grid_2d = aa.util.geometry.transform_grid_2d_to_reference_frame(
            grid_2d=grid_2d, centre=(5.0, 10.0), angle=0.0
        )

        assert transformed_grid_2d == pytest.approx(
            np.array([[-5.0, -9.0], [-4.0, -9.0], [-4.0, -10.0]])
        )

        transformed_grid_2d = aa.util.geometry.transform_grid_2d_to_reference_frame(
            grid_2d=grid_2d, centre=(5.0, 10.0), angle=90.0
        )

        assert transformed_grid_2d == pytest.approx(
            np.array([[9.0, -5.0], [9.0, -4.0], [10.0, -4.0]])
        )

    def test__transform_2d_grid_from_reference_frame(self):

        grid_2d = np.array([[0.0, 1.0], [1.0, 1.0], [1.0, 0.0]])

        transformed_grid_2d = aa.util.geometry.transform_grid_2d_from_reference_frame(
            grid_2d=grid_2d, centre=(0.0, 0.0), angle=0.0
        )

        assert transformed_grid_2d == pytest.approx(
            np.array([[0.0, 1.0], [1.0, 1.0], [1.0, 0.0]])
        )

        transformed_grid_2d = aa.util.geometry.transform_grid_2d_from_reference_frame(
            grid_2d=grid_2d, centre=(0.0, 0.0), angle=45.0
        )

        assert transformed_grid_2d == pytest.approx(
            np.array(
                [
                    [np.sqrt(2) / 2.0, np.sqrt(2) / 2.0],
                    [np.sqrt(2), 0.0],
                    [np.sqrt(2) / 2.0, -np.sqrt(2) / 2.0],
                ]
            )
        )

        transformed_grid_2d = aa.util.geometry.transform_grid_2d_from_reference_frame(
            grid_2d=grid_2d, centre=(2.0, 2.0), angle=90.0
        )

        assert transformed_grid_2d == pytest.approx(
            np.array([[3.0, 2.0], [3.0, 1.0], [2.0, 1.0]])
        )

        transformed_grid_2d = aa.util.geometry.transform_grid_2d_to_reference_frame(
            grid_2d=grid_2d, centre=(8.0, 5.0), angle=137.0
        )

        original_grid_2d = aa.util.geometry.transform_grid_2d_from_reference_frame(
            grid_2d=transformed_grid_2d, centre=(8.0, 5.0), angle=137.0
        )

        assert grid_2d == pytest.approx(original_grid_2d, 1.0e-4)
