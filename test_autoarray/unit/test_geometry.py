import numpy as np
import pytest

import autoarray as aa


class TestCoordinates:
    def test__central_pixel__depends_on_shape_pixel_scale_and_origin(self):

        central_pixel_coordinates = aa.geometry.central_pixel_coordinates_from(
            shape=(3,)
        )
        assert central_pixel_coordinates == (1,)

        central_pixel_coordinates = aa.geometry.central_pixel_coordinates_from(
            shape=(3, 3),
        )
        assert central_pixel_coordinates == (1, 1)

        central_pixel_coordinates = aa.geometry.central_pixel_coordinates_from(
            shape=(3, 3),
        )
        assert central_pixel_coordinates == (1, 1)

        central_pixel_coordinates = aa.geometry.central_pixel_coordinates_from(
            shape=(4, 4),
        )
        assert central_pixel_coordinates == (1.5, 1.5)

        central_pixel_coordinates = aa.geometry.central_pixel_coordinates_from(
            shape=(4, 4),
        )
        assert central_pixel_coordinates == (1.5, 1.5)

    # def test__pixel_coordinates_from_scaled_coordinates(self):
    #
    #     mask = aa.Mask2D.manual(
    #         mask=np.full(fill_value=False, shape=(2, 2)), pixel_scales=(2.0, 2.0)
    #     )
    #
    #     assert mask.geometry.pixel_coordinates_from_scaled_coordinates(
    #         scaled_coordinates=(1.0, -1.0)
    #     ) == (0, 0)
    #     assert mask.geometry.pixel_coordinates_from_scaled_coordinates(
    #         scaled_coordinates=(1.0, 1.0)
    #     ) == (0, 1)
    #     assert mask.geometry.pixel_coordinates_from_scaled_coordinates(
    #         scaled_coordinates=(-1.0, -1.0)
    #     ) == (1, 0)
    #     assert mask.geometry.pixel_coordinates_from_scaled_coordinates(
    #         scaled_coordinates=(-1.0, 1.0)
    #     ) == (1, 1)
    #
    #     mask = aa.Mask2D.manual(
    #         mask=np.full(fill_value=False, shape=(3, 3)), pixel_scales=(3.0, 3.0)
    #     )
    #
    #     assert mask.geometry.pixel_coordinates_from_scaled_coordinates(
    #         scaled_coordinates=(3.0, -3.0)
    #     ) == (0, 0)
    #     assert mask.geometry.pixel_coordinates_from_scaled_coordinates(
    #         scaled_coordinates=(3.0, 0.0)
    #     ) == (0, 1)
    #     assert mask.geometry.pixel_coordinates_from_scaled_coordinates(
    #         scaled_coordinates=(3.0, 3.0)
    #     ) == (0, 2)
    #     assert mask.geometry.pixel_coordinates_from_scaled_coordinates(
    #         scaled_coordinates=(0.0, -3.0)
    #     ) == (1, 0)
    #     assert mask.geometry.pixel_coordinates_from_scaled_coordinates(
    #         scaled_coordinates=(0.0, 0.0)
    #     ) == (1, 1)
    #     assert mask.geometry.pixel_coordinates_from_scaled_coordinates(
    #         scaled_coordinates=(0.0, 3.0)
    #     ) == (1, 2)
    #     assert mask.geometry.pixel_coordinates_from_scaled_coordinates(
    #         scaled_coordinates=(-3.0, -3.0)
    #     ) == (2, 0)
    #     assert mask.geometry.pixel_coordinates_from_scaled_coordinates(
    #         scaled_coordinates=(-3.0, 0.0)
    #     ) == (2, 1)
    #     assert mask.geometry.pixel_coordinates_from_scaled_coordinates(
    #         scaled_coordinates=(-3.0, 3.0)
    #     ) == (2, 2)
    #
    # def test__pixel_coordinates_from_scaled_coordinates__scaled_are_pixel_corners(self):
    #
    #     mask = aa.Mask2D.manual(
    #         mask=np.full(fill_value=False, shape=(2, 2)), pixel_scales=(2.0, 2.0)
    #     )
    #
    #     assert mask.geometry.pixel_coordinates_from_scaled_coordinates(
    #         scaled_coordinates=(1.99, -1.99)
    #     ) == (0, 0)
    #     assert mask.geometry.pixel_coordinates_from_scaled_coordinates(
    #         scaled_coordinates=(1.99, -0.01)
    #     ) == (0, 0)
    #     assert mask.geometry.pixel_coordinates_from_scaled_coordinates(
    #         scaled_coordinates=(0.01, -1.99)
    #     ) == (0, 0)
    #     assert mask.geometry.pixel_coordinates_from_scaled_coordinates(
    #         scaled_coordinates=(0.01, -0.01)
    #     ) == (0, 0)
    #
    #     assert mask.geometry.pixel_coordinates_from_scaled_coordinates(
    #         scaled_coordinates=(2.01, 0.01)
    #     ) == (0, 1)
    #     assert mask.geometry.pixel_coordinates_from_scaled_coordinates(
    #         scaled_coordinates=(2.01, 1.99)
    #     ) == (0, 1)
    #     assert mask.geometry.pixel_coordinates_from_scaled_coordinates(
    #         scaled_coordinates=(0.01, 0.01)
    #     ) == (0, 1)
    #     assert mask.geometry.pixel_coordinates_from_scaled_coordinates(
    #         scaled_coordinates=(0.01, 1.99)
    #     ) == (0, 1)
    #
    #     assert mask.geometry.pixel_coordinates_from_scaled_coordinates(
    #         scaled_coordinates=(-0.01, -1.99)
    #     ) == (1, 0)
    #     assert mask.geometry.pixel_coordinates_from_scaled_coordinates(
    #         scaled_coordinates=(-0.01, -0.01)
    #     ) == (1, 0)
    #     assert mask.geometry.pixel_coordinates_from_scaled_coordinates(
    #         scaled_coordinates=(-1.99, -1.99)
    #     ) == (1, 0)
    #     assert mask.geometry.pixel_coordinates_from_scaled_coordinates(
    #         scaled_coordinates=(-1.99, -0.01)
    #     ) == (1, 0)
    #
    #     assert mask.geometry.pixel_coordinates_from_scaled_coordinates(
    #         scaled_coordinates=(-0.01, 0.01)
    #     ) == (1, 1)
    #     assert mask.geometry.pixel_coordinates_from_scaled_coordinates(
    #         scaled_coordinates=(-0.01, 1.99)
    #     ) == (1, 1)
    #     assert mask.geometry.pixel_coordinates_from_scaled_coordinates(
    #         scaled_coordinates=(-1.99, 0.01)
    #     ) == (1, 1)
    #     assert mask.geometry.pixel_coordinates_from_scaled_coordinates(
    #         scaled_coordinates=(-1.99, 1.99)
    #     ) == (1, 1)
    #
    # def test__pixel_coordinates_from_scaled_coordinates___scaled_are_pixel_centres__nonzero_centre(
    #     self,
    # ):
    #     mask = aa.Mask2D.manual(
    #         mask=np.full(fill_value=False, shape=(2, 2)),
    #         pixel_scales=(2.0, 2.0),
    #         origin=(1.0, 1.0),
    #     )
    #
    #     assert mask.geometry.pixel_coordinates_from_scaled_coordinates(
    #         scaled_coordinates=(2.0, 0.0)
    #     ) == (0, 0)
    #     assert mask.geometry.pixel_coordinates_from_scaled_coordinates(
    #         scaled_coordinates=(2.0, 2.0)
    #     ) == (0, 1)
    #     assert mask.geometry.pixel_coordinates_from_scaled_coordinates(
    #         scaled_coordinates=(0.0, 0.0)
    #     ) == (1, 0)
    #     assert mask.geometry.pixel_coordinates_from_scaled_coordinates(
    #         scaled_coordinates=(0.0, 2.0)
    #     ) == (1, 1)
    #
    #     mask = aa.Mask2D.manual(
    #         mask=np.full(fill_value=False, shape=(3, 3)),
    #         pixel_scales=(3.0, 3.0),
    #         origin=(3.0, 3.0),
    #     )
    #
    #     assert mask.geometry.pixel_coordinates_from_scaled_coordinates(
    #         scaled_coordinates=(6.0, 0.0)
    #     ) == (0, 0)
    #     assert mask.geometry.pixel_coordinates_from_scaled_coordinates(
    #         scaled_coordinates=(6.0, 3.0)
    #     ) == (0, 1)
    #     assert mask.geometry.pixel_coordinates_from_scaled_coordinates(
    #         scaled_coordinates=(6.0, 6.0)
    #     ) == (0, 2)
    #     assert mask.geometry.pixel_coordinates_from_scaled_coordinates(
    #         scaled_coordinates=(3.0, 0.0)
    #     ) == (1, 0)
    #     assert mask.geometry.pixel_coordinates_from_scaled_coordinates(
    #         scaled_coordinates=(3.0, 3.0)
    #     ) == (1, 1)
    #     assert mask.geometry.pixel_coordinates_from_scaled_coordinates(
    #         scaled_coordinates=(3.0, 6.0)
    #     ) == (1, 2)
    #     assert mask.geometry.pixel_coordinates_from_scaled_coordinates(
    #         scaled_coordinates=(0.0, 0.0)
    #     ) == (2, 0)
    #     assert mask.geometry.pixel_coordinates_from_scaled_coordinates(
    #         scaled_coordinates=(0.0, 3.0)
    #     ) == (2, 1)
    #     assert mask.geometry.pixel_coordinates_from_scaled_coordinates(
    #         scaled_coordinates=(0.0, 6.0)
    #     ) == (2, 2)
    #
    # def test__pixel_coordinates_from_scaled_coordinates__scaled_are_pixel_corners__nonzero_centre(
    #     self,
    # ):
    #     mask = aa.Mask2D.manual(
    #         mask=np.full(fill_value=False, shape=(2, 2)),
    #         pixel_scales=(2.0, 2.0),
    #         origin=(1.0, 1.0),
    #     )
    #
    #     assert mask.geometry.pixel_coordinates_from_scaled_coordinates(
    #         scaled_coordinates=(2.99, -0.99)
    #     ) == (0, 0)
    #     assert mask.geometry.pixel_coordinates_from_scaled_coordinates(
    #         scaled_coordinates=(2.99, 0.99)
    #     ) == (0, 0)
    #     assert mask.geometry.pixel_coordinates_from_scaled_coordinates(
    #         scaled_coordinates=(1.01, -0.99)
    #     ) == (0, 0)
    #     assert mask.geometry.pixel_coordinates_from_scaled_coordinates(
    #         scaled_coordinates=(1.01, 0.99)
    #     ) == (0, 0)
    #
    #     assert mask.geometry.pixel_coordinates_from_scaled_coordinates(
    #         scaled_coordinates=(3.01, 1.01)
    #     ) == (0, 1)
    #     assert mask.geometry.pixel_coordinates_from_scaled_coordinates(
    #         scaled_coordinates=(3.01, 2.99)
    #     ) == (0, 1)
    #     assert mask.geometry.pixel_coordinates_from_scaled_coordinates(
    #         scaled_coordinates=(1.01, 1.01)
    #     ) == (0, 1)
    #     assert mask.geometry.pixel_coordinates_from_scaled_coordinates(
    #         scaled_coordinates=(1.01, 2.99)
    #     ) == (0, 1)
    #
    #     assert mask.geometry.pixel_coordinates_from_scaled_coordinates(
    #         scaled_coordinates=(0.99, -0.99)
    #     ) == (1, 0)
    #     assert mask.geometry.pixel_coordinates_from_scaled_coordinates(
    #         scaled_coordinates=(0.99, 0.99)
    #     ) == (1, 0)
    #     assert mask.geometry.pixel_coordinates_from_scaled_coordinates(
    #         scaled_coordinates=(-0.99, -0.99)
    #     ) == (1, 0)
    #     assert mask.geometry.pixel_coordinates_from_scaled_coordinates(
    #         scaled_coordinates=(-0.99, 0.99)
    #     ) == (1, 0)
    #
    #     assert mask.geometry.pixel_coordinates_from_scaled_coordinates(
    #         scaled_coordinates=(0.99, 1.01)
    #     ) == (1, 1)
    #     assert mask.geometry.pixel_coordinates_from_scaled_coordinates(
    #         scaled_coordinates=(0.99, 2.99)
    #     ) == (1, 1)
    #     assert mask.geometry.pixel_coordinates_from_scaled_coordinates(
    #         scaled_coordinates=(-0.99, 1.01)
    #     ) == (1, 1)
    #     assert mask.geometry.pixel_coordinates_from_scaled_coordinates(
    #         scaled_coordinates=(-0.99, 2.99)
    #     ) == (1, 1)
    #
    # def test__scaled_coordinates_from_pixel_coordinates___scaled_are_pixel_centres__nonzero_centre(
    #     self,
    # ):
    #     mask = aa.Mask2D.manual(
    #         mask=np.full(fill_value=False, shape=(3, 3)), pixel_scales=(3.0, 3.0)
    #     )
    #
    #     assert mask.geometry.scaled_coordinates_from_pixel_coordinates(
    #         pixel_coordinates=(0, 0)
    #     ) == (3.0, -3.0)
    #     assert mask.geometry.scaled_coordinates_from_pixel_coordinates(
    #         pixel_coordinates=(0, 1)
    #     ) == (3.0, 0.0)
    #     assert mask.geometry.scaled_coordinates_from_pixel_coordinates(
    #         pixel_coordinates=(0, 2)
    #     ) == (3.0, 3.0)
    #     assert mask.geometry.scaled_coordinates_from_pixel_coordinates(
    #         pixel_coordinates=(1, 0)
    #     ) == (0.0, -3.0)
    #     assert mask.geometry.scaled_coordinates_from_pixel_coordinates(
    #         pixel_coordinates=(1, 1)
    #     ) == (0.0, 0.0)
    #     assert mask.geometry.scaled_coordinates_from_pixel_coordinates(
    #         pixel_coordinates=(1, 2)
    #     ) == (0.0, 3.0)
    #     assert mask.geometry.scaled_coordinates_from_pixel_coordinates(
    #         pixel_coordinates=(2, 0)
    #     ) == (-3.0, -3.0)
    #     assert mask.geometry.scaled_coordinates_from_pixel_coordinates(
    #         pixel_coordinates=(2, 1)
    #     ) == (-3.0, 0.0)
    #     assert mask.geometry.scaled_coordinates_from_pixel_coordinates(
    #         pixel_coordinates=(2, 2)
    #     ) == (-3.0, 3.0)
    #
    #     mask = aa.Mask2D.manual(
    #         mask=np.full(fill_value=False, shape=(2, 2)),
    #         pixel_scales=(2.0, 2.0),
    #         origin=(1.0, 1.0),
    #     )
    #
    #     assert mask.geometry.scaled_coordinates_from_pixel_coordinates(
    #         pixel_coordinates=(0, 0)
    #     ) == (2.0, 0.0)
    #     assert mask.geometry.scaled_coordinates_from_pixel_coordinates(
    #         pixel_coordinates=(0, 1)
    #     ) == (2.0, 2.0)
    #     assert mask.geometry.scaled_coordinates_from_pixel_coordinates(
    #         pixel_coordinates=(1, 0)
    #     ) == (0.0, 0.0)
    #     assert mask.geometry.scaled_coordinates_from_pixel_coordinates(
    #         pixel_coordinates=(1, 1)
    #     ) == (0.0, 2.0)
    #
    #     mask = aa.Mask2D.manual(
    #         mask=np.full(fill_value=False, shape=(3, 3)),
    #         pixel_scales=(3.0, 3.0),
    #         origin=(3.0, 3.0),
    #     )
    #
    #     assert mask.geometry.scaled_coordinates_from_pixel_coordinates(
    #         pixel_coordinates=(0, 0)
    #     ) == (6.0, 0.0)
    #     assert mask.geometry.scaled_coordinates_from_pixel_coordinates(
    #         pixel_coordinates=(0, 1)
    #     ) == (6.0, 3.0)
    #     assert mask.geometry.scaled_coordinates_from_pixel_coordinates(
    #         pixel_coordinates=(0, 2)
    #     ) == (6.0, 6.0)
    #     assert mask.geometry.scaled_coordinates_from_pixel_coordinates(
    #         pixel_coordinates=(1, 0)
    #     ) == (3.0, 0.0)
    #     assert mask.geometry.scaled_coordinates_from_pixel_coordinates(
    #         pixel_coordinates=(1, 1)
    #     ) == (3.0, 3.0)
    #     assert mask.geometry.scaled_coordinates_from_pixel_coordinates(
    #         pixel_coordinates=(1, 2)
    #     ) == (3.0, 6.0)
    #     assert mask.geometry.scaled_coordinates_from_pixel_coordinates(
    #         pixel_coordinates=(2, 0)
    #     ) == (0.0, 0.0)
    #     assert mask.geometry.scaled_coordinates_from_pixel_coordinates(
    #         pixel_coordinates=(2, 1)
    #     ) == (0.0, 3.0)
    #     assert mask.geometry.scaled_coordinates_from_pixel_coordinates(
    #         pixel_coordinates=(2, 2)
    #     ) == (0.0, 6.0)
