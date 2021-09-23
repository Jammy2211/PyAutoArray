import numpy as np
from os import path
import pytest


import autoarray as aa
from autoarray.mock.mock import MockMapper, MockLinearEqn, MockInversion

directory = path.dirname(path.realpath(__file__))


class TestAbstractLinearEqn:
    def test__brightest_reconstruction_pixel_and_centre(self):

        matrix_shape = (9, 3)

        mapper = MockMapper(
            matrix_shape,
            source_pixelization_grid=aa.Grid2DVoronoi.manual_slim(
                [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [5.0, 0.0]]
            ),
        )

        linear_eqn = MockLinearEqn(mapper=mapper)

        reconstruction = np.array([2.0, 3.0, 5.0, 0.0])

        brightest_reconstruction_pixel = linear_eqn.brightest_reconstruction_pixel_from(
            reconstruction=reconstruction
        )

        assert brightest_reconstruction_pixel == 2

        brightest_reconstruction_pixel_centre = linear_eqn.brightest_reconstruction_pixel_centre_from(
            reconstruction=reconstruction
        )

        assert brightest_reconstruction_pixel_centre.in_list == [(5.0, 6.0)]
