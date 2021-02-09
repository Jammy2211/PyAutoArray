from os import path
import numpy as np
import pytest

import autoarray as aa
from autoarray import exc
from autoarray.structures import lines

test_line_dir = path.join(
    "{}".format(path.dirname(path.realpath(__file__))), "files", "lines"
)


class TestAPI:
    def test__manual__makes_line_with_pixel_scale(self):

        line = aa.Line1D.manual_slim(line=[1.0, 2.0, 3.0, 4.0], pixel_scales=1.0)

        assert type(line) == lines.Line1D
        assert (line.native == np.array([1.0, 2.0, 3.0, 4.0])).all()
        assert (line.slim == np.array([1.0, 2.0, 3.0, 4.0])).all()
        assert line.pixel_scale == 1.0
        assert line.pixel_scales == (1.0,)
        assert line.origin == (0.0,)

        line = aa.Line1D.manual_native(line=[1.0, 2.0, 3.0, 4.0], pixel_scales=1.0)

        assert type(line) == lines.Line1D
        assert (line.native == np.array([1.0, 2.0, 3.0, 4.0])).all()
        assert (line.slim == np.array([1.0, 2.0, 3.0, 4.0])).all()
        assert line.pixel_scale == 1.0
        assert line.pixel_scales == (1.0,)
        assert line.origin == (0.0,)

    def test__manual_mask__makes_line_using_input_mask(self):

        mask = aa.Mask1D.manual(
            mask=[True, False, False, True, False, False], pixel_scales=1.0, sub_size=1
        )

        line = aa.Line1D.manual_mask(line=[100.0, 1.0, 2.0, 100.0, 3.0, 4.0], mask=mask)

        assert type(line) == lines.Line1D
        assert (line.native == np.array([0.0, 1.0, 2.0, 0.0, 3.0, 4.0])).all()
        assert (line.slim == np.array([1.0, 2.0, 3.0, 4.0])).all()
        assert line.pixel_scale == 1.0
        assert line.pixel_scales == (1.0,)
        assert line.origin == (0.0,)
