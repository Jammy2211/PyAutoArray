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

        line = aa.Line.manual_1d(line=[1.0, 2.0, 3.0, 4.0], pixel_scales=1.0)

        assert type(line) == lines.Line
        assert (line.in_1d == np.array([1.0, 2.0, 3.0, 4.0])).all()
        assert line.pixel_scale == 1.0
        assert line.pixel_scales == (1.0,)
        assert line.origin == (0.0,)
