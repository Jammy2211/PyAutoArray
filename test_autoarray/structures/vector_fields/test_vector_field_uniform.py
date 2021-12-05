from os import path
import numpy as np
import pytest

from autoconf import conf
import autoarray as aa
from autoarray import exc

test_grid_dir = path.join("{}".format(path.dirname(path.realpath(__file__))), "files")


class TestAPI:
    def test__manual(self):

        vectors = aa.VectorYX2D.manual_native(
            vectors=[[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]],
            pixel_scales=1.0,
            sub_size=1,
        )

        assert type(vectors) == aa.VectorYX2D
        assert (
            vectors == np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
        ).all()
        assert (
            vectors.native
            == np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])
        ).all()
        assert (
            vectors.slim == np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
        ).all()
        assert (
            vectors.binned.native
            == np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])
        ).all()
        assert (
            vectors.binned == np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
        ).all()
        assert (
            vectors.grid.native
            == np.array([[[0.5, -0.5], [0.5, 0.5]], [[-0.5, -0.5], [-0.5, 0.5]]])
        ).all()
        assert vectors.pixel_scales == (1.0, 1.0)
        assert vectors.origin == (0.0, 0.0)
        assert vectors.sub_size == 1

        vectors = aa.VectorYX2D.manual_slim(
            vectors=[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]],
            shape_native=(1, 1),
            pixel_scales=1.0,
            sub_size=2,
            origin=(0.0, 1.0),
        )

        assert type(vectors) == aa.VectorYX2D
        assert (
            vectors == np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
        ).all()
        assert (
            vectors.native
            == np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])
        ).all()
        assert (
            vectors.slim == np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
        ).all()

        assert (vectors.binned.native == np.array([[[4.0, 5.0]]])).all()
        print(vectors.grid)
        assert (vectors.binned.slim == np.array([[4.0, 5.0]])).all()
        assert (
            vectors.grid.native
            == np.array([[[0.25, 0.75], [0.25, 1.25]], [[-0.25, 0.75], [-0.25, 1.25]]])
        ).all()
        assert vectors.pixel_scales == (1.0, 1.0)
        assert vectors.origin == (0.0, 1.0)
        assert vectors.sub_size == 2
