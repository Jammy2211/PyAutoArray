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

    def test__manual_mask(self):

        mask = aa.Mask2D.unmasked(shape_native=(2, 2), pixel_scales=1.0)

        vectors = aa.VectorYX2D.manual_mask(
            vectors=[[[1.0, 1.0], [2.0, 2.0]], [[3.0, 3.0], [4.0, 4.0]]], mask=mask
        )

        assert type(vectors) == aa.VectorYX2D
        assert (
            vectors.native
            == np.array([[[1.0, 1.0], [2.0, 2.0]], [[3.0, 3.0], [4.0, 4.0]]])
        ).all()
        assert (
            vectors.slim == np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]])
        ).all()
        assert (
            vectors.grid.native
            == np.array([[[0.5, -0.5], [0.5, 0.5]], [[-0.5, -0.5], [-0.5, 0.5]]])
        ).all()
        assert (
            vectors.grid.slim
            == np.array([[0.5, -0.5], [0.5, 0.5], [-0.5, -0.5], [-0.5, 0.5]])
        ).all()
        assert vectors.pixel_scales == (1.0, 1.0)
        assert vectors.origin == (0.0, 0.0)

        mask = aa.Mask2D.manual(
            mask=[[False, False], [True, False]], pixel_scales=1.0, origin=(0.0, 1.0)
        )

        vectors = aa.VectorYX2D.manual_mask(
            vectors=[[1.0, 1.0], [2.0, 2.0], [4.0, 4.0]], mask=mask
        )

        assert type(vectors) == aa.VectorYX2D
        assert (
            vectors.native
            == np.array([[[1.0, 1.0], [2.0, 2.0]], [[0.0, 0.0], [4.0, 4.0]]])
        ).all()
        assert (vectors.slim == np.array([[1.0, 1.0], [2.0, 2.0], [4.0, 4.0]])).all()
        assert (
            vectors.grid.native
            == np.array([[[0.5, 0.5], [0.5, 1.5]], [[0.0, 0.0], [-0.5, 1.5]]])
        ).all()
        assert (
            vectors.grid.slim == np.array([[0.5, 0.5], [0.5, 1.5], [-0.5, 1.5]])
        ).all()
        assert vectors.pixel_scales == (1.0, 1.0)
        assert vectors.origin == (0.0, 1.0)

        mask = aa.Mask2D.manual(
            mask=[[False, False], [True, False]], pixel_scales=1.0, origin=(0.0, 1.0)
        )

        vectors = aa.VectorYX2D.manual_mask(
            vectors=[[[1.0, 1.0], [2.0, 2.0]], [[3.0, 3.0], [4.0, 4.0]]], mask=mask
        )

        assert type(vectors) == aa.VectorYX2D
        assert (
            vectors.native
            == np.array([[[1.0, 1.0], [2.0, 2.0]], [[0.0, 0.0], [4.0, 4.0]]])
        ).all()
        assert (vectors.slim == np.array([[1.0, 1.0], [2.0, 2.0], [4.0, 4.0]])).all()
        assert (
            vectors.grid.native
            == np.array([[[0.5, 0.5], [0.5, 1.5]], [[0.0, 0.0], [-0.5, 1.5]]])
        ).all()
        assert (
            vectors.grid.slim == np.array([[0.5, 0.5], [0.5, 1.5], [-0.5, 1.5]])
        ).all()
        assert vectors.pixel_scales == (1.0, 1.0)
        assert vectors.origin == (0.0, 1.0)

        mask = aa.Mask2D.manual(mask=[[False], [True]], pixel_scales=2.0, sub_size=2)

        vectors = aa.VectorYX2D.manual_mask(
            vectors=[[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]], mask=mask
        )

        assert type(vectors) == aa.VectorYX2D
        assert (
            vectors == np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]])
        ).all()
        assert (
            vectors.native
            == np.array(
                [
                    [[1.0, 1.0], [2.0, 2.0]],
                    [[3.0, 3.0], [4.0, 4.0]],
                    [[0.0, 0.0], [0.0, 0.0]],
                    [[0.0, 0.0], [0.0, 0.0]],
                ]
            )
        ).all()
        assert (
            vectors.slim == np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]])
        ).all()
        assert (vectors.binned.native == np.array([[[2.5, 2.5]], [[0.0, 0.0]]])).all()
        assert (vectors.binned.slim == np.array([2.5])).all()
        assert vectors.pixel_scales == (2.0, 2.0)
        assert vectors.origin == (0.0, 0.0)
        assert vectors.mask.sub_size == 2

        mask = aa.Mask2D.manual(mask=[[False], [True]], pixel_scales=2.0, sub_size=2)
        vectors = aa.VectorYX2D.manual_slim(
            vectors=[
                [1.0, 1.0],
                [2.0, 2.0],
                [3.0, 3.0],
                [4.0, 4.0],
                [5.0, 5.0],
                [6.0, 6.0],
                [7.0, 7.0],
                [8.0, 8.0],
            ],
            shape_native=(2, 1),
            pixel_scales=2.0,
            sub_size=2,
        )
        vectors = vectors.apply_mask(mask=mask)

        assert (
            vectors.native
            == np.array(
                [
                    [[1.0, 1.0], [2.0, 2.0]],
                    [[3.0, 3.0], [4.0, 4.0]],
                    [[0.0, 0.0], [0.0, 0.0]],
                    [[0.0, 0.0], [0.0, 0.0]],
                ]
            )
        ).all()

    def test__full_ones_zeros(self):

        vectors = aa.VectorYX2D.ones(shape_native=(2, 2), pixel_scales=1.0, sub_size=1)

        assert type(vectors) == aa.VectorYX2D
        assert (
            vectors == np.array([[1.0, 1.0], [1.0, 1.0], [1.0, 1.0], [1.0, 1.0]])
        ).all()
        assert (
            vectors.native
            == np.array([[[1.0, 1.0], [1.0, 1.0]], [[1.0, 1.0], [1.0, 1.0]]])
        ).all()
        assert (
            vectors.slim == np.array([[1.0, 1.0], [1.0, 1.0], [1.0, 1.0], [1.0, 1.0]])
        ).all()
        assert (
            vectors.binned.native
            == np.array([[[1.0, 1.0], [1.0, 1.0]], [[1.0, 1.0], [1.0, 1.0]]])
        ).all()
        assert (
            vectors.binned == np.array([[1.0, 1.0], [1.0, 1.0], [1.0, 1.0], [1.0, 1.0]])
        ).all()
        assert (
            vectors.grid.native
            == np.array([[[0.5, -0.5], [0.5, 0.5]], [[-0.5, -0.5], [-0.5, 0.5]]])
        ).all()
        assert vectors.pixel_scales == (1.0, 1.0)
        assert vectors.origin == (0.0, 0.0)
        assert vectors.sub_size == 1

        vectors = aa.VectorYX2D.zeros(
            shape_native=(1, 1), pixel_scales=1.0, sub_size=2, origin=(0.0, 1.0)
        )

        assert type(vectors) == aa.VectorYX2D
        assert (
            vectors == np.array([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]])
        ).all()
        assert (
            vectors.native
            == np.array([[[0.0, 0.0], [0.0, 0.0]], [[0.0, 0.0], [0.0, 0.0]]])
        ).all()
        assert (
            vectors.slim == np.array([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]])
        ).all()

        assert (vectors.binned.native == np.array([[[0.0, 0.0]]])).all()
        print(vectors.grid)
        assert (vectors.binned.slim == np.array([[0.0, 0.0]])).all()
        assert (
            vectors.grid.native
            == np.array([[[0.25, 0.75], [0.25, 1.25]], [[-0.25, 0.75], [-0.25, 1.25]]])
        ).all()
        assert vectors.pixel_scales == (1.0, 1.0)
        assert vectors.origin == (0.0, 1.0)
        assert vectors.sub_size == 2
