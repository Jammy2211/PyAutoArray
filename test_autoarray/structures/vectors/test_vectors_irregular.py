from os import path

from matplotlib.patches import Ellipse
import pytest
import numpy as np

import autoarray as aa
from autoarray import exc

test_vectors_dir = path.join(
    "{}".format(path.dirname(path.realpath(__file__))), "files", "vectors"
)


def test__input_vectors_as_different_types__all_converted_to_ndarray_correctly():

    # Input tuples

    vectors = aa.VectorYX2DIrregular(
        vectors=[(1.0, -1.0), (1.0, 1.0)], grid=[[1.0, -1.0], [1.0, 1.0]]
    )

    assert type(vectors) == aa.VectorYX2DIrregular
    assert (vectors == np.array([[1.0, -1.0], [1.0, 1.0]])).all()
    assert vectors.in_list == [(1.0, -1.0), (1.0, 1.0)]

    # Input np array

    vectors = aa.VectorYX2DIrregular(
        vectors=[np.array([1.0, -1.0]), np.array([1.0, 1.0])],
        grid=[[1.0, -1.0], [1.0, 1.0]],
    )

    assert type(vectors) == aa.VectorYX2DIrregular
    assert (vectors == np.array([[1.0, -1.0], [1.0, 1.0]])).all()
    assert vectors.in_list == [(1.0, -1.0), (1.0, 1.0)]

    # Input list

    vectors = aa.VectorYX2DIrregular(
        vectors=[[1.0, -1.0], [1.0, 1.0]], grid=[[1.0, -1.0], [1.0, 1.0]]
    )

    assert type(vectors) == aa.VectorYX2DIrregular
    assert (vectors == np.array([[1.0, -1.0], [1.0, 1.0]])).all()
    assert vectors.in_list == [(1.0, -1.0), (1.0, 1.0)]


def test__input_grids_as_different_types__all_converted_to_grid_irregular_correctly():

    # Input tuples

    vectors = aa.VectorYX2DIrregular(
        vectors=[(1.0, -1.0), (1.0, 1.0)],
        grid=aa.Grid2DIrregular([[1.0, -1.0], [1.0, 1.0]]),
    )

    assert type(vectors.grid) == aa.Grid2DIrregular
    assert (vectors.grid == np.array([[1.0, -1.0], [1.0, 1.0]])).all()

    # Input np array

    vectors = aa.VectorYX2DIrregular(
        vectors=[(1.0, -1.0), (1.0, 1.0)], grid=[[1.0, -1.0], [1.0, 1.0]]
    )

    assert type(vectors.grid) == aa.Grid2DIrregular
    assert (vectors.grid == np.array([[1.0, -1.0], [1.0, 1.0]])).all()

    # Input list

    vectors = aa.VectorYX2DIrregular(
        vectors=[(1.0, -1.0), (1.0, 1.0)], grid=[(1.0, -1.0), (1.0, 1.0)]
    )

    assert type(vectors.grid) == aa.Grid2DIrregular
    assert (vectors.grid == np.array([[1.0, -1.0], [1.0, 1.0]])).all()


def test__vectors_from_grid_within_radius():

    vectors = aa.VectorYX2DIrregular(
        vectors=[(1.0, 1.0), (2.0, 2.0)], grid=[[0.0, 1.0], [0.0, 2.0]]
    )

    vectors_masked = vectors.vectors_within_radius(radius=3.0, centre=(0.0, 0.0))

    assert type(vectors_masked) == aa.VectorYX2DIrregular
    assert vectors_masked.in_list == [(1.0, 1.0), (2.0, 2.0)]
    assert vectors_masked.grid.in_list == [(0.0, 1.0), (0.0, 2.0)]

    vectors_masked = vectors.vectors_within_radius(radius=1.5, centre=(0.0, 0.0))

    assert type(vectors_masked) == aa.VectorYX2DIrregular
    assert vectors_masked.in_list == [(1.0, 1.0)]
    assert vectors_masked.grid.in_list == [(0.0, 1.0)]

    vectors_masked = vectors.vectors_within_radius(radius=0.5, centre=(0.0, 2.0))

    assert type(vectors_masked) == aa.VectorYX2DIrregular
    assert vectors_masked.in_list == [(2.0, 2.0)]
    assert vectors_masked.grid.in_list == [(0.0, 2.0)]

    with pytest.raises(exc.VectorFieldException):
        vectors.vectors_within_radius(radius=0.0, centre=(0.0, 0.0))
