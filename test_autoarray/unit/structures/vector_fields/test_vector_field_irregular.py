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

    vector_field = aa.VectorFieldIrregular(
        vectors=[(1.0, -1.0), (1.0, 1.0)], grid=[[1.0, -1.0], [1.0, 1.0]]
    )

    assert type(vector_field) == aa.VectorFieldIrregular
    assert (vector_field == np.array([[1.0, -1.0], [1.0, 1.0]])).all()
    assert vector_field.in_1d_list == [(1.0, -1.0), (1.0, 1.0)]

    # Input np array

    vector_field = aa.VectorFieldIrregular(
        vectors=[np.array([1.0, -1.0]), np.array([1.0, 1.0])],
        grid=[[1.0, -1.0], [1.0, 1.0]],
    )

    assert type(vector_field) == aa.VectorFieldIrregular
    assert (vector_field == np.array([[1.0, -1.0], [1.0, 1.0]])).all()
    assert vector_field.in_1d_list == [(1.0, -1.0), (1.0, 1.0)]

    # Input list

    vector_field = aa.VectorFieldIrregular(
        vectors=[[1.0, -1.0], [1.0, 1.0]], grid=[[1.0, -1.0], [1.0, 1.0]]
    )

    assert type(vector_field) == aa.VectorFieldIrregular
    assert (vector_field == np.array([[1.0, -1.0], [1.0, 1.0]])).all()
    assert vector_field.in_1d_list == [(1.0, -1.0), (1.0, 1.0)]


def test__input_grids_as_different_types__all_converted_to_grid_irregular_correctly():

    # Input tuples

    vector_field = aa.VectorFieldIrregular(
        vectors=[(1.0, -1.0), (1.0, 1.0)],
        grid=aa.GridIrregular([[1.0, -1.0], [1.0, 1.0]]),
    )

    assert type(vector_field.grid) == aa.GridIrregular
    assert (vector_field.grid == np.array([[1.0, -1.0], [1.0, 1.0]])).all()

    # Input np array

    vector_field = aa.VectorFieldIrregular(
        vectors=[(1.0, -1.0), (1.0, 1.0)], grid=[[1.0, -1.0], [1.0, 1.0]]
    )

    assert type(vector_field.grid) == aa.GridIrregular
    assert (vector_field.grid == np.array([[1.0, -1.0], [1.0, 1.0]])).all()

    # Input list

    vector_field = aa.VectorFieldIrregular(
        vectors=[(1.0, -1.0), (1.0, 1.0)], grid=[(1.0, -1.0), (1.0, 1.0)]
    )

    assert type(vector_field.grid) == aa.GridIrregular
    assert (vector_field.grid == np.array([[1.0, -1.0], [1.0, 1.0]])).all()


def test__elliptical_properties_and_patches():

    vector_field = aa.VectorFieldIrregular(
        vectors=[(0.0, 1.0), (1.0, 0.0), (1.0, 1.0)],
        grid=[[1.0, -1.0], [1.0, 1.0], [0.0, 0.0]],
    )

    assert isinstance(vector_field.ellipticities, aa.Values)
    assert vector_field.ellipticities.in_1d_list == [1.0, 1.0, np.sqrt(2.0)]

    assert isinstance(vector_field.semi_major_axes, aa.Values)
    assert vector_field.semi_major_axes.in_1d_list == pytest.approx(
        [6.0, 6.0, 7.242640], 1.0e-4
    )

    assert isinstance(vector_field.semi_minor_axes, aa.Values)
    assert vector_field.semi_minor_axes.in_1d_list == pytest.approx(
        [0.0, 0.0, -1.242640], 1.0e-4
    )

    assert isinstance(vector_field.phis, aa.Values)
    assert vector_field.phis.in_1d_list == pytest.approx([0.0, 45.0, 22.5], 1.0e-4)

    assert isinstance(vector_field.elliptical_patches[0], Ellipse)
    assert vector_field.elliptical_patches[1].center == pytest.approx(
        (1.0, 1.0), 1.0e-4
    )
    assert vector_field.elliptical_patches[1].width == pytest.approx(6.0, 1.0e-4)
    assert vector_field.elliptical_patches[1].height == pytest.approx(0.0, 1.0e-4)
    assert vector_field.elliptical_patches[1].angle == pytest.approx(45.0, 1.0e-4)


def test__vectors_from_grid_within_radius():

    vector_field = aa.VectorFieldIrregular(
        vectors=[(1.0, 1.0), (2.0, 2.0)], grid=[[0.0, 1.0], [0.0, 2.0]]
    )

    vector_field_masked = vector_field.vectors_within_radius(
        radius=3.0, centre=(0.0, 0.0)
    )

    assert type(vector_field_masked) == aa.VectorFieldIrregular
    assert vector_field_masked.in_1d_list == [(1.0, 1.0), (2.0, 2.0)]
    assert vector_field_masked.grid.in_1d_list == [(0.0, 1.0), (0.0, 2.0)]

    vector_field_masked = vector_field.vectors_within_radius(
        radius=1.5, centre=(0.0, 0.0)
    )

    assert type(vector_field_masked) == aa.VectorFieldIrregular
    assert vector_field_masked.in_1d_list == [(1.0, 1.0)]
    assert vector_field_masked.grid.in_1d_list == [(0.0, 1.0)]

    vector_field_masked = vector_field.vectors_within_radius(
        radius=0.5, centre=(0.0, 2.0)
    )

    assert type(vector_field_masked) == aa.VectorFieldIrregular
    assert vector_field_masked.in_1d_list == [(2.0, 2.0)]
    assert vector_field_masked.grid.in_1d_list == [(0.0, 2.0)]

    with pytest.raises(exc.VectorFieldException):
        vector_field.vectors_within_radius(radius=0.0, centre=(0.0, 0.0))
