from os import path

import numpy as np

import autoarray as aa

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
