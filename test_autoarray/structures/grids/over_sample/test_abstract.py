from os import path
import numpy as np
import pytest

from autoconf import conf
import autoarray as aa
from autoarray import exc


def test__structure_2d_from():
    mask = np.array(
        [
            [True, True, True, True],
            [True, False, False, True],
            [True, False, False, True],
            [True, True, True, True],
        ]
    )

    mask = aa.Mask2D(mask=mask, pixel_scales=(1.0, 1.0))

    over_sample = aa.OverSampleUniform()

    result = over_sample.structure_2d_from(
        result=np.array([1.0, 2.0, 3.0, 4.0]), mask=mask
    )

    assert isinstance(result, aa.Array2D)
    assert (
        result.native
        == np.array(
            [
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 2.0, 0.0],
                [0.0, 3.0, 4.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
            ]
        )
    ).all()

    result = over_sample.structure_2d_from(
        result=np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]]), mask=mask
    )

    assert isinstance(result, aa.Grid2D)
    assert (
        result.native
        == np.array(
            [
                [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
                [[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [0.0, 0.0]],
                [[0.0, 0.0], [3.0, 3.0], [4.0, 4.0], [0.0, 0.0]],
                [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
            ]
        )
    ).all()


def test__structure_2d_list_from():
    mask = np.array(
        [
            [True, True, True, True],
            [True, False, False, True],
            [True, False, False, True],
            [True, True, True, True],
        ]
    )

    mask = aa.Mask2D(mask=mask, pixel_scales=(1.0, 1.0))

    over_sample = aa.OverSampleUniform()

    result = over_sample.structure_2d_list_from(
        result_list=[np.array([1.0, 2.0, 3.0, 4.0])], mask=mask
    )

    assert isinstance(result[0], aa.Array2D)
    assert (
        result[0].native
        == np.array(
            [
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 2.0, 0.0],
                [0.0, 3.0, 4.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
            ]
        )
    ).all()

    result = over_sample.structure_2d_list_from(
        result_list=[np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]])],
        mask=mask,
    )

    assert isinstance(result[0], aa.Grid2D)
    assert (
        result[0].native
        == np.array(
            [
                [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
                [[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [0.0, 0.0]],
                [[0.0, 0.0], [3.0, 3.0], [4.0, 4.0], [0.0, 0.0]],
                [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
            ]
        )
    ).all()
