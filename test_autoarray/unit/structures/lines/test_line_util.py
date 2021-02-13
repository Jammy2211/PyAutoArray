import autoarray as aa
import numpy as np
import pytest


def test__line_1d_slim_from():

    mask_1d = np.array([False, False, False, False])

    line_1d_native = np.array([1.0, 2.0, 3.0, 4.0])

    line_1d_slim = aa.util.line.line_1d_slim_from(
        line_1d_native=line_1d_native, mask_1d=mask_1d, sub_size=1
    )

    assert (line_1d_slim == np.array([1.0, 2.0, 3.0, 4.0])).all()

    mask_1d = np.array([True, False, False, True, False, False])

    line_1d_native = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])

    line_1d_slim = aa.util.line.line_1d_slim_from(
        line_1d_native=line_1d_native, mask_1d=mask_1d, sub_size=1
    )

    assert (line_1d_slim == np.array([2.0, 3.0, 5.0, 6.0])).all()

    mask_1d = np.array([True, True, False, True, False, False])

    line_1d_native = np.array(
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0]
    )

    line_1d_slim = aa.util.line.line_1d_slim_from(
        line_1d_native=line_1d_native, mask_1d=mask_1d, sub_size=2
    )

    assert (line_1d_slim == np.array([5.0, 6.0, 9.0, 10.0, 11.0, 12.0])).all()


def test__line_1d_native_from():

    mask_1d = np.array([False, False, False, False])

    line_1d_slim = np.array([1.0, 2.0, 3.0, 4.0])

    line_1d_native = aa.util.line.line_1d_native_from(
        line_1d_slim=line_1d_slim, mask_1d=mask_1d, sub_size=1
    )

    assert (line_1d_native == np.array([1.0, 2.0, 3.0, 4.0])).all()

    mask_1d = np.array([False, False, True, True, False, False])

    line_1d_slim = np.array([1.0, 2.0, 3.0, 4.0])

    line_1d_native = aa.util.line.line_1d_native_from(
        line_1d_slim=line_1d_slim, mask_1d=mask_1d, sub_size=1
    )

    assert (line_1d_native == np.array([1.0, 2.0, 0.0, 0.0, 3.0, 4.0])).all()

    mask_1d = np.array([True, False, False, True, False, False])

    line_1d_slim = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])

    line_1d_native = aa.util.line.line_1d_native_from(
        line_1d_slim=line_1d_slim, mask_1d=mask_1d, sub_size=2
    )

    assert (
        line_1d_native
        == np.array([0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 0.0, 0.0, 5.0, 6.0, 7.0, 8.0])
    ).all()
