import numpy as np
import pytest

import autoarray as aa


def _make_grid_2d():
    mask = aa.Mask2D(
        mask=[
            [True, True, True, True],
            [True, False, False, True],
            [True, False, False, True],
            [True, True, True, True],
        ],
        pixel_scales=(1.0, 1.0),
    )
    return aa.Grid2D.from_mask(mask=mask)


def test__transform_applied():
    obj = aa.m.MockTransformProfile(centre=(1.0, 2.0))

    grid = _make_grid_2d()

    result = obj.scalar_from(grid=grid)

    # Grid coords are approximately [(0.5,-0.5),(0.5,0.5),(-0.5,-0.5),(-0.5,0.5)]
    # After transform (subtract centre [1.0, 2.0]):
    #   [0.5-1.0, -0.5-2.0] = [-0.5, -2.5] -> sum = -3.0
    #   [0.5-1.0,  0.5-2.0] = [-0.5, -1.5] -> sum = -2.0
    #   [-0.5-1.0,-0.5-2.0] = [-1.5, -2.5] -> sum = -4.0
    #   [-0.5-1.0, 0.5-2.0] = [-1.5, -1.5] -> sum = -3.0
    assert result == pytest.approx(np.array([-3.0, -2.0, -4.0, -3.0]))


def test__already_transformed_skipped():
    obj = aa.m.MockTransformProfile(centre=(1.0, 2.0))

    grid = _make_grid_2d()
    grid.is_transformed = True

    result = obj.scalar_from(grid=grid)

    # No transformation applied — sum of raw coords
    # [(0.5,-0.5),(0.5,0.5),(-0.5,-0.5),(-0.5,0.5)]
    # sums: [0.0, 1.0, -1.0, 0.0]
    assert result == pytest.approx(np.array([0.0, 1.0, -1.0, 0.0]))


def test__rotate_back():
    obj = aa.m.MockTransformProfile(centre=(1.0, 2.0))

    grid = _make_grid_2d()

    result = obj.vector_from(grid=grid)

    # Grid transformed (subtract centre), then doubled, then negated (rotate_back mock)
    # [-0.5,-2.5]*2 = [-1.0,-5.0] -> negated = [1.0, 5.0]
    # [-0.5,-1.5]*2 = [-1.0,-3.0] -> negated = [1.0, 3.0]
    # [-1.5,-2.5]*2 = [-3.0,-5.0] -> negated = [3.0, 5.0]
    # [-1.5,-1.5]*2 = [-3.0,-3.0] -> negated = [3.0, 3.0]
    expected = np.array([[1.0, 5.0], [1.0, 3.0], [3.0, 5.0], [3.0, 3.0]])
    assert result == pytest.approx(expected)
