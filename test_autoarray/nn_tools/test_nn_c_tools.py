import autoarray as aa
import numpy as np
from autoarray.nn_tools import nn_c_tools


def test__returning_weights_correct():

    pixelization_grid = aa.Grid2D.manual_slim(
            [[1.0, 1.0], [0.0, 1.0], [-1.0, 1.0], [-1.0, 0.0], [-1.0, -1.0], [0.0, -1.0], [1.0, -1.0], [1.0, 0.0], [0.0, 0.0]],
            shape_native=(3, 3),
            pixel_scales=1.0,
            )

    interpolate_grid = aa.Grid2D.manual_slim(
            [[0.5, 0.5], [-0.5, 0.5]],
            shape_native=(2, 1),
            pixel_scales=1.0,
            )

    max_nneighbours = int(30)

    weights, neighbour_indexes = nn_c_tools.natural_interpolation_weights(
            pixelization_grid[:, 1],
            pixelization_grid[:, 0],
            interpolate_grid[:, 1],
            interpolate_grid[:, 0],
            max_nneighbours)

    weights_answer = np.zeros((2, max_nneighbours))
    weights_answer[0][:4] = 0.25
    weights_answer[1][:4] = 0.25

    indexes_answer = np.zeros((2, max_nneighbours), dtype=np.intc) - 1
    indexes_answer[0][0] = 7
    indexes_answer[0][1] = 1
    indexes_answer[0][2] = 0
    indexes_answer[0][3] = 8

    indexes_answer[1][0] = 8
    indexes_answer[1][1] = 2 
    indexes_answer[1][2] = 1 
    indexes_answer[1][3] = 3 

    assert (neighbour_indexes == indexes_answer).all()

    assert (weights == weights_answer).all()



def test__nn_interpolation_correct():

    pixelization_grid = aa.Grid2D.manual_slim(
            [[1.0, 1.0], [0.0, 1.0], [-1.0, 1.0], [-1.0, 0.0], [-1.0, -1.0], [0.0, -1.0], [1.0, -1.0], [1.0, 0.0], [0.0, 0.0]],
            shape_native=(3, 3),
            pixel_scales=1.0,
            )

    input_values = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])

    interpolate_grid = aa.Grid2D.manual_slim(
            [[0.5, 0.5], [-0.5, 0.5], [2.0, 2.0]],
            shape_native=(3, 1),
            pixel_scales=1.0,
            )

    max_nneighbours = int(30)

    interpolated_values = nn_c_tools.natural_interpolation(
            pixelization_grid[:, 1],
            pixelization_grid[:, 0],
            input_values,
            interpolate_grid[:, 1],
            interpolate_grid[:, 0])

    answer = np.array([5.0, 4.5, 1.0])

    assert (interpolated_values == answer).all()

