import autoarray as aa
import numpy as np

from autoarray.mock.mock import MockPixelizationGrid, MockMapper


def test__regularization_matrix__matches_util():

    pixel_neighbors = np.array(
        [
            [1, 3, 7, 2],
            [4, 2, 0, -1],
            [1, 5, 3, -1],
            [4, 6, 0, -1],
            [7, 1, 5, 3],
            [4, 2, 8, -1],
            [7, 3, 0, -1],
            [4, 8, 6, -1],
            [7, 5, -1, -1],
        ]
    )

    pixel_neighbors_sizes = np.array([4, 3, 3, 3, 4, 3, 3, 3, 2])

    pixelization_grid = MockPixelizationGrid(
        pixel_neighbors=pixel_neighbors, pixel_neighbors_sizes=pixel_neighbors_sizes
    )

    mapper = MockMapper(source_pixelization_grid=pixelization_grid)

    reg = aa.reg.Constant(coefficient=1.0)
    regularization_matrix = reg.regularization_matrix_from(mapper=mapper)

    regularization_matrix_util = aa.util.regularization.constant_regularization_matrix_from(
        coefficient=1.0,
        pixel_neighbors=pixel_neighbors,
        pixel_neighbors_sizes=pixel_neighbors_sizes,
    )

    assert (regularization_matrix == regularization_matrix_util).all()
