import autoarray as aa
import numpy as np

np.set_printoptions(threshold=np.inf)


def test__regularization_matrix__matches_util():

    neighbors = np.array(
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

    neighbors_sizes = np.array([4, 3, 3, 3, 4, 3, 3, 3, 2])

    mesh_grid = aa.m.MockMeshGrid(neighbors=neighbors, neighbors_sizes=neighbors_sizes)

    mapper = aa.m.MockMapper(source_mesh_grid=mesh_grid)

    reg = aa.reg.Constant(coefficient=2.0)
    regularization_matrix = reg.regularization_matrix_from(linear_obj=mapper)

    regularization_matrix_util = aa.util.regularization.constant_regularization_matrix_from(
        coefficient=2.0, neighbors=neighbors, neighbors_sizes=neighbors_sizes
    )

    assert reg.coefficient == 2.0
    assert (regularization_matrix == regularization_matrix_util).all()

    reg = aa.reg.ConstantSplit(coefficient=3.0)

    assert reg.coefficient == 3.0
