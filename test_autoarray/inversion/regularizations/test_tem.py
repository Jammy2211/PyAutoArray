import autoarray as aa
import numpy as np

import pytest


def test__splitted():
    # Here, we define the pixel_neighbors first here and make the B matrices based on them.

    # You'll notice that actually, the B Matrix doesn't have to have the -1's going down the diagonal and we
    # don't have to have as many B matrices as we do the pix pixel with the most  vertices. We can combine
    # the rows of each B matrix wherever we like ;0.

    splitted_mappings = np.array([[0, -1, -1, -1, -1],
                                  [1, 3, -1, -1, -1],
                                  [1, 4, 2, -1, -1],
                                  [2, 3, -1, -1, -1],
                                  [1, 2, 3, 4, -1],
                                  [0, 3, 4, -1, -1],
                                  [4, -1, -1, -1, -1],
                                  [3, -1, -1, -1, -1],
                                  [0, 3, -1, -1, -1],
                                  [2, 3, -1, -1, -1],
                                  [0, -1, -1, -1, -1],
                                  [3, -1, -1, -1, -1],
                                  [4, 2, -1, -1, -1],
                                  [1, 4, -1, -1, -1],
                                  [2, 4, -1, -1, -1],
                                  [3, 1, 2, -1, -1],
                                  [2, 1, 4, -1, -1],
                                  [2, -1, -1, -1, -1],
                                  [3, 4, -1, -1, -1], 
                                  [1, 4, -1, -1, -1]])

    splitted_sizes = np.sum(splitted_mappings != -1, axis=1)

    splitted_weights = np.array([[1.0, 0.0, 0.0, 0.0, 0.0],
                                  [0.2, 0.8, 0.0, 0.0, 0.0],
                                  [0.1, 0.3, 0.6, 0.0, 0.0],
                                  [0.15, 0.85, 0.0, 0.0, 0.0],
                                  [0.2, 0.25, 0.1, 0.45, 0.0],
                                  [0.3, 0.6, 0.1, 0.0, 0.0],
                                  [1.0, 0.0, 0.0, 0.0, 0.0],
                                  [1.0, 0.0, 0.0, 0.0, 0.0],
                                  [0.7, 0.3, 0.0, 0.0, 0.0],
                                  [0.36, 0.64, 0.0, 0.0, 0.0],
                                  [1.0, 0.0, 0.0, 0.0, 0.0],
                                  [1.0, 0.0, 0.0, 0.0, 0.0],
                                  [0.95, 0.05, 0.0, 0.0, 0.0],
                                  [0.1, 0.9, 0.0, 0.0, 0.0],
                                  [0.77, 0.23, 0.0, 0.0, 0.0],
                                  [0.12, 0.4, 0.48, 0.0, 0.0],
                                  [0.6, 0.15, 0.25, 0.0, 0.0],
                                  [1.0, 0.0, 0.0, 0.0, 0.0],
                                  [0.66, 0.34, 0.0, 0.0, 0.0], 
                                  [0.57, 0.43, 0.0, 0.0, 0.0]])

    splitted_weights *= -1.0

    for i in range(len(splitted_mappings)):
        pixel_index = i // 4
        flag = 0
        for j in range(splitted_sizes[i]):
            if splitted_mappings[i][j] == pixel_index:
                splitted_weights[i][j] += 1.0
                flag = 1

        if flag == 0:
            splitted_mappings[i][j + 1] = pixel_index
            splitted_sizes[i] += 1
            splitted_weights[i][j + 1] = 1.0


    regularization_matrix = aa.inversion.regularization.regularization_util.constant_pixel_splitted_regularization_matrix_from(
        coefficient=1.0,
        splitted_mappings=splitted_mappings,
        splitted_sizes=splitted_sizes,
        splitted_weights=splitted_weights
    )


    assert (pytest.approx(regularization_matrix[0], 1e-4) == np.array([4.58, -0.6, -2.45, -1.26, -0.27]))


if __name__ == '__main__':
    test__splitted()



