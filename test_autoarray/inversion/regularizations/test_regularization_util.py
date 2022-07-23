import autoarray as aa
import numpy as np

import pytest


class TestConstantMatrix:
    def test__1_b_matrix_size_3x3__weight_list_all_1s__makes_correct_regularization_matrix(
        self,
    ):
        # Here, we define the neighbors first here and make the B matrices based on them.

        # You'll notice that actually, the B Matrix doesn't have to have the -1's going down the diagonal and we
        # don't have to have as many B matrices as we do the pix pixel with the most  vertices. We can combine
        # the rows of each B matrix wherever we like ;0.

        neighbors = np.array([[1, 2, -1], [0, -1, -1], [0, -1, -1]])

        neighbors_sizes = np.array([2, 1, 1])

        test_b_matrix = np.array(
            [[-1, 1, 0], [-1, 0, 1], [0, 0, 0]]  # Pair 1  # Pair 2
        )  # Pair 1 flip

        test_regularization_matrix = np.matmul(
            test_b_matrix.T, test_b_matrix
        ) + 1e-8 * np.identity(3)

        regularization_matrix = aa.util.regularization.constant_regularization_matrix_from(
            coefficient=1.0,
            neighbors=neighbors,
            neighbors_sizes=neighbors_sizes,
        )

        assert (regularization_matrix == test_regularization_matrix).all()
        assert abs(np.linalg.det(regularization_matrix)) > 1e-8

    def test__1_b_matrix_size_4x4__weight_list_all_1s__makes_correct_regularization_matrix(
        self,
    ):

        test_b_matrix = np.array(
            [[-1, 1, 0, 0], [0, -1, 1, 0], [0, 0, -1, 1], [1, 0, 0, -1]]
        )

        test_regularization_matrix = np.matmul(
            test_b_matrix.T, test_b_matrix
        ) + 1e-8 * np.identity(4)

        neighbors = np.array(
            [[1, 3, -1, -1], [0, 2, -1, -1], [1, 3, -1, -1], [0, 2, -1, -1]]
        )

        neighbors_sizes = np.array([2, 2, 2, 2])

        regularization_matrix = aa.util.regularization.constant_regularization_matrix_from(
            coefficient=1.0,
            neighbors=neighbors,
            neighbors_sizes=neighbors_sizes,
        )

        assert (regularization_matrix == test_regularization_matrix).all()
        assert abs(np.linalg.det(regularization_matrix)) > 1e-8

    def test__1_b_matrix_size_4x4__coefficient_2__makes_correct_regularization_matrix(
        self,
    ):

        neighbors = np.array(
            [[1, 3, -1, -1], [0, 2, -1, -1], [1, 3, -1, -1], [0, 2, -1, -1]]
        )

        neighbors_sizes = np.array([2, 2, 2, 2])

        test_b_matrix = 2.0 * np.array(
            [[-1, 1, 0, 0], [0, -1, 1, 0], [0, 0, -1, 1], [1, 0, 0, -1]]
        )

        test_regularization_matrix = np.matmul(
            test_b_matrix.T, test_b_matrix
        ) + 1e-8 * np.identity(4)

        regularization_matrix = aa.util.regularization.constant_regularization_matrix_from(
            coefficient=2.0,
            neighbors=neighbors,
            neighbors_sizes=neighbors_sizes,
        )

        assert (regularization_matrix == test_regularization_matrix).all()
        assert abs(np.linalg.det(regularization_matrix)) > 1e-8

    def test__1_b_matrix_size_9x9__coefficient_2__makes_correct_regularization_matrix(
        self,
    ):

        neighbors = np.array(
            [
                [1, 3, -1, -1],
                [4, 2, 0, -1],
                [1, 5, -1, -1],
                [4, 6, 0, -1],
                [7, 1, 5, 3],
                [4, 2, 8, -1],
                [7, 3, -1, -1],
                [4, 8, 6, -1],
                [7, 5, -1, -1],
            ]
        )

        neighbors_sizes = np.array([2, 3, 2, 3, 4, 3, 2, 3, 2])

        test_b_matrix_0 = np.array(
            [
                [-1, 1, 0, 0, 0, 0, 0, 0, 0],
                [-1, 0, 0, 1, 0, 0, 0, 0, 0],
                [0, -1, 1, 0, 0, 0, 0, 0, 0],
                [0, -1, 0, 0, 1, 0, 0, 0, 0],
                [0, 0, -1, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, -1, 1, 0, 0, 0, 0],
                [0, 0, 0, -1, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, -1, 1, 0, 0, 0],
                [0, 0, 0, 0, -1, 0, 0, 1, 0],
            ]
        )

        test_b_matrix_1 = np.array(
            [
                [0, 0, 0, 0, 0, -1, 0, 0, 1],
                [0, 0, 0, 0, 0, 0, -1, 1, 0],
                [0, 0, 0, 0, 0, 0, 0, -1, 1],
                [0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0],
            ]
        )

        test_regularization_matrix_0 = np.matmul(test_b_matrix_0.T, test_b_matrix_0)
        test_regularization_matrix_1 = np.matmul(test_b_matrix_1.T, test_b_matrix_1)

        test_regularization_matrix = (
            test_regularization_matrix_0
            + test_regularization_matrix_1
            + 1e-8 * np.identity(9)
        )

        regularization_matrix = aa.util.regularization.constant_regularization_matrix_from(
            coefficient=1.0,
            neighbors=neighbors,
            neighbors_sizes=neighbors_sizes,
        )

        assert (regularization_matrix == test_regularization_matrix).all()
        assert abs(np.linalg.det(regularization_matrix)) > 1e-8


class TestAdaptiveWeightList:
    def test__pixel_signals_all_1s__coefficients_all_1s__weight_list_all_1s(self):

        pixel_signals = np.array([1.0, 1.0, 1.0])

        weight_list = aa.util.regularization.adaptive_regularization_weights_from(
            inner_coefficient=1.0, outer_coefficient=1.0, pixel_signals=pixel_signals
        )

        assert (weight_list == np.array([1.0, 1.0, 1.0])).all()

    def test__pixel_signals_vary__coefficents_all_1s__weight_list_still_all_1s(self):

        pixel_signals = np.array([0.25, 0.5, 0.75])

        weight_list = aa.util.regularization.adaptive_regularization_weights_from(
            inner_coefficient=1.0, outer_coefficient=1.0, pixel_signals=pixel_signals
        )

        assert (weight_list == np.array([1.0, 1.0, 1.0])).all()

    def test__pixel_signals_vary__coefficents_1_and_0__weight_list_are_pixel_signals_squared(
        self,
    ):

        pixel_signals = np.array([0.25, 0.5, 0.75])

        weight_list = aa.util.regularization.adaptive_regularization_weights_from(
            inner_coefficient=1.0, outer_coefficient=0.0, pixel_signals=pixel_signals
        )

        assert (weight_list == np.array([0.25 ** 2.0, 0.5 ** 2.0, 0.75 ** 2.0])).all()

    def test__pixel_signals_vary__coefficents_0_and_1__weight_list_are_1_minus_pixel_signals_squared(
        self,
    ):

        pixel_signals = np.array([0.25, 0.5, 0.75])

        weight_list = aa.util.regularization.adaptive_regularization_weights_from(
            inner_coefficient=0.0, outer_coefficient=1.0, pixel_signals=pixel_signals
        )

        assert (weight_list == np.array([0.75 ** 2.0, 0.5 ** 2.0, 0.25 ** 2.0])).all()


class TestWeightedRegularizationMatrix:
    def test__1_b_matrix_size_4x4__weight_list_all_1s__makes_correct_regularization_matrix(
        self,
    ):

        neighbors = np.array([[2], [3], [0], [1]])

        neighbors_sizes = np.array([1, 1, 1, 1])

        test_b_matrix = np.array(
            [[-1, 0, 1, 0], [0, -1, 0, 1], [1, 0, -1, 0], [0, 1, 0, -1]]
        )

        test_regularization_matrix = np.matmul(test_b_matrix.T, test_b_matrix)

        regularization_weights = np.ones((4,))

        regularization_matrix = aa.util.regularization.weighted_regularization_matrix_from(
            regularization_weights=regularization_weights,
            neighbors=neighbors,
            neighbors_sizes=neighbors_sizes,
        )

        assert regularization_matrix == pytest.approx(
            test_regularization_matrix, 1.0e-4
        )

    def test__2_b_matrices_size_3x3__weight_list_all_1s__makes_correct_regularization_matrix(
        self,
    ):
        # Here, we define the neighbors first here and make the B matrices based on them.

        # You'll notice that actually, the B Matrix doesn't have to have the -1's going down the diagonal and we
        # don't have to have as many B matrices as we do the pix pixel with the most  vertices. We can combine
        # the rows of each B matrix wherever we like ;0.

        neighbors = np.array([[1, 2], [0, -1], [0, -1]])

        neighbors_sizes = np.array([2, 1, 1])

        test_b_matrix_1 = np.array(
            [[-1, 1, 0], [-1, 0, 1], [1, -1, 0]]  # Pair 1  # Pair 2
        )  # Pair 1 flip

        test_regularization_matrix_1 = np.matmul(test_b_matrix_1.T, test_b_matrix_1)

        test_b_matrix_2 = np.array([[1, 0, -1], [0, 0, 0], [0, 0, 0]])  # Pair 2 flip

        test_regularization_matrix_2 = np.matmul(test_b_matrix_2.T, test_b_matrix_2)

        test_regularization_matrix = (
            test_regularization_matrix_1 + test_regularization_matrix_2
        )

        regularization_weights = np.ones((3))

        regularization_matrix = aa.util.regularization.weighted_regularization_matrix_from(
            regularization_weights=regularization_weights,
            neighbors=neighbors,
            neighbors_sizes=neighbors_sizes,
        )

        assert regularization_matrix == pytest.approx(
            test_regularization_matrix, 1.0e-4
        )

    def test__2_b_matrices_size_4x4__weight_list_all_1s__makes_correct_regularization_matrix(
        self,
    ):

        test_b_matrix_1 = np.array(
            [[-1, 1, 0, 0], [0, -1, 1, 0], [0, 0, -1, 1], [1, 0, 0, -1]]
        )

        test_regularization_matrix_1 = np.matmul(test_b_matrix_1.T, test_b_matrix_1)

        test_b_matrix_2 = np.array(
            [[-1, 0, 0, 1], [1, -1, 0, 0], [0, 1, -1, 0], [0, 0, 1, -1]]
        )

        test_regularization_matrix_2 = np.matmul(test_b_matrix_2.T, test_b_matrix_2)

        test_regularization_matrix = (
            test_regularization_matrix_1 + test_regularization_matrix_2
        )

        neighbors = np.array([[1, 3], [0, 2], [1, 3], [0, 2]])

        neighbors_sizes = np.array([2, 2, 2, 2])

        regularization_weights = np.ones((4,))

        regularization_matrix = aa.util.regularization.weighted_regularization_matrix_from(
            regularization_weights=regularization_weights,
            neighbors=neighbors,
            neighbors_sizes=neighbors_sizes,
        )

        assert regularization_matrix == pytest.approx(
            test_regularization_matrix, 1.0e-4
        )

    def test__4_b_matrices_size_6x6__weight_list_all_1s__makes_correct_regularization_matrix(
        self,
    ):
        # Again, lets exploit the freedom we have when setting up our B matrices to make matching it to pairs a
        # lot less Stressful.

        neighbors = np.array(
            [
                [2, 3, 4, -1],
                [2, 5, -1, -1],
                [0, 1, 3, 5],
                [0, 2, -1, -1],
                [5, 0, -1, -1],
                [4, 1, 2, -1],
            ]
        )

        neighbors_sizes = np.array([3, 2, 4, 2, 2, 3])

        test_b_matrix_1 = np.array(
            [
                [-1, 0, 1, 0, 0, 0],  # Pair 1
                [0, -1, 1, 0, 0, 0],  # Pair 2
                [-1, 0, 0, 1, 0, 0],  # Pair 3
                [0, 0, 0, 0, -1, 1],  # Pair 4
                [0, -1, 0, 0, 0, 1],  # Pair 5
                [-1, 0, 0, 0, 1, 0],
            ]
        )  # Pair 6

        test_regularization_matrix_1 = np.matmul(test_b_matrix_1.T, test_b_matrix_1)

        test_b_matrix_2 = np.array(
            [
                [0, 0, -1, 1, 0, 0],  # Pair 7
                [0, 0, -1, 0, 0, 1],  # Pair 8
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
            ]
        )

        test_regularization_matrix_2 = np.matmul(test_b_matrix_2.T, test_b_matrix_2)

        test_b_matrix_3 = np.array(
            [
                [1, 0, -1, 0, 0, 0],  # Pair 1 flip
                [0, 1, -1, 0, 0, 0],  # Pair 2 flip
                [1, 0, 0, -1, 0, 0],  # Pair 3 flip
                [0, 0, 0, 0, 1, -1],  # Pair 4 flip
                [0, 1, 0, 0, 0, -1],  # Pair 5 flip
                [1, 0, 0, 0, -1, 0],
            ]
        )  # Pair 6 flip

        test_regularization_matrix_3 = np.matmul(test_b_matrix_3.T, test_b_matrix_3)

        test_b_matrix_4 = np.array(
            [
                [0, 0, 1, -1, 0, 0],  # Pair 7 flip
                [0, 0, 1, 0, 0, -1],  # Pair 8 flip
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
            ]
        )

        test_regularization_matrix_4 = np.matmul(test_b_matrix_4.T, test_b_matrix_4)

        test_regularization_matrix = (
            test_regularization_matrix_1
            + test_regularization_matrix_2
            + test_regularization_matrix_3
            + +test_regularization_matrix_4
        )

        regularization_weights = np.ones((6))

        regularization_matrix = aa.util.regularization.weighted_regularization_matrix_from(
            regularization_weights=regularization_weights,
            neighbors=neighbors,
            neighbors_sizes=neighbors_sizes,
        )

        assert regularization_matrix == pytest.approx(
            test_regularization_matrix, 1.0e-4
        )

    def test__2_b_matrices_size_4x4_models_regularization_weights__makes_correct_regularization_matrix(
        self,
    ):
        # Simple case, where we have just one regularization direction, regularizing pixel 0 -> 1 and 1 -> 2.

        # This means our B matrix is:

        # [-1, 1, 0]
        # [0, -1, 1]
        # [0, 0, -1]

        # Regularization Matrix, H = B * B.T.I can

        regularization_weights = np.array([2.0, 4.0, 1.0, 8.0])

        test_b_matrix_1 = np.array(
            [[-2, 2, 0, 0], [-2, 0, 2, 0], [0, -4, 4, 0], [0, -4, 0, 4]]
        )

        test_b_matrix_2 = np.array(
            [[4, -4, 0, 0], [1, 0, -1, 0], [0, 1, -1, 0], [0, 8, 0, -8]]
        )

        test_regularization_matrix_1 = np.matmul(test_b_matrix_1.T, test_b_matrix_1)
        test_regularization_matrix_2 = np.matmul(test_b_matrix_2.T, test_b_matrix_2)

        test_regularization_matrix = (
            test_regularization_matrix_1 + test_regularization_matrix_2
        )

        neighbors = np.array(
            [[1, 2, -1, -1], [0, 2, 3, -1], [0, 1, -1, -1], [1, -1, -1, -1]]
        )

        neighbors_sizes = np.array([2, 3, 2, 1])

        regularization_matrix = aa.util.regularization.weighted_regularization_matrix_from(
            regularization_weights=regularization_weights,
            neighbors=neighbors,
            neighbors_sizes=neighbors_sizes,
        )

        assert regularization_matrix == pytest.approx(
            test_regularization_matrix, 1.0e-4
        )

    def test__4_b_matrices_size_6x6_with_regularization_weights__makes_correct_regularization_matrix(
        self,
    ):

        neighbors = np.array(
            [
                [1, 4, -1, -1],
                [2, 4, 0, -1],
                [3, 4, 5, 1],
                [5, 2, -1, -1],
                [5, 0, 1, 2],
                [2, 3, 4, -1],
            ]
        )

        neighbors_sizes = np.array([2, 3, 4, 2, 4, 3])
        regularization_weights = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])

        # I'm inputting the regularization weight_list directly thiss time, as it'd be a pain to multiply with a
        # loop.

        test_b_matrix_1 = np.array(
            [
                [-1, 1, 0, 0, 0, 0],  # Pair 1
                [-1, 0, 0, 0, 1, 0],  # Pair 2
                [0, -2, 2, 0, 0, 0],  # Pair 3
                [0, -2, 0, 0, 2, 0],  # Pair 4
                [0, 0, -3, 3, 0, 0],  # Pair 5
                [0, 0, -3, 0, 3, 0],
            ]
        )  # Pair 6

        test_b_matrix_2 = np.array(
            [
                [0, 0, -3, 0, 0, 3],  # Pair 7
                [0, 0, 0, -4, 0, 4],  # Pair 8
                [0, 0, 0, 0, -5, 5],  # Pair 9
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
            ]
        )

        # Now do the same pairs but with the regularization direction and weight_list swapped.

        test_b_matrix_3 = np.array(
            [
                [2, -2, 0, 0, 0, 0],  # Pair 1
                [5, 0, 0, 0, -5, 0],  # Pair 2
                [0, 3, -3, 0, 0, 0],  # Pair 3
                [0, 5, 0, 0, -5, 0],  # Pair 4
                [0, 0, 4, -4, 0, 0],  # Pair 5
                [0, 0, 5, 0, -5, 0],
            ]
        )  # Pair 6

        test_b_matrix_4 = np.array(
            [
                [0, 0, 6, 0, 0, -6],  # Pair 7
                [0, 0, 0, 6, 0, -6],  # Pair 8
                [0, 0, 0, 0, 6, -6],  # Pair 9
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
            ]
        )

        test_regularization_matrix_1 = np.matmul(test_b_matrix_1.T, test_b_matrix_1)
        test_regularization_matrix_2 = np.matmul(test_b_matrix_2.T, test_b_matrix_2)
        test_regularization_matrix_3 = np.matmul(test_b_matrix_3.T, test_b_matrix_3)
        test_regularization_matrix_4 = np.matmul(test_b_matrix_4.T, test_b_matrix_4)

        test_regularization_matrix = (
            test_regularization_matrix_1
            + test_regularization_matrix_2
            + test_regularization_matrix_3
            + test_regularization_matrix_4
        )

        regularization_matrix = aa.util.regularization.weighted_regularization_matrix_from(
            regularization_weights=regularization_weights,
            neighbors=neighbors,
            neighbors_sizes=neighbors_sizes,
        )

        assert regularization_matrix == pytest.approx(
            test_regularization_matrix, 1.0e-4
        )


class TestSplitted:
    def test__constant_pixel_splitted_regularization_matrix(self):
        splitted_mappings = np.array(
            [
                [0, -1, -1, -1, -1],
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
                [1, 4, -1, -1, -1],
            ]
        )

        splitted_sizes = np.sum(splitted_mappings != -1, axis=1)

        splitted_weights = np.array(
            [
                [1.0, 0.0, 0.0, 0.0, 0.0],
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
                [0.57, 0.43, 0.0, 0.0, 0.0],
            ]
        )

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

        pixels = int(len(splitted_mappings) / 4)

        regularization_matrix = aa.util.regularization.pixel_splitted_regularization_matrix_from(
            regularization_weights=np.full(fill_value=1.0, shape=(pixels,)),
            splitted_mappings=splitted_mappings,
            splitted_sizes=splitted_sizes,
            splitted_weights=splitted_weights,
        )

        assert pytest.approx(regularization_matrix[0], 1e-4) == np.array(
            [4.58, -0.6, -2.45, -1.26, -0.27]
        )

        regularization_weights = np.array([2.0, 4.0, 2.0, 2.0, 2.0])

        regularization_matrix = aa.util.regularization.pixel_splitted_regularization_matrix_from(
            regularization_weights=regularization_weights,
            splitted_mappings=splitted_mappings,
            splitted_sizes=splitted_sizes,
            splitted_weights=splitted_weights,
        )

        assert pytest.approx(regularization_matrix[0], 1e-4) == np.array(
            [19.4, -6, -9.8, -2.88, -0.72]
        )
