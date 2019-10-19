import autoarray as aa
import numpy as np


class TestBinnedPaddingArray:
    def test__bin_up_factor_is_1__array_2d_does_not_change_shape(self):

        array_2d = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])

        array_2d_padded = aa.util.binning.padded_binning_array_2d_from_array_2d(
            array_2d=array_2d, bin_up_factor=1
        )

        assert (array_2d_padded == array_2d).all()

    def test__bin_up_factor_gives_no_remainder__array_2d_does_not_change_shape(self):

        array_2d = np.ones(shape=(6, 6))

        array_2d_padded = aa.util.binning.padded_binning_array_2d_from_array_2d(
            array_2d=array_2d, bin_up_factor=2
        )
        assert array_2d_padded.shape == (6, 6)

        array_2d_padded = aa.util.binning.padded_binning_array_2d_from_array_2d(
            array_2d=array_2d, bin_up_factor=3
        )
        assert array_2d_padded.shape == (6, 6)

        array_2d_padded = aa.util.binning.padded_binning_array_2d_from_array_2d(
            array_2d=array_2d, bin_up_factor=6
        )
        assert array_2d_padded.shape == (6, 6)

        array_2d = np.ones(shape=(8, 8))

        array_2d_padded = aa.util.binning.padded_binning_array_2d_from_array_2d(
            array_2d=array_2d, bin_up_factor=2
        )
        assert array_2d_padded.shape == (8, 8)

        array_2d_padded = aa.util.binning.padded_binning_array_2d_from_array_2d(
            array_2d=array_2d, bin_up_factor=4
        )
        assert array_2d_padded.shape == (8, 8)

        array_2d_padded = aa.util.binning.padded_binning_array_2d_from_array_2d(
            array_2d=array_2d, bin_up_factor=8
        )
        assert array_2d_padded.shape == (8, 8)

        array_2d = np.ones(shape=(9, 9))

        array_2d_padded = aa.util.binning.padded_binning_array_2d_from_array_2d(
            array_2d=array_2d, bin_up_factor=3
        )
        assert array_2d_padded.shape == (9, 9)

        array_2d = np.ones(shape=(16, 16))

        array_2d_padded = aa.util.binning.padded_binning_array_2d_from_array_2d(
            array_2d=array_2d, bin_up_factor=2
        )
        assert array_2d_padded.shape == (16, 16)

        array_2d_padded = aa.util.binning.padded_binning_array_2d_from_array_2d(
            array_2d=array_2d, bin_up_factor=4
        )
        assert array_2d_padded.shape == (16, 16)

        array_2d_padded = aa.util.binning.padded_binning_array_2d_from_array_2d(
            array_2d=array_2d, bin_up_factor=8
        )
        assert array_2d_padded.shape == (16, 16)

        array_2d = np.ones(shape=(12, 16))

        array_2d_padded = aa.util.binning.padded_binning_array_2d_from_array_2d(
            array_2d=array_2d, bin_up_factor=2
        )
        assert array_2d_padded.shape == (12, 16)

        array_2d_padded = aa.util.binning.padded_binning_array_2d_from_array_2d(
            array_2d=array_2d, bin_up_factor=4
        )
        assert array_2d_padded.shape == (12, 16)

        array_2d = np.ones(shape=(16, 12))

        array_2d_padded = aa.util.binning.padded_binning_array_2d_from_array_2d(
            array_2d=array_2d, bin_up_factor=2
        )
        assert array_2d_padded.shape == (16, 12)

        array_2d_padded = aa.util.binning.padded_binning_array_2d_from_array_2d(
            array_2d=array_2d, bin_up_factor=4
        )
        assert array_2d_padded.shape == (16, 12)

    def test__bin_up_factor_gives_remainder__array_2d_padded_to_give_no_remainder(self):

        array_2d = np.ones(shape=(6, 6))

        array_2d_padded = aa.util.binning.padded_binning_array_2d_from_array_2d(
            array_2d=array_2d, bin_up_factor=4
        )
        assert array_2d_padded.shape == (8, 8)

        array_2d_padded = aa.util.binning.padded_binning_array_2d_from_array_2d(
            array_2d=array_2d, bin_up_factor=5
        )
        assert array_2d_padded.shape == (10, 10)

        array_2d_padded = aa.util.binning.padded_binning_array_2d_from_array_2d(
            array_2d=array_2d, bin_up_factor=7
        )
        assert array_2d_padded.shape == (7, 7)

        array_2d = np.ones(shape=(10, 10))

        array_2d_padded = aa.util.binning.padded_binning_array_2d_from_array_2d(
            array_2d=array_2d, bin_up_factor=3
        )
        assert array_2d_padded.shape == (12, 12)

        array_2d_padded = aa.util.binning.padded_binning_array_2d_from_array_2d(
            array_2d=array_2d, bin_up_factor=4
        )
        assert array_2d_padded.shape == (12, 12)

        array_2d_padded = aa.util.binning.padded_binning_array_2d_from_array_2d(
            array_2d=array_2d, bin_up_factor=6
        )
        assert array_2d_padded.shape == (12, 12)

        array_2d_padded = aa.util.binning.padded_binning_array_2d_from_array_2d(
            array_2d=array_2d, bin_up_factor=7
        )
        assert array_2d_padded.shape == (14, 14)

        array_2d = np.ones(shape=(7, 10))

        array_2d_padded = aa.util.binning.padded_binning_array_2d_from_array_2d(
            array_2d=array_2d, bin_up_factor=3
        )
        assert array_2d_padded.shape == (9, 12)

        array_2d_padded = aa.util.binning.padded_binning_array_2d_from_array_2d(
            array_2d=array_2d, bin_up_factor=5
        )
        assert array_2d_padded.shape == (10, 10)

        array_2d_padded = aa.util.binning.padded_binning_array_2d_from_array_2d(
            array_2d=array_2d, bin_up_factor=7
        )
        assert array_2d_padded.shape == (7, 14)

    def test__padding_using_arrays_and_not_shapes(self):

        array_2d = np.ones(shape=(4, 4))
        array_2d[1, 1] = 2.0

        array_2d_padded = aa.util.binning.padded_binning_array_2d_from_array_2d(
            array_2d=array_2d, bin_up_factor=3
        )
        assert array_2d_padded.shape == (6, 6)
        assert (
            array_2d_padded
            == np.array(
                [
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 1.0, 1.0, 1.0, 0.0],
                    [0.0, 1.0, 2.0, 1.0, 1.0, 0.0],
                    [0.0, 1.0, 1.0, 1.0, 1.0, 0.0],
                    [0.0, 1.0, 1.0, 1.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                ]
            )
        ).all()

        array_2d_padded = aa.util.binning.padded_binning_array_2d_from_array_2d(
            array_2d=array_2d, bin_up_factor=5
        )
        assert array_2d_padded.shape == (5, 5)
        assert (
            array_2d_padded
            == np.array(
                [
                    [1.0, 1.0, 1.0, 1.0, 0.0],
                    [1.0, 2.0, 1.0, 1.0, 0.0],
                    [1.0, 1.0, 1.0, 1.0, 0.0],
                    [1.0, 1.0, 1.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0],
                ]
            )
        ).all()

        array_2d = np.ones(shape=(2, 3))
        array_2d[1, 1] = 2.0
        array_2d_padded = aa.util.binning.padded_binning_array_2d_from_array_2d(
            array_2d=array_2d, bin_up_factor=2
        )
        assert array_2d_padded.shape == (2, 4)
        assert (
            array_2d_padded == np.array([[0.0, 1.0, 1.0, 1.0], [0.0, 1.0, 2.0, 1.0]])
        ).all()

        array_2d = np.ones(shape=(3, 2))
        array_2d[1, 1] = 2.0
        array_2d_padded = aa.util.binning.padded_binning_array_2d_from_array_2d(
            array_2d=array_2d, bin_up_factor=2
        )
        assert array_2d_padded.shape == (4, 2)
        assert (
            array_2d_padded
            == np.array([[0.0, 0.0], [1.0, 1.0], [1.0, 2.0], [1.0, 1.0]])
        ).all()


class TestBinnedArrays2d:
    def test__bin_using_mean__array_4x4_to_2x2__uses_mean_correctly(self):

        array_2d = np.array(
            [
                [1.0, 1.0, 2.0, 2.0],
                [1.0, 1.0, 2.0, 2.0],
                [3.0, 3.0, 4.0, 4.0],
                [3.0, 3.0, 4.0, 4.0],
            ]
        )

        binned_array_2d = aa.util.binning.bin_array_2d_via_mean(
            array_2d=array_2d, bin_up_factor=2
        )

        assert (binned_array_2d == np.array([[1.0, 2.0], [3.0, 4.0]])).all()

        array_2d = np.array(
            [
                [1.0, 2.0, 2.0, 2.0],
                [1.0, 6.0, 2.0, 10.0],
                [9.0, 3.0, 4.0, 0.0],
                [3.0, 3.0, 4.0, 4.0],
            ]
        )

        binned_array_2d = aa.util.binning.bin_array_2d_via_mean(
            array_2d=array_2d, bin_up_factor=2
        )

        assert (binned_array_2d == np.array([[2.5, 4.0], [4.5, 3.0]])).all()

    def test__bin_using_mean__array_6x3_to_2x1_and_3x6_to_1x2__uses_mean_correctly(
        self
    ):

        array_2d = np.array(
            [
                [1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0],
                [2.0, 2.0, 2.0],
                [2.0, 2.0, 2.0],
                [2.0, 2.0, 2.0],
            ]
        )

        binned_array_2d = aa.util.binning.bin_array_2d_via_mean(
            array_2d=array_2d, bin_up_factor=3
        )

        assert (binned_array_2d == np.array([[1.0], [2.0]])).all()

        array_2d = np.array(
            [
                [1.0, 1.0, 1.0],
                [1.0, 10.0, 1.0],
                [1.0, 1.0, 1.0],
                [2.0, 11.0, 2.0],
                [2.0, 2.0, 2.0],
                [2.0, 2.0, 2.0],
            ]
        )

        binned_array_2d = aa.util.binning.bin_array_2d_via_mean(
            array_2d=array_2d, bin_up_factor=3
        )
        assert (binned_array_2d == np.array([[2.0], [3.0]])).all()

        array_2d = np.array(
            [
                [1.0, 1.0, 1.0, 2.0, 2.0, 2.0],
                [1.0, 1.0, 1.0, 2.0, 2.0, 2.0],
                [1.0, 1.0, 1.0, 2.0, 2.0, 2.0],
            ]
        )

        binned_array_2d = aa.util.binning.bin_array_2d_via_mean(
            array_2d=array_2d, bin_up_factor=3
        )
        assert (binned_array_2d == np.array([[1.0, 2.0]])).all()

        array_2d = np.array(
            [
                [1.0, 1.0, 1.0, 2.0, 2.0, 2.0],
                [1.0, 10.0, 1.0, 11.0, 2.0, 2.0],
                [1.0, 1.0, 1.0, 2.0, 2.0, 2.0],
            ]
        )

        binned_array_2d = aa.util.binning.bin_array_2d_via_mean(
            array_2d=array_2d, bin_up_factor=3
        )
        assert (binned_array_2d == np.array([[2.0, 3.0]])).all()

    def test__bin_using_mean__bin_includes_padding_image_with_zeros(self):

        # Padded array:

        # [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        #  [0.0, 1.0, 1.0, 1.0, 1.0, 0.0],
        #  [0.0, 1.0, 2.0, 1.0, 1.0, 0.0],
        #  [0.0, 1.0, 1.0, 1.0, 1.0, 0.0],
        #  [0.0, 1.0, 1.0, 1.0, 1.0, 0.0],
        #  [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]

        array_2d = np.ones(shape=(4, 4))
        array_2d[1, 1] = 2.0
        binned_array_2d = aa.util.binning.bin_array_2d_via_mean(
            array_2d=array_2d, bin_up_factor=3
        )
        assert (
            binned_array_2d
            == np.array([[(5.0 / 9.0), (4.0 / 9.0)], [(4.0 / 9.0), (4.0 / 9.0)]])
        ).all()

        # Padded Array:

        # np.array([[0.0, 1.0, 1.0, 1.0],
        #           [0.0, 1.0, 2.0, 1.0]]

        array_2d = np.ones(shape=(2, 3))
        array_2d[1, 1] = 2.0
        binned_2d_array = aa.util.binning.bin_array_2d_via_mean(
            array_2d=array_2d, bin_up_factor=2
        )
        assert (binned_2d_array == np.array([[0.5, 1.25]])).all()

    def test__bin_using_quadrature__array_4x4_to_2x2__uses_quadrature_correctly(self):

        array_2d = np.array(
            [
                [1.0, 1.0, 2.0, 2.0],
                [1.0, 1.0, 2.0, 2.0],
                [3.0, 3.0, 4.0, 4.0],
                [3.0, 3.0, 4.0, 4.0],
            ]
        )

        binned_array_2d = aa.util.binning.bin_array_2d_via_quadrature(
            array_2d=array_2d, bin_up_factor=2
        )

        assert (
            binned_array_2d
            == np.array(
                [
                    [np.sqrt(4.0) / 4.0, np.sqrt(16.0) / 4.0],
                    [np.sqrt(36.0) / 4.0, np.sqrt(64.0) / 4.0],
                ]
            )
        ).all()

        array_2d = np.array(
            [
                [1.0, 2.0, 2.0, 2.0],
                [1.0, 6.0, 2.0, 10.0],
                [9.0, 3.0, 4.0, 0.0],
                [3.0, 3.0, 4.0, 4.0],
            ]
        )

        binned_array_2d = aa.util.binning.bin_array_2d_via_quadrature(
            array_2d=array_2d, bin_up_factor=2
        )

        assert (
            binned_array_2d
            == np.array(
                [
                    [np.sqrt(42.0) / 4.0, np.sqrt(112.0) / 4.0],
                    [np.sqrt(108.0) / 4.0, np.sqrt(48.0) / 4.0],
                ]
            )
        ).all()

    def test__bin_using_quadrature__array_6x3_to_2x1_and_3x6_to_1x2__uses_quadrature_correctly(
        self
    ):

        array_2d = np.array(
            [
                [1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0],
                [2.0, 2.0, 2.0],
                [2.0, 2.0, 2.0],
                [2.0, 2.0, 2.0],
            ]
        )

        binned_array_2d = aa.util.binning.bin_array_2d_via_quadrature(
            array_2d=array_2d, bin_up_factor=3
        )

        assert (
            binned_array_2d == np.array([[np.sqrt(9.0) / 9.0], [np.sqrt(36.0) / 9.0]])
        ).all()

        array_2d = np.array(
            [
                [1.0, 1.0, 1.0],
                [1.0, 10.0, 1.0],
                [1.0, 1.0, 1.0],
                [2.0, 4.0, 2.0],
                [2.0, 2.0, 2.0],
                [2.0, 2.0, 2.0],
            ]
        )

        binned_array_2d = aa.util.binning.bin_array_2d_via_quadrature(
            array_2d=array_2d, bin_up_factor=3
        )
        assert (
            binned_array_2d == np.array([[np.sqrt(108.0) / 9.0], [np.sqrt(48.0) / 9.0]])
        ).all()

        array_2d = np.array(
            [
                [1.0, 1.0, 1.0, 2.0, 2.0, 2.0],
                [1.0, 1.0, 1.0, 2.0, 2.0, 2.0],
                [1.0, 1.0, 1.0, 2.0, 2.0, 2.0],
            ]
        )

        binned_array_2d = aa.util.binning.bin_array_2d_via_quadrature(
            array_2d=array_2d, bin_up_factor=3
        )
        assert (
            binned_array_2d == np.array([[np.sqrt(9.0) / 9.0, np.sqrt(36.0) / 9.0]])
        ).all()

        array_2d = np.array(
            [
                [1.0, 1.0, 1.0, 2.0, 2.0, 2.0],
                [1.0, 10.0, 1.0, 4.0, 2.0, 2.0],
                [1.0, 1.0, 1.0, 2.0, 2.0, 2.0],
            ]
        )

        binned_array_2d = aa.util.binning.bin_array_2d_via_quadrature(
            array_2d=array_2d, bin_up_factor=3
        )
        assert (
            binned_array_2d == np.array([[np.sqrt(108.0) / 9.0, np.sqrt(48.0) / 9.0]])
        ).all()

    def test__bin_using_quadrature__bin_includes_padding_image_with_zeros(self):

        # Padded array:

        # [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        #  [0.0, 1.0, 1.0, 1.0, 1.0, 0.0],
        #  [0.0, 1.0, 2.0, 1.0, 1.0, 0.0],
        #  [0.0, 1.0, 1.0, 1.0, 1.0, 0.0],
        #  [0.0, 1.0, 1.0, 1.0, 1.0, 0.0],
        #  [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]

        array_2d = np.ones(shape=(4, 4))
        array_2d[1, 1] = 2.0
        binned_array_2d = aa.util.binning.bin_array_2d_via_quadrature(
            array_2d=array_2d, bin_up_factor=3
        )
        assert (
            binned_array_2d
            == np.array(
                [
                    [np.sqrt(7.0) / 9.0, np.sqrt(4.0) / 9.0],
                    [np.sqrt(4.0) / 9.0, np.sqrt(4.0) / 9.0],
                ]
            )
        ).all()

        # Padded Array:

        # np.array([[0.0, 1.0, 1.0, 1.0],
        #           [0.0, 1.0, 2.0, 1.0]]

        array_2d = np.ones(shape=(2, 3))
        array_2d[1, 1] = 2.0
        binned_2d_array = aa.util.binning.bin_array_2d_via_quadrature(
            array_2d=array_2d, bin_up_factor=2
        )
        assert (
            binned_2d_array == np.array([[np.sqrt(2.0) / 4.0, np.sqrt(7.0) / 4.0]])
        ).all()

    def test__bin_using_sum__array_4x4_to_2x2__uses_sum_correctly(self):

        array_2d = np.array(
            [
                [1.0, 1.0, 2.0, 2.0],
                [1.0, 1.0, 2.0, 2.0],
                [3.0, 3.0, 4.0, 4.0],
                [3.0, 3.0, 4.0, 4.0],
            ]
        )

        binned_array_2d = aa.util.binning.bin_array_2d_via_sum(
            array_2d=array_2d, bin_up_factor=2
        )

        assert (binned_array_2d == np.array([[4.0, 8.0], [12.0, 16.0]])).all()

        array_2d = np.array(
            [
                [1.0, 2.0, 2.0, 2.0],
                [1.0, 6.0, 2.0, 10.0],
                [9.0, 3.0, 4.0, 0.0],
                [3.0, 3.0, 4.0, 4.0],
            ]
        )

        binned_array_2d = aa.util.binning.bin_array_2d_via_sum(
            array_2d=array_2d, bin_up_factor=2
        )

        assert (binned_array_2d == np.array([[10.0, 16.0], [18.0, 12.0]])).all()

    def test__bin_using_sum__array_6x3_to_2x1_and_3x6_to_1x2__uses_sum_correctly(self):

        array_2d = np.array(
            [
                [1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0],
                [2.0, 2.0, 2.0],
                [2.0, 2.0, 2.0],
                [2.0, 2.0, 2.0],
            ]
        )

        binned_array_2d = aa.util.binning.bin_array_2d_via_sum(
            array_2d=array_2d, bin_up_factor=3
        )

        assert (binned_array_2d == np.array([[9.0], [18.0]])).all()

        array_2d = np.array(
            [
                [1.0, 1.0, 1.0],
                [1.0, 10.0, 1.0],
                [1.0, 1.0, 1.0],
                [2.0, 11.0, 2.0],
                [2.0, 2.0, 2.0],
                [2.0, 2.0, 2.0],
            ]
        )

        binned_array_2d = aa.util.binning.bin_array_2d_via_sum(
            array_2d=array_2d, bin_up_factor=3
        )
        assert (binned_array_2d == np.array([[18.0], [27.0]])).all()

        array_2d = np.array(
            [
                [1.0, 1.0, 1.0, 2.0, 2.0, 2.0],
                [1.0, 1.0, 1.0, 2.0, 2.0, 2.0],
                [1.0, 1.0, 1.0, 2.0, 2.0, 2.0],
            ]
        )

        binned_array_2d = aa.util.binning.bin_array_2d_via_sum(
            array_2d=array_2d, bin_up_factor=3
        )
        assert (binned_array_2d == np.array([[9.0, 18.0]])).all()

        array_2d = np.array(
            [
                [1.0, 1.0, 1.0, 2.0, 2.0, 2.0],
                [1.0, 10.0, 1.0, 11.0, 2.0, 2.0],
                [1.0, 1.0, 1.0, 2.0, 2.0, 2.0],
            ]
        )

        binned_array_2d = aa.util.binning.bin_array_2d_via_sum(
            array_2d=array_2d, bin_up_factor=3
        )
        assert (binned_array_2d == np.array([[18.0, 27.0]])).all()

    def test__bin_using_sum__bin_includes_padding_image_with_zeros(self):

        # Padded array:

        # [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        #  [0.0, 1.0, 1.0, 1.0, 1.0, 0.0],
        #  [0.0, 1.0, 2.0, 1.0, 1.0, 0.0],
        #  [0.0, 1.0, 1.0, 1.0, 1.0, 0.0],
        #  [0.0, 1.0, 1.0, 1.0, 1.0, 0.0],
        #  [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]

        array_2d = np.ones(shape=(4, 4))
        array_2d[1, 1] = 2.0
        binned_array_2d = aa.util.binning.bin_array_2d_via_sum(
            array_2d=array_2d, bin_up_factor=3
        )
        assert (binned_array_2d == np.array([[5.0, 4.0], [4.0, 4.0]])).all()

        # Padded Array:

        # np.array([[0.0, 1.0, 1.0, 1.0],
        #           [0.0, 1.0, 2.0, 1.0]]

        array_2d = np.ones(shape=(2, 3))
        array_2d[1, 1] = 2.0
        binned_2d_array = aa.util.binning.bin_array_2d_via_sum(
            array_2d=array_2d, bin_up_factor=2
        )
        assert (binned_2d_array == np.array([[2.0, 5.0]])).all()


class TestBinUpMask2d:
    def test__mask_4x4_to_2x2__creates_correct_binned_up_mask(self):

        mask_2d = np.array(
            [
                [True, False, True, True],
                [True, True, True, True],
                [True, True, False, False],
                [False, True, True, True],
            ]
        )

        binned_mask = aa.util.binning.bin_mask_2d(
            mask_2d=mask_2d, bin_up_factor=2
        )

        assert (binned_mask == np.array([[False, True], [False, False]])).all()

        mask_2d = np.array(
            [
                [True, True, True, True],
                [True, True, True, True],
                [True, True, False, False],
                [True, True, True, True],
            ]
        )

        binned_mask = aa.util.binning.bin_mask_2d(
            mask_2d=mask_2d, bin_up_factor=2
        )

        assert (binned_mask == np.array([[True, True], [True, False]])).all()

    def test__mask_6x3_to_2x1_and_3x6_to_1x2__sets_up_correct_mask(self):

        mask_2d = np.array(
            [
                [True, True, True],
                [True, True, True],
                [True, True, True],
                [True, True, True],
                [True, True, True],
                [True, True, True],
            ]
        )

        binned_mask = aa.util.binning.bin_mask_2d(
            mask_2d=mask_2d, bin_up_factor=3
        )

        assert (binned_mask == np.array([[True], [True]])).all()

        mask_2d = np.array(
            [
                [True, True, True],
                [True, True, False],
                [True, True, True],
                [True, True, True],
                [True, True, True],
                [True, True, True],
            ]
        )

        binned_mask = aa.util.binning.bin_mask_2d(
            mask_2d=mask_2d, bin_up_factor=3
        )
        assert (binned_mask == np.array([[False], [True]])).all()

        mask_2d = np.array(
            [
                [True, True, True, True, True, True],
                [True, True, True, True, True, True],
                [True, True, True, True, True, True],
            ]
        )

        binned_mask = aa.util.binning.bin_mask_2d(
            mask_2d=mask_2d, bin_up_factor=3
        )
        assert (binned_mask == np.array([[True, True]])).all()

        mask_2d = np.array(
            [
                [True, True, True, True, True, True],
                [True, True, True, True, True, True],
                [True, True, True, True, True, False],
            ]
        )

        binned_mask = aa.util.binning.bin_mask_2d(
            mask_2d=mask_2d, bin_up_factor=3
        )
        assert (binned_mask == np.array([[True, False]])).all()

    def test__bin_includes_padding_image_with_zeros(self):
        # Padded mask:

        # [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        #  [0.0, 1.0, 1.0, 1.0, 1.0, 0.0],
        #  [0.0, 1.0, 2.0, 1.0, 1.0, 0.0],
        #  [0.0, 1.0, 1.0, 1.0, 1.0, 0.0],
        #  [0.0, 1.0, 1.0, 1.0, 1.0, 0.0],
        #  [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]

        mask_2d = np.full(shape=(4, 4), fill_value=True)
        mask_2d[1, 1] = False
        mask_2d[3, 3] = False
        binned_mask = aa.util.binning.bin_mask_2d(
            mask_2d=mask_2d, bin_up_factor=3
        )
        assert (binned_mask == np.array([[False, True], [True, False]])).all()

        # Padded Array:

        # np.array([[0.0, 1.0, 1.0, 1.0],
        #           [0.0, 1.0, 2.0, 1.0]]

        mask_2d = np.full(shape=(2, 3), fill_value=True)
        mask_2d[1, 2] = False
        binned_2d_mask = aa.util.binning.bin_mask_2d(
            mask_2d=mask_2d, bin_up_factor=2
        )
        assert (binned_2d_mask == np.array([[True, False]])).all()


class TestMask1DIndexForPaddedMask2dIndex(object):
    def test__no_padding__mask_is_full_of_false__returns_indexes_in_ascending_order(
        self
    ):

        mask_2d = np.full(fill_value=False, shape=(4, 4))

        mask_1d_index_for_padded_mask_2d_index = aa.util.binning.mask_1d_index_for_padded_mask_2d_index_via_mask_2d(
            mask_2d=mask_2d, bin_up_factor=1
        )

        assert (
            mask_1d_index_for_padded_mask_2d_index
            == np.array([[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15]])
        ).all()

        mask_2d = np.array([[False, False], [True, False], [True, False]])

        mask_1d_index_for_padded_mask_2d_index = aa.util.binning.mask_1d_index_for_padded_mask_2d_index_via_mask_2d(
            mask_2d=mask_2d, bin_up_factor=1
        )

        assert (
            mask_1d_index_for_padded_mask_2d_index
            == np.array([[0, 1], [-1, 2], [-1, 3]])
        ).all()

    def test__includes_padding__padded_entries_are_given_minus_ones(self):

        mask_2d = np.full(fill_value=False, shape=(4, 4))

        mask_1d_index_for_padded_mask_2d_index = aa.util.binning.mask_1d_index_for_padded_mask_2d_index_via_mask_2d(
            mask_2d=mask_2d, bin_up_factor=3
        )

        assert (
            mask_1d_index_for_padded_mask_2d_index
            == np.array(
                [
                    [-1, -1, -1, -1, -1, -1],
                    [-1, 0, 1, 2, 3, -1],
                    [-1, 4, 5, 6, 7, -1],
                    [-1, 8, 9, 10, 11, -1],
                    [-1, 12, 13, 14, 15, -1],
                    [-1, -1, -1, -1, -1, -1],
                ]
            )
        ).all()

        mask_2d = np.array([[False, False], [True, False], [True, False]])

        mask_1d_index_for_padded_mask_2d_index = aa.util.binning.mask_1d_index_for_padded_mask_2d_index_via_mask_2d(
            mask_2d=mask_2d, bin_up_factor=2
        )

        assert (
            mask_1d_index_for_padded_mask_2d_index
            == np.array([[-1, -1], [0, 1], [-1, 2], [-1, 3]])
        ).all()


class TestBinnedMask1dIndexesForMask2dIndex:
    def test__masks_are_full_arrays_and_bin_up_factor_2__mapping_is_correct(self):

        mask_2d = np.full(fill_value=False, shape=(4, 4))

        binned_mask_1d_index_for_mask_2d_index = aa.util.binning.binned_mask_1d_index_for_padded_mask_2d_index_via_mask_2d(
            mask_2d=mask_2d, bin_up_factor=2
        )

        assert (
            binned_mask_1d_index_for_mask_2d_index
            == np.array([[0, 0, 1, 1], [0, 0, 1, 1], [2, 2, 3, 3], [2, 2, 3, 3]])
        ).all()

        mask_2d = np.full(fill_value=False, shape=(9, 9))

        binned_mask_1d_index_for_mask_2d_index = aa.util.binning.binned_mask_1d_index_for_padded_mask_2d_index_via_mask_2d(
            mask_2d=mask_2d, bin_up_factor=3
        )

        assert (
            binned_mask_1d_index_for_mask_2d_index
            == np.array(
                [
                    [0, 0, 0, 1, 1, 1, 2, 2, 2],
                    [0, 0, 0, 1, 1, 1, 2, 2, 2],
                    [0, 0, 0, 1, 1, 1, 2, 2, 2],
                    [3, 3, 3, 4, 4, 4, 5, 5, 5],
                    [3, 3, 3, 4, 4, 4, 5, 5, 5],
                    [3, 3, 3, 4, 4, 4, 5, 5, 5],
                    [6, 6, 6, 7, 7, 7, 8, 8, 8],
                    [6, 6, 6, 7, 7, 7, 8, 8, 8],
                    [6, 6, 6, 7, 7, 7, 8, 8, 8],
                ]
            )
        ).all()

    def test__masks_are_rectangular_arrays__include_areas_which_bin_up_is_all_true(
        self
    ):

        mask_2d = np.array(
            [
                [True, False, True, True, True, True],
                [False, False, False, True, True, True],
            ]
        )

        binned_mask_1d_index_for_mask_2d_index = aa.util.binning.binned_mask_1d_index_for_padded_mask_2d_index_via_mask_2d(
            mask_2d=mask_2d, bin_up_factor=2
        )

        assert (
            binned_mask_1d_index_for_mask_2d_index
            == np.array([[-1, 0, -1, -1, -1, -1], [0, 0, 1, -1, -1, -1]])
        ).all()

        mask_2d = np.array(
            [
                [True, False],
                [False, False],
                [False, True],
                [True, True],
                [True, True],
                [True, True],
            ]
        )

        binned_mask_1d_index_for_mask_2d_index = aa.util.binning.binned_mask_1d_index_for_padded_mask_2d_index_via_mask_2d(
            mask_2d=mask_2d, bin_up_factor=2
        )

        assert (
            binned_mask_1d_index_for_mask_2d_index
            == np.array([[-1, 0], [0, 0], [1, -1], [-1, -1], [-1, -1], [-1, -1]])
        ).all()

    def test__mask_includes_padding__mapper_mask_accounts_for_padding(self):

        mask_2d = np.full(fill_value=False, shape=(5, 5))

        binned_mask_1d_index_for_mask_2d_index = aa.util.binning.binned_mask_1d_index_for_padded_mask_2d_index_via_mask_2d(
            mask_2d=mask_2d, bin_up_factor=2
        )

        assert (
            binned_mask_1d_index_for_mask_2d_index
            == np.array(
                [
                    [-1, -1, -1, -1, -1, -1],
                    [-1, 0, 1, 1, 2, 2],
                    [-1, 3, 4, 4, 5, 5],
                    [-1, 3, 4, 4, 5, 5],
                    [-1, 6, 7, 7, 8, 8],
                    [-1, 6, 7, 7, 8, 8],
                ]
            )
        ).all()


class TestMaskedArray1DToBininedMaskedArray1d:
    def test__masks_are_full_arrays_and_bin_up_factor_2__mapping_is_correct(self):

        mask_2d = np.full(fill_value=False, shape=(4, 4))

        binned_masked_array_1d_for_masked_array_1d = aa.util.binning.binned_masked_array_1d_for_masked_array_1d_via_mask_2d(
            mask_2d=mask_2d, bin_up_factor=2
        )

        assert (
            binned_masked_array_1d_for_masked_array_1d
            == np.array([0, 0, 1, 1, 0, 0, 1, 1, 2, 2, 3, 3, 2, 2, 3, 3])
        ).all()

        mask_2d = np.full(fill_value=False, shape=(9, 9))

        binned_masked_array_1d_for_masked_array_1d = aa.util.binning.binned_masked_array_1d_for_masked_array_1d_via_mask_2d(
            mask_2d=mask_2d, bin_up_factor=3
        )

        assert (
            binned_masked_array_1d_for_masked_array_1d
            == np.array(
                [
                    0,
                    0,
                    0,
                    1,
                    1,
                    1,
                    2,
                    2,
                    2,
                    0,
                    0,
                    0,
                    1,
                    1,
                    1,
                    2,
                    2,
                    2,
                    0,
                    0,
                    0,
                    1,
                    1,
                    1,
                    2,
                    2,
                    2,
                    3,
                    3,
                    3,
                    4,
                    4,
                    4,
                    5,
                    5,
                    5,
                    3,
                    3,
                    3,
                    4,
                    4,
                    4,
                    5,
                    5,
                    5,
                    3,
                    3,
                    3,
                    4,
                    4,
                    4,
                    5,
                    5,
                    5,
                    6,
                    6,
                    6,
                    7,
                    7,
                    7,
                    8,
                    8,
                    8,
                    6,
                    6,
                    6,
                    7,
                    7,
                    7,
                    8,
                    8,
                    8,
                    6,
                    6,
                    6,
                    7,
                    7,
                    7,
                    8,
                    8,
                    8,
                ]
            )
        ).all()

    def test__masks_are_rectangular_arrays__include_areas_which_bin_up_is_all_true(
        self
    ):

        mask_2d = np.array(
            [
                [True, False, True, True, True, True],
                [False, False, False, True, True, True],
            ]
        )

        binned_masked_array_1d_for_masked_array_1d = aa.util.binning.binned_masked_array_1d_for_masked_array_1d_via_mask_2d(
            mask_2d=mask_2d, bin_up_factor=2
        )

        assert (
            binned_masked_array_1d_for_masked_array_1d == np.array([0, 0, 0, 1])
        ).all()

        mask_2d = np.array(
            [
                [True, False],
                [False, False],
                [False, True],
                [True, True],
                [True, False],
                [True, False],
            ]
        )

        binned_masked_array_1d_for_masked_array_1d = aa.util.binning.binned_masked_array_1d_for_masked_array_1d_via_mask_2d(
            mask_2d=mask_2d, bin_up_factor=2
        )

        assert (
            binned_masked_array_1d_for_masked_array_1d == np.array([0, 0, 0, 1, 2, 2])
        ).all()

    def test__mask_includes_padding__mapper_mask_accounts_for_padding(self):

        mask_2d = np.full(fill_value=False, shape=(5, 5))

        binned_masked_array_1d_for_masked_array_1d = aa.util.binning.binned_masked_array_1d_for_masked_array_1d_via_mask_2d(
            mask_2d=mask_2d, bin_up_factor=2
        )

        assert (
            binned_masked_array_1d_for_masked_array_1d
            == np.array(
                [
                    0,
                    1,
                    1,
                    2,
                    2,
                    3,
                    4,
                    4,
                    5,
                    5,
                    3,
                    4,
                    4,
                    5,
                    5,
                    6,
                    7,
                    7,
                    8,
                    8,
                    6,
                    7,
                    7,
                    8,
                    8,
                ]
            )
        ).all()


class TestBinnedMaskArrayToMaskedArray:
    def test__masks_are_full_arrays_and_bin_up_factor_2__mapping_is_correct(self):

        mask_2d = np.full(fill_value=False, shape=(4, 4))

        masked_array_1d_for_binned_masked_array_1d = aa.util.binning.masked_array_1d_for_binned_masked_array_1d_via_mask_2d(
            mask_2d=mask_2d, bin_up_factor=2
        )

        assert (
            masked_array_1d_for_binned_masked_array_1d == np.array([0, 2, 8, 10])
        ).all()

        mask_2d = np.full(fill_value=False, shape=(9, 9))

        masked_array_1d_for_binned_masked_array_1d = aa.util.binning.masked_array_1d_for_binned_masked_array_1d_via_mask_2d(
            mask_2d=mask_2d, bin_up_factor=3
        )

        assert (
            masked_array_1d_for_binned_masked_array_1d
            == np.array([0, 3, 6, 27, 30, 33, 54, 57, 60])
        ).all()

    def test__masks_are_rectangular_arrays__include_areas_which_bin_up_is_all_true(
        self
    ):

        mask_2d = np.array(
            [
                [True, False, True, True, True, True],
                [False, False, False, True, True, True],
            ]
        )

        masked_array_1d_for_binned_masked_array_1d = aa.util.binning.masked_array_1d_for_binned_masked_array_1d_via_mask_2d(
            mask_2d=mask_2d, bin_up_factor=2
        )

        assert (masked_array_1d_for_binned_masked_array_1d == np.array([0, 3])).all()

        mask_2d = np.array(
            [
                [True, False],
                [False, False],
                [False, True],
                [True, True],
                [True, True],
                [True, True],
            ]
        )

        masked_array_1d_for_binned_masked_array_1d = aa.util.binning.masked_array_1d_for_binned_masked_array_1d_via_mask_2d(
            mask_2d=mask_2d, bin_up_factor=2
        )

        assert (masked_array_1d_for_binned_masked_array_1d == np.array([0, 3])).all()

    def test__mask_includes_padding__mapper_mask_accounts_for_padding(self):

        mask_2d = np.full(fill_value=False, shape=(5, 5))

        masked_array_1d_for_binned_masked_array_1d = aa.util.binning.masked_array_1d_for_binned_masked_array_1d_via_mask_2d(
            mask_2d=mask_2d, bin_up_factor=2
        )

        assert (
            masked_array_1d_for_binned_masked_array_1d
            == np.array([0, 1, 3, 5, 6, 8, 15, 16, 18])
        ).all()


class TestBinnedMaskArrayToMaskedArrayAll:
    def test__masks_are_full_arrays_and_bin_up_factor_2__mapping_is_correct(self):

        mask_2d = np.full(fill_value=False, shape=(4, 4))

        masked_array_1d_for_binned_masked_array_1d_all, binned_masked_array_1d_sizes = aa.util.binning.masked_array_1d_for_binned_masked_array_1d_all_via_mask_2d(
            mask_2d=mask_2d, bin_up_factor=2
        )

        assert (binned_masked_array_1d_sizes == np.array([4, 4, 4, 4])).all()

        assert (
            masked_array_1d_for_binned_masked_array_1d_all
            == np.array([[0, 1, 4, 5], [2, 3, 6, 7], [8, 9, 12, 13], [10, 11, 14, 15]])
        ).all()

        mask_2d = np.full(fill_value=False, shape=(9, 9))

        masked_array_1d_for_binned_masked_array_1d_all, binned_masked_array_1d_sizes = aa.util.binning.masked_array_1d_for_binned_masked_array_1d_all_via_mask_2d(
            mask_2d=mask_2d, bin_up_factor=3
        )

        assert (
            binned_masked_array_1d_sizes == np.array([9, 9, 9, 9, 9, 9, 9, 9, 9])
        ).all()

        assert (
            masked_array_1d_for_binned_masked_array_1d_all
            == np.array(
                [
                    [0, 1, 2, 9, 10, 11, 18, 19, 20],
                    [3, 4, 5, 12, 13, 14, 21, 22, 23],
                    [6, 7, 8, 15, 16, 17, 24, 25, 26],
                    [27, 28, 29, 36, 37, 38, 45, 46, 47],
                    [30, 31, 32, 39, 40, 41, 48, 49, 50],
                    [33, 34, 35, 42, 43, 44, 51, 52, 53],
                    [54, 55, 56, 63, 64, 65, 72, 73, 74],
                    [57, 58, 59, 66, 67, 68, 75, 76, 77],
                    [60, 61, 62, 69, 70, 71, 78, 79, 80],
                ]
            )
        ).all()

    def test__masks_are_rectangular_arrays__include_areas_which_bin_up_is_all_true(
        self
    ):

        mask_2d = np.array(
            [
                [True, False, True, True, True, True],
                [False, False, False, True, True, True],
            ]
        )

        masked_array_1d_for_binned_masked_array_1d_all, binned_masked_array_1d_sizes = aa.util.binning.masked_array_1d_for_binned_masked_array_1d_all_via_mask_2d(
            mask_2d=mask_2d, bin_up_factor=2
        )

        assert (binned_masked_array_1d_sizes == np.array([3, 1])).all()

        assert (
            masked_array_1d_for_binned_masked_array_1d_all
            == np.array([[0, 1, 2, -1], [3, -1, -1, -1]])
        ).all()

        mask_2d = np.array(
            [
                [True, False],
                [False, False],
                [False, True],
                [True, True],
                [True, True],
                [True, True],
            ]
        )

        masked_array_1d_for_binned_masked_array_1d_all, binned_masked_array_1d_sizes = aa.util.binning.masked_array_1d_for_binned_masked_array_1d_all_via_mask_2d(
            mask_2d=mask_2d, bin_up_factor=2
        )

        assert (binned_masked_array_1d_sizes == np.array([3, 1])).all()

        assert (
            masked_array_1d_for_binned_masked_array_1d_all
            == np.array([[0, 1, 2, -1], [3, -1, -1, -1]])
        ).all()

    def test__mask_includes_padding__mapper_mask_accounts_for_padding(self):

        mask_2d = np.full(fill_value=False, shape=(5, 5))

        masked_array_1d_for_binned_masked_array_1d_all, binned_masked_array_1d_sizes = aa.util.binning.masked_array_1d_for_binned_masked_array_1d_all_via_mask_2d(
            mask_2d=mask_2d, bin_up_factor=2
        )

        assert (
            binned_masked_array_1d_sizes == np.array([1, 2, 2, 2, 4, 4, 2, 4, 4])
        ).all()

        assert (
            masked_array_1d_for_binned_masked_array_1d_all
            == np.array(
                [
                    [0, -1, -1, -1],
                    [1, 2, -1, -1],
                    [3, 4, -1, -1],
                    [5, 10, -1, -1],
                    [6, 7, 11, 12],
                    [8, 9, 13, 14],
                    [15, 20, -1, -1],
                    [16, 17, 21, 22],
                    [18, 19, 23, 24],
                ]
            )
        ).all()
