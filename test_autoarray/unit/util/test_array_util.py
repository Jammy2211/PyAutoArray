import autoarray as aa
import os
import shutil
import numpy as np
import pytest


test_data_path = "{}/../test_files/array/".format(
    os.path.dirname(os.path.realpath(__file__))
)


@pytest.fixture(name="memoizer")
def make_memoizer():
    return aa.util.array.Memoizer()


class TestMemoizer(object):
    def test_storing(self, memoizer):
        @memoizer
        def func(arg):
            return "result for {}".format(arg)

        func(1)
        func(2)
        func(1)

        assert memoizer.results == {
            "('arg', 1)": "result for 1",
            "('arg', 2)": "result for 2",
        }
        assert memoizer.calls == 2

    def test_multiple_arguments(self, memoizer):
        @memoizer
        def func(arg1, arg2):
            return arg1 * arg2

        func(1, 2)
        func(2, 1)
        func(1, 2)

        assert memoizer.results == {
            "('arg1', 1), ('arg2', 2)": 2,
            "('arg1', 2), ('arg2', 1)": 2,
        }
        assert memoizer.calls == 2

    def test_key_word_arguments(self, memoizer):
        @memoizer
        def func(arg1=0, arg2=0):
            return arg1 * arg2

        func(arg1=1)
        func(arg2=1)
        func(arg1=1)
        func(arg1=1, arg2=1)

        assert memoizer.results == {
            "('arg1', 1)": 0,
            "('arg2', 1)": 0,
            "('arg1', 1), ('arg2', 1)": 1,
        }
        assert memoizer.calls == 3

    def test_key_word_for_positional(self, memoizer):
        @memoizer
        def func(arg):
            return "result for {}".format(arg)

        func(1)
        func(arg=2)
        func(arg=1)

        assert memoizer.calls == 2

    def test_methods(self, memoizer):
        class Class(object):
            def __init__(self, value):
                self.value = value

            @memoizer
            def method(self):
                return self.value

        one = Class(1)
        two = Class(2)

        assert one.method() == 1
        assert two.method() == 2


class TestResize:
    def test__trim__from_7x7_to_3x3(self):
        array = np.ones((7, 7))
        array[3, 3] = 2.0

        modified = aa.util.array.resized_array_2d_from_array_2d(
            array_2d=array, resized_shape=(3, 3)
        )

        assert (
            modified == np.array([[1.0, 1.0, 1.0], [1.0, 2.0, 1.0], [1.0, 1.0, 1.0]])
        ).all()

    def test__trim__from_7x7_to_4x4(self):
        array = np.ones((7, 7))
        array[3, 3] = 2.0

        modified = aa.util.array.resized_array_2d_from_array_2d(
            array_2d=array, resized_shape=(4, 4)
        )

        assert (
            modified
            == np.array(
                [
                    [1.0, 1.0, 1.0, 1.0],
                    [1.0, 1.0, 1.0, 1.0],
                    [1.0, 1.0, 2.0, 1.0],
                    [1.0, 1.0, 1.0, 1.0],
                ]
            )
        ).all()

    def test__trim__from_6x6_to_4x4(self):

        array = np.ones((6, 6))
        array[2:4, 2:4] = 2.0

        modified = aa.util.array.resized_array_2d_from_array_2d(
            array_2d=array, resized_shape=(4, 4)
        )

        assert (
            modified
            == np.array(
                [
                    [1.0, 1.0, 1.0, 1.0],
                    [1.0, 2.0, 2.0, 1.0],
                    [1.0, 2.0, 2.0, 1.0],
                    [1.0, 1.0, 1.0, 1.0],
                ]
            )
        ).all()

    def test__trim__from_6x6_to_3x3(self):

        array = np.ones((6, 6))
        array[2:4, 2:4] = 2.0

        modified = aa.util.array.resized_array_2d_from_array_2d(
            array_2d=array, resized_shape=(3, 3)
        )

        assert (
            modified == np.array([[2.0, 2.0, 1.0], [2.0, 2.0, 1.0], [1.0, 1.0, 1.0]])
        ).all()

    def test__trim__from_5x4_to_3x2(self):
        array = np.ones((5, 4))
        array[2, 1:3] = 2.0

        modified = aa.util.array.resized_array_2d_from_array_2d(
            array_2d=array, resized_shape=(3, 2)
        )

        assert (modified == np.array([[1.0, 1.0], [2.0, 2.0], [1.0, 1.0]])).all()

    def test__trim__from_4x5_to_2x3(self):
        array = np.ones((4, 5))
        array[1:3, 2] = 2.0

        modified = aa.util.array.resized_array_2d_from_array_2d(
            array_2d=array, resized_shape=(2, 3)
        )

        assert (modified == np.array([[1.0, 2.0, 1.0], [1.0, 2.0, 1.0]])).all()

    def test__trim_with_new_origin_as_input(self):

        array = np.ones((7, 7))
        array[4, 4] = 2.0
        modified = aa.util.array.resized_array_2d_from_array_2d(
            array_2d=array, resized_shape=(3, 3), origin=(4, 4)
        )
        assert (
            modified == np.array([[1.0, 1.0, 1.0], [1.0, 2.0, 1.0], [1.0, 1.0, 1.0]])
        ).all()

        array = np.ones((6, 6))
        array[3, 4] = 2.0
        modified = aa.util.array.resized_array_2d_from_array_2d(
            array_2d=array, resized_shape=(3, 3), origin=(3, 4)
        )
        assert (
            modified == np.array([[1.0, 1.0, 1.0], [1.0, 2.0, 1.0], [1.0, 1.0, 1.0]])
        ).all()

        array = np.ones((9, 8))
        array[4, 3] = 2.0
        modified = aa.util.array.resized_array_2d_from_array_2d(
            array_2d=array, resized_shape=(3, 3), origin=(4, 3)
        )
        assert (
            modified == np.array([[1.0, 1.0, 1.0], [1.0, 2.0, 1.0], [1.0, 1.0, 1.0]])
        ).all()

        array = np.ones((8, 9))
        array[3, 5] = 2.0
        modified = aa.util.array.resized_array_2d_from_array_2d(
            array_2d=array, resized_shape=(3, 3), origin=(3, 5)
        )
        assert (
            modified == np.array([[1.0, 1.0, 1.0], [1.0, 2.0, 1.0], [1.0, 1.0, 1.0]])
        ).all()

    def test__pad__from_3x3_to_5x5(self):

        array = np.ones((3, 3))
        array[1, 1] = 2.0

        modified = aa.util.array.resized_array_2d_from_array_2d(
            array_2d=array, resized_shape=(5, 5)
        )

        assert (
            modified
            == np.array(
                [
                    [0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 1.0, 1.0, 0.0],
                    [0.0, 1.0, 2.0, 1.0, 0.0],
                    [0.0, 1.0, 1.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0],
                ]
            )
        ).all()

    def test__pad__from_3x3_to_4x4(self):

        array = np.ones((3, 3))
        array[1, 1] = 2.0

        modified = aa.util.array.resized_array_2d_from_array_2d(
            array_2d=array, resized_shape=(4, 4)
        )

        assert (
            modified
            == np.array(
                [
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 1.0, 1.0],
                    [0.0, 1.0, 2.0, 1.0],
                    [0.0, 1.0, 1.0, 1.0],
                ]
            )
        ).all()

    def test__pad__from_4x4_to_6x6(self):

        array = np.ones((4, 4))
        array[1:3, 1:3] = 2.0

        modified = aa.util.array.resized_array_2d_from_array_2d(
            array_2d=array, resized_shape=(6, 6)
        )

        assert (
            modified
            == np.array(
                [
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 1.0, 1.0, 1.0, 0.0],
                    [0.0, 1.0, 2.0, 2.0, 1.0, 0.0],
                    [0.0, 1.0, 2.0, 2.0, 1.0, 0.0],
                    [0.0, 1.0, 1.0, 1.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                ]
            )
        ).all()

    def test__pad__from_4x4_to_5x5(self):

        array = np.ones((4, 4))
        array[1:3, 1:3] = 2.0

        modified = aa.util.array.resized_array_2d_from_array_2d(
            array_2d=array, resized_shape=(5, 5)
        )

        assert (
            modified
            == np.array(
                [
                    [1.0, 1.0, 1.0, 1.0, 0.0],
                    [1.0, 2.0, 2.0, 1.0, 0.0],
                    [1.0, 2.0, 2.0, 1.0, 0.0],
                    [1.0, 1.0, 1.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0],
                ]
            )
        ).all()

    def test__pad__from_3x2_to_5x4(self):
        array = np.ones((3, 2))
        array[1, 0:2] = 2.0

        modified = aa.util.array.resized_array_2d_from_array_2d(
            array_2d=array, resized_shape=(5, 4)
        )

        assert (
            modified
            == np.array(
                [
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 1.0, 0.0],
                    [0.0, 2.0, 2.0, 0.0],
                    [0.0, 1.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                ]
            )
        ).all()

    def test__pad__from_2x3_to_4x5(self):
        array = np.ones((2, 3))
        array[0:2, 1] = 2.0

        modified = aa.util.array.resized_array_2d_from_array_2d(
            array_2d=array, resized_shape=(4, 5)
        )

        assert (
            modified
            == np.array(
                [
                    [0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 2.0, 1.0, 0.0],
                    [0.0, 1.0, 2.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0],
                ]
            )
        ).all()

    def test__pad__with_input_new_origin(self):

        array = np.ones((3, 3))
        array[2, 2] = 2.0
        modified = aa.util.array.resized_array_2d_from_array_2d(
            array_2d=array, resized_shape=(5, 5), origin=(2, 2)
        )

        assert (
            modified
            == np.array(
                [
                    [1.0, 1.0, 1.0, 0.0, 0.0],
                    [1.0, 1.0, 1.0, 0.0, 0.0],
                    [1.0, 1.0, 2.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0],
                ]
            )
        ).all()

        array = np.ones((2, 3))
        array[0, 0] = 2.0
        modified = aa.util.array.resized_array_2d_from_array_2d(
            array_2d=array, resized_shape=(4, 5), origin=(0, 1)
        )

        assert (
            modified
            == np.array(
                [
                    [0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 2.0, 1.0, 1.0, 0.0],
                    [0.0, 1.0, 1.0, 1.0, 0.0],
                ]
            )
        ).all()


class TestFits:
    def test__numpy_array_1d_from_fits(self):
        arr = aa.util.array.numpy_array_1d_from_fits(
            file_path=test_data_path + "3_ones.fits", hdu=0
        )

        assert (arr == np.ones((3))).all()

    def test__numpy_array_1d_to_fits__output_and_load(self):

        if os.path.exists(test_data_path + "test_autoarray.fits"):
            os.remove(test_data_path + "test_autoarray.fits")

        arr = np.array([10.0, 30.0, 40.0, 92.0, 19.0, 20.0])

        aa.util.array.numpy_array_1d_to_fits(
            arr, file_path=test_data_path + "test_autoarray.fits"
        )

        array_load = aa.util.array.numpy_array_1d_from_fits(
            file_path=test_data_path + "test_autoarray.fits", hdu=0
        )

        assert (arr == array_load).all()

    def test__numpy_array_2d_from_fits(self):
        arr = aa.util.array.numpy_array_2d_from_fits(
            file_path=test_data_path + "3x3_ones.fits", hdu=0
        )

        assert (arr == np.ones((3, 3))).all()

        arr = aa.util.array.numpy_array_2d_from_fits(
            file_path=test_data_path + "4x3_ones.fits", hdu=0
        )

        assert (arr == np.ones((4, 3))).all()

    def test__numpy_array_2d_to_fits__output_and_load(self):

        if os.path.exists(test_data_path + "test_autoarray.fits"):
            os.remove(test_data_path + "test_autoarray.fits")

        arr = np.array([[10.0, 30.0, 40.0], [92.0, 19.0, 20.0]])

        aa.util.array.numpy_array_2d_to_fits(
            arr, file_path=test_data_path + "test_autoarray.fits"
        )

        array_load = aa.util.array.numpy_array_2d_from_fits(
            file_path=test_data_path + "test_autoarray.fits", hdu=0
        )

        assert (arr == array_load).all()


class TestReplaceNegativeNoise:
    def test__2x2_array__no_negative_values__no_change(self):

        image_2d = np.ones(shape=(2, 2))

        noise_map_2d = np.array([[1.0, 2.0], [3.0, 4.0]])

        noise_map_2d = aa.util.array.replace_noise_map_2d_values_where_image_2d_values_are_negative(
            image_2d=image_2d, noise_map_2d=noise_map_2d, target_signal_to_noise=1.0
        )

        assert (noise_map_2d == noise_map_2d).all()

    def test__2x2_array__negative_values__do_not_produce_absolute_signal_to_noise_values_above_target__no_change(
        self
    ):

        image_2d = -1.0 * np.ones(shape=(2, 2))

        noise_map_2d = np.array([[1.0, 0.5], [0.25, 0.125]])

        noise_map_2d = aa.util.array.replace_noise_map_2d_values_where_image_2d_values_are_negative(
            image_2d=image_2d, noise_map_2d=noise_map_2d, target_signal_to_noise=10.0
        )

        assert (noise_map_2d == noise_map_2d).all()

    def test__2x2_array__negative_values__values_give_absolute_signal_to_noise_below_target__replaces_their_noise(
        self
    ):

        image_2d = -1.0 * np.ones(shape=(2, 2))

        noise_map_2d = np.array([[1.0, 0.5], [0.25, 0.125]])

        noise_map_2d = aa.util.array.replace_noise_map_2d_values_where_image_2d_values_are_negative(
            image_2d=image_2d, noise_map_2d=noise_map_2d, target_signal_to_noise=4.0
        )

        assert (noise_map_2d == np.array([[1.0, 0.5], [0.25, 0.25]])).all()

        noise_map_2d = np.array([[1.0, 0.5], [0.25, 0.125]])

        noise_map_2d = aa.util.array.replace_noise_map_2d_values_where_image_2d_values_are_negative(
            image_2d=image_2d, noise_map_2d=noise_map_2d, target_signal_to_noise=2.0
        )

        assert (noise_map_2d == np.array([[1.0, 0.5], [0.5, 0.5]])).all()

        noise_map_2d = np.array([[1.0, 0.5], [0.25, 0.125]])

        noise_map_2d = aa.util.array.replace_noise_map_2d_values_where_image_2d_values_are_negative(
            image_2d=image_2d, noise_map_2d=noise_map_2d, target_signal_to_noise=1.0
        )

        assert (noise_map_2d == np.array([[1.0, 1.0], [1.0, 1.0]])).all()

        noise_map_2d = np.array([[1.0, 0.5], [0.25, 0.125]])

        noise_map_2d = aa.util.array.replace_noise_map_2d_values_where_image_2d_values_are_negative(
            image_2d=image_2d, noise_map_2d=noise_map_2d, target_signal_to_noise=0.5
        )

        assert (noise_map_2d == np.array([[2.0, 2.0], [2.0, 2.0]])).all()

    def test__same_as_above__image_not_all_negative_ones(self):

        image_2d = np.array([[1.0, -2.0], [5.0, -4.0]])

        noise_map_2d = np.array([[3.0, 1.0], [4.0, 8.0]])

        noise_map_2d = aa.util.array.replace_noise_map_2d_values_where_image_2d_values_are_negative(
            image_2d=image_2d, noise_map_2d=noise_map_2d, target_signal_to_noise=1.0
        )

        assert (noise_map_2d == np.array([[3.0, 2.0], [4.0, 8.0]])).all()

        image_2d = np.array([[-10.0, -20.0], [100.0, -30.0]])

        noise_map_2d = np.array([[1.0, 2.0], [40.0, 3.0]])

        noise_map_2d = aa.util.array.replace_noise_map_2d_values_where_image_2d_values_are_negative(
            image_2d=image_2d, noise_map_2d=noise_map_2d, target_signal_to_noise=5.0
        )

        assert (noise_map_2d == np.array([[2.0, 4.0], [40.0, 6.0]])).all()

    def test__rectangular_2x3_and_3x2_arrays(self):

        image_2d = -1.0 * np.ones(shape=(2, 3))

        noise_map_2d = np.array([[1.0, 0.5, 0.25], [0.25, 0.125, 2.0]])

        noise_map_2d = aa.util.array.replace_noise_map_2d_values_where_image_2d_values_are_negative(
            image_2d=image_2d, noise_map_2d=noise_map_2d, target_signal_to_noise=2.0
        )

        assert (noise_map_2d == np.array([[1.0, 0.5, 0.5], [0.5, 0.5, 2.0]])).all()

        image_2d = -1.0 * np.ones(shape=(3, 2))

        noise_map_2d = np.array([[1.0, 0.5], [0.25, 0.125], [0.25, 2.0]])

        noise_map_2d = aa.util.array.replace_noise_map_2d_values_where_image_2d_values_are_negative(
            image_2d=image_2d, noise_map_2d=noise_map_2d, target_signal_to_noise=2.0
        )

        assert (noise_map_2d == np.array([[1.0, 0.5], [0.5, 0.5], [0.5, 2.0]])).all()


class Test2dIndexesfrom1dIndex(object):
    def test__9_1d_indexes_from_0_to_8__map_to_shape_3x3(self):

        indexes_1d = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8])

        index_2d_for_index_1d = aa.util.array.index_2d_for_index_1d_from_shape(
            indexes_1d=indexes_1d, shape=(3, 3)
        )

        assert (
            index_2d_for_index_1d
            == np.array(
                [[0, 0], [0, 1], [0, 2], [1, 0], [1, 1], [1, 2], [2, 0], [2, 1], [2, 2]]
            )
        ).all()

    def test__6_1d_indexes_from_0_to_5__map_to_shape_2x3(self):

        indexes_1d = np.array([0, 1, 2, 3, 4, 5])

        index_2d_for_index_1d = aa.util.array.index_2d_for_index_1d_from_shape(
            indexes_1d=indexes_1d, shape=(2, 3)
        )

        assert (
            index_2d_for_index_1d
            == np.array([[0, 0], [0, 1], [0, 2], [1, 0], [1, 1], [1, 2]])
        ).all()

    def test__6_1d_indexes_from_0_to_5__map_to_shape_3x2(self):

        indexes_1d = np.array([0, 1, 2, 3, 4, 5])

        indexes_2d = aa.util.array.index_2d_for_index_1d_from_shape(
            indexes_1d=indexes_1d, shape=(3, 2)
        )

        assert (
            indexes_2d == np.array([[0, 0], [0, 1], [1, 0], [1, 1], [2, 0], [2, 1]])
        ).all()

    def test__9_1d_indexes_from_0_to_8_different_order__map_to_shape_3x3(self):

        indexes_1d = np.array([1, 4, 7, 8, 0, 2, 3, 5, 6])

        index_2d_for_index_1d = aa.util.array.index_2d_for_index_1d_from_shape(
            indexes_1d=indexes_1d, shape=(3, 3)
        )

        assert (
            index_2d_for_index_1d
            == np.array(
                [[0, 1], [1, 1], [2, 1], [2, 2], [0, 0], [0, 2], [1, 0], [1, 2], [2, 0]]
            )
        ).all()


class Test1dIndexFromIndex2D(object):
    def test__9_2d_indexes_from_0_0_to_2_2__map_to_shape_3x3(self):

        indexes_2d = np.array(
            [[0, 0], [0, 1], [0, 2], [1, 0], [1, 1], [1, 2], [2, 0], [2, 1], [2, 2]]
        )

        index_1d_for_index_2d = aa.util.array.index_1d_for_index_2d_from_shape(
            indexes_2d=indexes_2d, shape=(3, 3)
        )

        assert (index_1d_for_index_2d == np.array([0, 1, 2, 3, 4, 5, 6, 7, 8])).all()

    def test__6_1d_indexes_from_0_0_to_1_2__map_to_shape_2x3(self):

        indexes_2d = np.array([[0, 0], [0, 1], [0, 2], [1, 0], [1, 1], [1, 2]])

        index_1d_for_index_2d = aa.util.array.index_1d_for_index_2d_from_shape(
            indexes_2d=indexes_2d, shape=(2, 3)
        )

        assert (index_1d_for_index_2d == np.array([0, 1, 2, 3, 4, 5])).all()

    def test__6_1d_indexes_from_0_0_to_2_1__map_to_shape_3x2(self):

        indexes_2d = np.array([[0, 0], [0, 1], [1, 0], [1, 1], [2, 0], [2, 1]])

        index_1d_for_index_2d = aa.util.array.index_1d_for_index_2d_from_shape(
            indexes_2d=indexes_2d, shape=(3, 2)
        )

        assert (index_1d_for_index_2d == np.array([0, 1, 2, 3, 4, 5])).all()

    def test__9_1d_indexes_from_0_0_to_2_2_different_order__map_to_shape_3x3(self):

        indexes_2d = np.array(
            [[0, 1], [1, 1], [2, 1], [2, 2], [0, 0], [0, 2], [1, 0], [1, 2], [2, 0]]
        )

        index_1d_for_index_2d = aa.util.array.index_1d_for_index_2d_from_shape(
            indexes_2d=indexes_2d, shape=(3, 3)
        )

        assert (index_1d_for_index_2d == np.array([1, 4, 7, 8, 0, 2, 3, 5, 6])).all()


class TestArray1DFromArray2d(object):
    def test__map_simple_data__sub_size_1(self):

        array_2d = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

        mask = np.array([[True, True, True], [True, False, True], [True, True, True]])

        array_1d = aa.util.array.sub_array_1d_from_sub_array_2d(
            mask=mask, sub_array_2d=array_2d, sub_size=1
        )

        assert (array_1d == np.array([5])).all()

        array_2d = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

        mask = np.array(
            [[True, False, True], [False, False, False], [True, False, True]]
        )

        array_1d = aa.util.array.sub_array_1d_from_sub_array_2d(
            mask=mask, sub_array_2d=array_2d, sub_size=1
        )

        assert (array_1d == np.array([2, 4, 5, 6, 8])).all()

        array_2d = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])

        mask = np.array(
            [
                [True, False, True, True],
                [False, False, False, True],
                [True, False, True, False],
            ]
        )

        array_1d = aa.util.array.sub_array_1d_from_sub_array_2d(
            mask=mask, sub_array_2d=array_2d, sub_size=1
        )

        assert (array_1d == np.array([2, 5, 6, 7, 10, 12])).all()

        array_2d = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])

        mask = np.array(
            [
                [True, False, True],
                [False, False, False],
                [True, False, True],
                [True, True, True],
            ]
        )

        array_1d = aa.util.array.sub_array_1d_from_sub_array_2d(
            mask=mask, sub_array_2d=array_2d, sub_size=1
        )

        assert (array_1d == np.array([2, 4, 5, 6, 8])).all()

    def test__map_simple_data__sub_size_2(self):

        sub_array_2d = np.array(
            [
                [1, 2, 3, 4, 5, 6],
                [7, 8, 9, 10, 11, 12],
                [13, 14, 15, 16, 17, 18],
                [1, 2, 3, 4, 5, 6],
                [7, 8, 9, 10, 11, 12],
                [13, 14, 15, 16, 17, 18],
            ]
        )

        mask = np.array([[True, True, True], [True, False, True], [True, True, True]])

        sub_array_1d = aa.util.array.sub_array_1d_from_sub_array_2d(
            sub_array_2d=sub_array_2d, mask=mask, sub_size=2
        )

        assert (sub_array_1d == np.array([15, 16, 3, 4])).all()

        mask = np.array([[True, False, True], [True, False, True], [True, True, False]])

        sub_array_1d = aa.util.array.sub_array_1d_from_sub_array_2d(
            sub_array_2d=sub_array_2d, mask=mask, sub_size=2
        )

        assert (
            sub_array_1d == np.array([3, 4, 9, 10, 15, 16, 3, 4, 11, 12, 17, 18])
        ).all()

        sub_array_2d = np.array(
            [
                [1, 2, 3, 4, 5, 6, 7, 7, 7],
                [7, 8, 9, 10, 11, 12, 7, 7, 7],
                [13, 14, 15, 16, 17, 18, 7, 7, 7],
                [1, 2, 3, 4, 5, 6, 7, 7, 7],
                [7, 8, 9, 10, 11, 12, 7, 7, 7],
                [13, 14, 15, 16, 17, 18, 7, 7, 7],
            ]
        )

        mask = np.array(
            [
                [True, False, True, True],
                [False, False, False, True],
                [True, False, True, False],
            ]
        )

        sub_array_1d = aa.util.array.sub_array_1d_from_sub_array_2d(
            sub_array_2d=sub_array_2d, mask=mask, sub_size=2
        )

        assert (
            sub_array_1d
            == np.array(
                [
                    3,
                    4,
                    9,
                    10,
                    13,
                    14,
                    1,
                    2,
                    15,
                    16,
                    3,
                    4,
                    17,
                    18,
                    5,
                    6,
                    9,
                    10,
                    15,
                    16,
                    7,
                    7,
                    7,
                    7,
                ]
            )
        ).all()

        sub_array_2d = np.array(
            [
                [1, 2, 3, 4, 5, 6],
                [7, 8, 9, 10, 11, 12],
                [13, 14, 15, 16, 17, 18],
                [1, 2, 3, 4, 5, 6],
                [7, 8, 9, 10, 11, 12],
                [13, 14, 15, 16, 17, 18],
                [7, 7, 7, 7, 7, 7],
                [7, 7, 7, 7, 7, 7],
            ]
        )

        mask = np.array(
            [
                [True, False, True],
                [False, False, False],
                [True, False, True],
                [True, True, True],
            ]
        )

        sub_array_1d = aa.util.array.sub_array_1d_from_sub_array_2d(
            sub_array_2d=sub_array_2d, mask=mask, sub_size=2
        )

        assert (
            sub_array_1d
            == np.array(
                [3, 4, 9, 10, 13, 14, 1, 2, 15, 16, 3, 4, 17, 18, 5, 6, 9, 10, 15, 16]
            )
        ).all()

    def test__setup_2x2_data__sub_size_3(self):

        sub_array_2d = np.array(
            [
                [1, 2, 3, 4, 5, 6],
                [7, 8, 9, 10, 11, 12],
                [13, 14, 15, 16, 17, 18],
                [1, 2, 3, 4, 5, 6],
                [7, 8, 9, 10, 11, 12],
                [13, 14, 15, 16, 17, 18],
            ]
        )

        mask = np.array([[False, True], [True, False]])

        sub_array_1d = aa.util.array.sub_array_1d_from_sub_array_2d(
            sub_array_2d=sub_array_2d, mask=mask, sub_size=3
        )

        assert (
            sub_array_1d
            == np.array([1, 2, 3, 7, 8, 9, 13, 14, 15, 4, 5, 6, 10, 11, 12, 16, 17, 18])
        ).all()


class TestArray2dForArray1d(object):
    def test__simple_2d_array__is_masked_and_mapped__sub_size_1(self):

        array_1d = np.array([1.0, 2.0, 3.0, 4.0])

        mask = np.full(fill_value=False, shape=(2, 2))

        array_2d = aa.util.array.sub_array_2d_from_sub_array_1d(
            sub_array_1d=array_1d, mask=mask, sub_size=1
        )

        assert (array_2d == np.array([[1.0, 2.0], [3.0, 4.0]])).all()

        array_1d = np.array([1.0, 2.0, 3.0])

        mask = np.array([[False, False], [False, True]])

        array_2d = aa.util.array.sub_array_2d_from_sub_array_1d(
            sub_array_1d=array_1d, mask=mask, sub_size=1
        )

        assert (array_2d == np.array([[1.0, 2.0], [3.0, 0.0]])).all()

        array_1d = np.array([1.0, 2.0, 3.0, -1.0, -2.0, -3.0])

        mask = np.array(
            [
                [False, False, True, True],
                [False, True, True, True],
                [False, False, True, False],
            ]
        )

        array_2d = aa.util.array.sub_array_2d_from_sub_array_1d(
            sub_array_1d=array_1d, mask=mask, sub_size=1
        )

        assert (
            array_2d
            == np.array(
                [[1.0, 2.0, 0.0, 0.0], [3.0, 0.0, 0.0, 0.0], [-1.0, -2.0, 0.0, -3.0]]
            )
        ).all()

    def test__simple_2d_array__is_masked_and_mapped__sub_size_2(self):

        array_1d = np.array(
            [1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0, 4.0]
        )

        mask = np.array([[False, False], [False, True]])

        array_2d = aa.util.array.sub_array_2d_from_sub_array_1d(
            sub_array_1d=array_1d, mask=mask, sub_size=2
        )

        assert (
            array_2d
            == np.array(
                [
                    [1.0, 1.0, 2.0, 2.0],
                    [1.0, 1.0, 2.0, 2.0],
                    [3.0, 3.0, 0.0, 0.0],
                    [3.0, 4.0, 0.0, 0.0],
                ]
            )
        ).all()


class TestPeakPixels:
    def test__simple_arrays(self):

        array = aa.array.manual_2d(
            array=[
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
            ]
        )

        peak_pixels = aa.util.array.peak_pixels_from_array_2d(array_2d=array.in_2d)

        assert peak_pixels == [[2, 2]]

        array = aa.array.manual_2d(
            array=[
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
            ]
        )

        peak_pixels = aa.util.array.peak_pixels_from_array_2d(array_2d=array.in_2d)

        assert peak_pixels == [[2, 1], [2, 3]]

        array = aa.array.manual_2d(
            array=[
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0, 0.0],
                [0.0, 1.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
            ]
        )

        peak_pixels = aa.util.array.peak_pixels_from_array_2d(array_2d=array.in_2d)

        assert peak_pixels == [[2, 1]]

        array = aa.array.manual_2d(
            array=[
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 2.0, 0.0],
                [0.0, 1.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
            ]
        )

        peak_pixels = aa.util.array.peak_pixels_from_array_2d(array_2d=array.in_2d)

        assert peak_pixels == [[1, 3], [2, 1]]

        array = aa.array.manual_2d(
            array=[
                [0.0, 0.0, 7.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [4.0, 1.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 8.0, 0.0, 0.0],
            ]
        )

        peak_pixels = aa.util.array.peak_pixels_from_array_2d(array_2d=array.in_2d)

        assert peak_pixels == [[2, 3]]

        array = aa.array.manual_2d(
            array=[
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [2.0, 8.0, 7.0, 6.0, 8.0],
                [4.0, 9.0, 4.0, 1.0, 8.0],
                [1.0, 0.5, 7.0, 9.0, 8.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
            ]
        )

        peak_pixels = aa.util.array.peak_pixels_from_array_2d(array_2d=array.in_2d)

        assert peak_pixels == [[2, 1], [3, 3]]

    def test__simple_arrays_with_mask(self):

        array = aa.array.manual_2d(
            array=[
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
            ]
        )

        mask_2d = aa.mask.manual(
            mask_2d=[
                [True, True, True, True, True],
                [True, True, True, True, True],
                [True, True, True, False, True],
                [True, True, True, True, True],
                [True, True, True, True, True],
            ]
        )

        peak_pixels = aa.util.array.peak_pixels_from_array_2d(
            array_2d=array.in_2d, mask_2d=mask_2d
        )

        assert peak_pixels == [[2, 3]]

        array = aa.array.manual_2d(
            array=[
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0, 0.0],
                [0.0, 1.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
            ]
        )

        mask_2d = aa.mask.manual(
            mask_2d=[
                [True, True, True, True, True],
                [True, True, True, True, True],
                [True, True, True, True, True],
                [True, True, True, True, True],
                [True, True, True, True, True],
            ]
        )

        peak_pixels = aa.util.array.peak_pixels_from_array_2d(
            array_2d=array.in_2d, mask_2d=mask_2d
        )

        assert peak_pixels == []

        array = aa.array.manual_2d(
            array=[
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [2.0, 8.0, 7.0, 6.0, 8.0],
                [4.0, 9.0, 4.0, 1.0, 8.0],
                [1.0, 0.5, 7.0, 9.0, 8.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
            ]
        )

        mask_2d = aa.mask.manual(
            mask_2d=[
                [True, True, True, True, True],
                [True, True, True, True, True],
                [True, True, True, True, True],
                [True, True, True, False, True],
                [True, True, True, True, True],
            ]
        )

        peak_pixels = aa.util.array.peak_pixels_from_array_2d(
            array_2d=array.in_2d, mask_2d=mask_2d
        )

        assert peak_pixels == [[3, 3]]


class TestTroughPixels:
    def test__simple_arrays(self):

        array = aa.array.manual_2d(
            array=[
                [9.0, 9.0, 9.0, 9.0, 9.0],
                [9.0, 9.0, 9.0, 9.0, 9.0],
                [9.0, 9.0, 1.0, 9.0, 9.0],
                [9.0, 9.0, 9.0, 9.0, 9.0],
                [9.0, 9.0, 9.0, 9.0, 9.0],
            ]
        )

        trough_pixels = aa.util.array.trough_pixels_from_array_2d(array_2d=array.in_2d)

        assert trough_pixels == [[2, 2]]

        array = aa.array.manual_2d(
            array=[
                [9.0, 9.0, 9.0, 9.0, 9.0],
                [9.0, 9.0, 9.0, 9.0, 9.0],
                [9.0, 1.0, 9.0, 1.0, 9.0],
                [9.0, 9.0, 9.0, 9.0, 9.0],
                [9.0, 9.0, 9.0, 9.0, 9.0],
            ]
        )

        trough_pixels = aa.util.array.trough_pixels_from_array_2d(array_2d=array.in_2d)

        assert trough_pixels == [[2, 1], [2, 3]]

        array = aa.array.manual_2d(
            array=[
                [9.0, 9.0, 9.0, 9.0, 9.0],
                [9.0, 9.0, 9.0, 1.0, 9.0],
                [9.0, 1.0, 9.0, 1.0, 9.0],
                [9.0, 9.0, 9.0, 9.0, 9.0],
                [9.0, 9.0, 9.0, 9.0, 9.0],
            ]
        )

        trough_pixels = aa.util.array.trough_pixels_from_array_2d(array_2d=array.in_2d)

        assert trough_pixels == [[2, 1]]

        array = aa.array.manual_2d(
            array=[
                [9.0, 9.0, 9.0, 9.0, 9.0],
                [9.0, 9.0, 9.0, 1.0, 9.0],
                [9.0, 1.0, 9.0, 2.0, 9.0],
                [9.0, 9.0, 9.0, 9.0, 9.0],
                [9.0, 9.0, 9.0, 9.0, 9.0],
            ]
        )

        trough_pixels = aa.util.array.trough_pixels_from_array_2d(array_2d=array.in_2d)

        assert trough_pixels == [[1, 3], [2, 1]]

        array = aa.array.manual_2d(
            array=[
                [9.0, 9.0, 7.0, 9.0, 9.0],
                [9.0, 9.0, 9.0, 9.0, 9.0],
                [4.0, 1.0, 9.0, 1.0, 9.0],
                [9.0, 1.0, 9.0, 9.0, 9.0],
                [9.0, 9.0, 8.0, 9.0, 9.0],
            ]
        )

        trough_pixels = aa.util.array.trough_pixels_from_array_2d(array_2d=array.in_2d)

        assert trough_pixels == [[2, 3]]

        array = aa.array.manual_2d(
            array=[
                [9.0, 9.0, 9.0, 9.0, 9.0],
                [2.0, 8.0, 7.0, 6.0, 8.0],
                [4.0, 0.1, 4.0, 1.0, 8.0],
                [1.0, 0.5, 7.0, 0.1, 8.0],
                [9.0, 9.0, 9.0, 9.0, 9.0],
            ]
        )

        trough_pixels = aa.util.array.trough_pixels_from_array_2d(array_2d=array.in_2d)

        assert trough_pixels == [[2, 1], [3, 3]]

    def test__simple_arrays_with_mask(self):

        array = aa.array.manual_2d(
            array=[
                [9.0, 9.0, 9.0, 9.0, 9.0],
                [9.0, 9.0, 9.0, 9.0, 9.0],
                [9.0, 1.0, 9.0, 1.0, 9.0],
                [9.0, 9.0, 9.0, 9.0, 9.0],
                [9.0, 9.0, 9.0, 9.0, 9.0],
            ]
        )

        mask_2d = aa.mask.manual(
            mask_2d=[
                [True, True, True, True, True],
                [True, True, True, True, True],
                [True, True, True, False, True],
                [True, True, True, True, True],
                [True, True, True, True, True],
            ]
        )

        trough_pixels = aa.util.array.trough_pixels_from_array_2d(
            array_2d=array.in_2d, mask_2d=mask_2d
        )

        assert trough_pixels == [[2, 3]]

        array = aa.array.manual_2d(
            array=[
                [9.0, 9.0, 9.0, 9.0, 9.0],
                [9.0, 9.0, 9.0, 1.0, 9.0],
                [9.0, 1.0, 9.0, 1.0, 9.0],
                [9.0, 9.0, 9.0, 9.0, 9.0],
                [9.0, 9.0, 9.0, 9.0, 9.0],
            ]
        )

        mask_2d = aa.mask.manual(
            mask_2d=[
                [True, True, True, True, True],
                [True, True, True, True, True],
                [True, True, True, True, True],
                [True, True, True, True, True],
                [True, True, True, True, True],
            ]
        )

        trough_pixels = aa.util.array.trough_pixels_from_array_2d(
            array_2d=array.in_2d, mask_2d=mask_2d
        )

        assert trough_pixels == []

        array = aa.array.manual_2d(
            array=[
                [9.0, 9.0, 9.0, 9.0, 9.0],
                [2.0, 8.0, 7.0, 6.0, 8.0],
                [4.0, 9.0, 4.0, 1.0, 8.0],
                [1.0, 0.5, 7.0, 0.1, 8.0],
                [9.0, 9.0, 9.0, 9.0, 9.0],
            ]
        )

        mask_2d = aa.mask.manual(
            mask_2d=[
                [True, True, True, True, True],
                [True, True, True, True, True],
                [True, True, True, True, True],
                [True, True, True, False, True],
                [True, True, True, True, True],
            ]
        )

        trough_pixels = aa.util.array.trough_pixels_from_array_2d(
            array_2d=array.in_2d, mask_2d=mask_2d
        )

        assert trough_pixels == [[3, 3]]
