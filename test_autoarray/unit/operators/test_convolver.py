import numpy as np
import pytest

import autoarray as aa
from autoarray import exc


@pytest.fixture(name="simple_mask_7x7")
def make_simple_mask_7x7():

    mask = np.array(
        [
            [True, True, True, True, True, True, True],
            [True, True, True, True, True, True, True],
            [True, True, False, False, False, True, True],
            [True, True, False, False, False, True, True],
            [True, True, False, False, False, True, True],
            [True, True, True, True, True, True, True],
            [True, True, True, True, True, True, True],
        ]
    )

    return aa.Mask.manual(mask=mask, sub_size=1)


@pytest.fixture(name="simple_mask_5x5")
def make_simple_mask_5x5():

    mask = np.array(
        [
            [True, True, True, True, True],
            [True, False, False, False, True],
            [True, False, False, False, True],
            [True, False, False, False, True],
            [True, True, True, True, True],
        ]
    )

    return aa.Mask.manual(mask=mask, sub_size=1)


@pytest.fixture(name="simple_mask_index_array")
def make_simple_mask_index_array():
    return np.array([[6, 7, 8], [11, 12, 13], [16, 17, 18]])


@pytest.fixture(name="cross_mask")
def make_cross_mask():
    mask = np.full((5, 5), True)

    mask[2, 2] = False
    mask[1, 2] = False
    mask[3, 2] = False
    mask[2, 1] = False
    mask[2, 3] = False

    return aa.Mask.manual(mask=mask, sub_size=1)


@pytest.fixture(name="cross_mask_index_array")
def make_cross_mask_index_array():
    return np.array([[-1, 0, -1], [1, 2, 3], [-1, 4, -1]])


@pytest.fixture(name="simple_image_frame_indexes")
def make_simple_image_frame_indexes(simple_convolver):
    return simple_convolver.make_image_frame_indexes((3, 3))


@pytest.fixture(name="cross_image_frame_indexes")
def make_cross_image_frame_indexes(cross_convolver):
    return cross_convolver.make_image_frame_indexes((3, 3))


@pytest.fixture(name="cross_mask_image_frame_indexes")
def make_cross_mask_image_frame_indexes(cross_convolver):
    return cross_convolver.make_blurring_image_frame_indexes((3, 3))


@pytest.fixture(name="simple_convolver")
def make_simple_convolver(simple_mask_5x5):

    return aa.Convolver(
        mask=simple_mask_5x5,
        kernel=aa.Kernel.manual_2d([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]),
    )


@pytest.fixture(name="cross_convolver")
def make_cross_convolver(cross_mask):
    return aa.Convolver(
        mask=cross_mask,
        kernel=aa.Kernel.manual_2d([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]),
    )


@pytest.fixture(name="simple_kernel")
def make_simple_kernel():
    return aa.Kernel.manual_2d([[0, 0.1, 0], [0.1, 0.6, 0.1], [0, 0.1, 0]])


class TestNumbering:
    def test_simple_numbering(self, simple_mask_5x5, simple_mask_index_array):

        convolver = aa.Convolver(
            mask=simple_mask_5x5, kernel=aa.Kernel.ones(shape_2d=(1, 1))
        )

        mask_index_array = convolver.mask_index_array

        assert mask_index_array.shape == (5, 5)
        # noinspection PyUnresolvedReferences
        assert (
            mask_index_array
            == np.array(
                [
                    [-1, -1, -1, -1, -1],
                    [-1, 0, 1, 2, -1],
                    [-1, 3, 4, 5, -1],
                    [-1, 6, 7, 8, -1],
                    [-1, -1, -1, -1, -1],
                ]
            )
        ).all()

    def test__cross_mask(self, cross_mask):
        convolver = aa.Convolver(
            mask=cross_mask, kernel=aa.Kernel.ones(shape_2d=(1, 1))
        )

        assert (
            convolver.mask_index_array
            == np.array(
                [
                    [-1, -1, -1, -1, -1],
                    [-1, -1, 0, -1, -1],
                    [-1, 1, 2, 3, -1],
                    [-1, -1, 4, -1, -1],
                    [-1, -1, -1, -1, -1],
                ]
            )
        ).all()

    def test__even_kernel_failure(self):
        with pytest.raises(exc.ConvolverException):
            aa.Convolver(
                mask=np.full((3, 3), False), kernel=aa.Kernel.ones(shape_2d=(2, 2))
            )


class TestFrameExtraction:
    def test__frame_at_coords(self, simple_mask_5x5, simple_convolver):
        frame, kernel_frame = simple_convolver.frame_at_coordinates_jit(
            coordinates=(2, 2),
            mask=simple_mask_5x5,
            mask_index_array=simple_convolver.mask_index_array,
            kernel_2d=simple_convolver.kernel.in_2d,
        )

        assert (frame == np.array([i for i in range(9)])).all()

        corner_frame = np.array([0, 1, 3, 4, -1, -1, -1, -1, -1])

        frame, kernel_frame = simple_convolver.frame_at_coordinates_jit(
            coordinates=(1, 1),
            mask=simple_mask_5x5,
            mask_index_array=simple_convolver.mask_index_array,
            kernel_2d=simple_convolver.kernel.in_2d,
        )

        assert (frame == corner_frame).all()

    def test__kernel_frame_at_coords(self, simple_mask_5x5, simple_convolver):

        frame, kernel_frame = simple_convolver.frame_at_coordinates_jit(
            coordinates=(2, 2),
            mask=simple_mask_5x5,
            mask_index_array=simple_convolver.mask_index_array,
            kernel_2d=simple_convolver.kernel.in_2d,
        )

        assert (
            kernel_frame == np.array([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]])
        ).all()

        frame, kernel_frame = simple_convolver.frame_at_coordinates_jit(
            coordinates=(1, 1),
            mask=simple_mask_5x5,
            mask_index_array=simple_convolver.mask_index_array,
            kernel_2d=simple_convolver.kernel.in_2d,
        )

        assert (
            kernel_frame == np.array([5.0, 6.0, 8.0, 9.0, -1, -1, -1, -1, -1])
        ).all()

    def test__simple_square(self, simple_convolver):
        assert 9 == len(simple_convolver.image_frame_1d_indexes)

        assert (
            simple_convolver.image_frame_1d_indexes[4]
            == np.array([i for i in range(9)])
        ).all()

    def test__frame_5x5_kernel__at_coords(self, simple_mask_7x7):
        convolver = aa.Convolver(
            mask=simple_mask_7x7,
            kernel=aa.Kernel.manual_2d(
                [
                    [1.0, 2.0, 3.0, 4.0, 5.0],
                    [6.0, 7.0, 8.0, 9.0, 10.0],
                    [11.0, 12.0, 13.0, 14.0, 15.0],
                    [16.0, 17.0, 18.0, 19.0, 20.0],
                    [21.0, 22.0, 23.0, 24.0, 25.0],
                ]
            ),
        )

        frame, kernel_frame = convolver.frame_at_coordinates_jit(
            coordinates=(2, 2),
            mask=simple_mask_7x7,
            mask_index_array=convolver.mask_index_array,
            kernel_2d=convolver.kernel.in_2d,
        )

        assert (
            kernel_frame
            == np.array(
                [
                    13.0,
                    14.0,
                    15.0,
                    18.0,
                    19.0,
                    20.0,
                    23.0,
                    24.0,
                    25.0,
                    -1.0,
                    -1.0,
                    -1.0,
                    -1.0,
                    -1.0,
                    -1.0,
                    -1.0,
                    -1.0,
                    -1.0,
                    -1.0,
                    -1.0,
                    -1.0,
                    -1.0,
                    -1.0,
                    -1.0,
                    -1.0,
                ]
            )
        ).all()

        frame, kernel_frame = convolver.frame_at_coordinates_jit(
            coordinates=(3, 2),
            mask=simple_mask_7x7,
            mask_index_array=convolver.mask_index_array,
            kernel_2d=convolver.kernel.in_2d,
        )

        assert (
            kernel_frame
            == np.array(
                [
                    8.0,
                    9.0,
                    10.0,
                    13.0,
                    14.0,
                    15.0,
                    18.0,
                    19.0,
                    20.0,
                    -1.0,
                    -1.0,
                    -1.0,
                    -1.0,
                    -1.0,
                    -1.0,
                    -1.0,
                    -1.0,
                    -1.0,
                    -1.0,
                    -1.0,
                    -1.0,
                    -1.0,
                    -1.0,
                    -1.0,
                    -1,
                ]
            )
        ).all()

        frame, kernel_frame = convolver.frame_at_coordinates_jit(
            coordinates=(3, 3),
            mask=simple_mask_7x7,
            mask_index_array=convolver.mask_index_array,
            kernel_2d=convolver.kernel.in_2d,
        )

        assert (
            kernel_frame
            == np.array(
                [
                    7.0,
                    8.0,
                    9.0,
                    12.0,
                    13.0,
                    14.0,
                    17.0,
                    18.0,
                    19.0,
                    -1.0,
                    -1.0,
                    -1.0,
                    -1.0,
                    -1.0,
                    -1.0,
                    -1.0,
                    -1.0,
                    -1.0,
                    -1.0,
                    -1.0,
                    -1.0,
                    -1.0,
                    -1.0,
                    -1.0,
                    -1,
                ]
            )
        ).all()


class TestImageFrameIndexes:
    def test__masked_cross__3x3_kernel(self, cross_convolver):
        assert 5 == len(cross_convolver.image_frame_1d_indexes)

        assert (
            cross_convolver.image_frame_1d_indexes[0]
            == np.array([0, 1, 2, 3, -1, -1, -1, -1, -1])
        ).all()
        assert (
            cross_convolver.image_frame_1d_indexes[1]
            == np.array([0, 1, 2, 4, -1, -1, -1, -1, -1])
        ).all()
        assert (
            cross_convolver.image_frame_1d_indexes[2]
            == np.array([0, 1, 2, 3, 4, -1, -1, -1, -1])
        ).all()
        assert (
            cross_convolver.image_frame_1d_indexes[3]
            == np.array([0, 2, 3, 4, -1, -1, -1, -1, -1])
        ).all()
        assert (
            cross_convolver.image_frame_1d_indexes[4]
            == np.array([1, 2, 3, 4, -1, -1, -1, -1, -1])
        ).all()

    def test__masked_square__3x5_kernel__loses_edge_of_top_and_bottom_rows(
        self, simple_mask_7x7
    ):
        convolver = aa.Convolver(
            mask=simple_mask_7x7, kernel=aa.Kernel.ones(shape_2d=(3, 5))
        )

        assert (
            convolver.image_frame_1d_indexes[0]
            == np.array([0, 1, 2, 3, 4, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1])
        ).all()
        assert (
            convolver.image_frame_1d_indexes[1]
            == np.array([0, 1, 2, 3, 4, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1])
        ).all()
        assert (
            convolver.image_frame_1d_indexes[2]
            == np.array([0, 1, 2, 3, 4, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1])
        ).all()
        assert (
            convolver.image_frame_1d_indexes[3]
            == np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, -1, -1, -1, -1, -1, -1])
        ).all()
        assert (
            convolver.image_frame_1d_indexes[4]
            == np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, -1, -1, -1, -1, -1, -1])
        ).all()
        assert (
            convolver.image_frame_1d_indexes[5]
            == np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, -1, -1, -1, -1, -1, -1])
        ).all()
        assert (
            convolver.image_frame_1d_indexes[6]
            == np.array([3, 4, 5, 6, 7, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1])
        ).all()
        assert (
            convolver.image_frame_1d_indexes[7]
            == np.array([3, 4, 5, 6, 7, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1])
        ).all()
        assert (
            convolver.image_frame_1d_indexes[8]
            == np.array([3, 4, 5, 6, 7, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1])
        ).all()

    def test__masked_square__5x3_kernel__loses_edge_of_left_and_right_columns(
        self, simple_mask_7x7
    ):
        convolver = aa.Convolver(
            mask=simple_mask_7x7, kernel=aa.Kernel.ones(shape_2d=(5, 3))
        )

        assert (
            convolver.image_frame_1d_indexes[0]
            == np.array([0, 1, 3, 4, 6, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1])
        ).all()
        assert (
            convolver.image_frame_1d_indexes[1]
            == np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, -1, -1, -1, -1, -1, -1])
        ).all()
        assert (
            convolver.image_frame_1d_indexes[2]
            == np.array([1, 2, 4, 5, 7, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1])
        ).all()
        assert (
            convolver.image_frame_1d_indexes[3]
            == np.array([0, 1, 3, 4, 6, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1])
        ).all()
        assert (
            convolver.image_frame_1d_indexes[4]
            == np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, -1, -1, -1, -1, -1, -1])
        ).all()
        assert (
            convolver.image_frame_1d_indexes[5]
            == np.array([1, 2, 4, 5, 7, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1])
        ).all()
        assert (
            convolver.image_frame_1d_indexes[6]
            == np.array([0, 1, 3, 4, 6, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1])
        ).all()
        assert (
            convolver.image_frame_1d_indexes[7]
            == np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, -1, -1, -1, -1, -1, -1])
        ).all()
        assert (
            convolver.image_frame_1d_indexes[8]
            == np.array([1, 2, 4, 5, 7, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1])
        ).all()

    def test__masked_square__7x7_kernel(self, simple_mask_7x7):
        convolver = aa.Convolver(
            mask=simple_mask_7x7, kernel=aa.Kernel.ones(shape_2d=(5, 5))
        )

        assert (
            convolver.image_frame_1d_indexes[0]
            == np.array(
                [
                    0,
                    1,
                    2,
                    3,
                    4,
                    5,
                    6,
                    7,
                    8,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                ]
            )
        ).all()
        assert (
            convolver.image_frame_1d_indexes[1]
            == np.array(
                [
                    0,
                    1,
                    2,
                    3,
                    4,
                    5,
                    6,
                    7,
                    8,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                ]
            )
        ).all()
        assert (
            convolver.image_frame_1d_indexes[2]
            == np.array(
                [
                    0,
                    1,
                    2,
                    3,
                    4,
                    5,
                    6,
                    7,
                    8,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                ]
            )
        ).all()
        assert (
            convolver.image_frame_1d_indexes[3]
            == np.array(
                [
                    0,
                    1,
                    2,
                    3,
                    4,
                    5,
                    6,
                    7,
                    8,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                ]
            )
        ).all()
        assert (
            convolver.image_frame_1d_indexes[4]
            == np.array(
                [
                    0,
                    1,
                    2,
                    3,
                    4,
                    5,
                    6,
                    7,
                    8,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                ]
            )
        ).all()
        assert (
            convolver.image_frame_1d_indexes[5]
            == np.array(
                [
                    0,
                    1,
                    2,
                    3,
                    4,
                    5,
                    6,
                    7,
                    8,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                ]
            )
        ).all()
        assert (
            convolver.image_frame_1d_indexes[6]
            == np.array(
                [
                    0,
                    1,
                    2,
                    3,
                    4,
                    5,
                    6,
                    7,
                    8,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                ]
            )
        ).all()
        assert (
            convolver.image_frame_1d_indexes[7]
            == np.array(
                [
                    0,
                    1,
                    2,
                    3,
                    4,
                    5,
                    6,
                    7,
                    8,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                ]
            )
        ).all()
        assert (
            convolver.image_frame_1d_indexes[8]
            == np.array(
                [
                    0,
                    1,
                    2,
                    3,
                    4,
                    5,
                    6,
                    7,
                    8,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                ]
            )
        ).all()


class TestImageFrameKernels:
    def test_simple_square(self, simple_convolver):
        assert 9 == len(simple_convolver.image_frame_1d_indexes)

        assert (
            simple_convolver.image_frame_1d_kernels[4]
            == np.array([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]])
        ).all()

    def test_masked_square__3x5_kernel__loses_edge_of_top_and_bottom_rows(
        self, simple_mask_7x7
    ):
        convolver = aa.Convolver(
            mask=simple_mask_7x7,
            kernel=aa.Kernel.manual_2d(
                [
                    [1.0, 2.0, 3.0, 4.0, 5.0],
                    [6.0, 7.0, 8.0, 9.0, 10.0],
                    [11.0, 12.0, 13.0, 14.0, 15.0],
                ]
            ),
        )

        assert (
            convolver.image_frame_1d_kernels[0]
            == np.array(
                [8.0, 9.0, 10.0, 13.0, 14.0, 15.0, -1, -1, -1, -1, -1, -1, -1, -1, -1]
            )
        ).all()
        assert (
            convolver.image_frame_1d_kernels[1]
            == np.array(
                [7.0, 8.0, 9.0, 12.0, 13.0, 14.0, -1, -1, -1, -1, -1, -1, -1, -1, -1]
            )
        ).all()
        assert (
            convolver.image_frame_1d_kernels[2]
            == np.array(
                [6.0, 7.0, 8.0, 11.0, 12.0, 13.0, -1, -1, -1, -1, -1, -1, -1, -1, -1]
            )
        ).all()
        assert (
            convolver.image_frame_1d_kernels[3]
            == np.array(
                [
                    3.0,
                    4.0,
                    5.0,
                    8.0,
                    9.0,
                    10.0,
                    13.0,
                    14.0,
                    15.0,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                ]
            )
        ).all()
        assert (
            convolver.image_frame_1d_kernels[4]
            == np.array(
                [2.0, 3.0, 4.0, 7.0, 8.0, 9.0, 12.0, 13.0, 14.0, -1, -1, -1, -1, -1, -1]
            )
        ).all()
        assert (
            convolver.image_frame_1d_kernels[5]
            == np.array(
                [1.0, 2.0, 3.0, 6.0, 7.0, 8.0, 11.0, 12.0, 13.0, -1, -1, -1, -1, -1, -1]
            )
        ).all()
        assert (
            convolver.image_frame_1d_kernels[6]
            == np.array(
                [3.0, 4.0, 5.0, 8.0, 9.0, 10.0, -1, -1, -1, -1, -1, -1, -1, -1, -1]
            )
        ).all()
        assert (
            convolver.image_frame_1d_kernels[7]
            == np.array(
                [2.0, 3.0, 4.0, 7.0, 8.0, 9.0, -1, -1, -1, -1, -1, -1, -1, -1, -1]
            )
        ).all()
        assert (
            convolver.image_frame_1d_kernels[8]
            == np.array(
                [1.0, 2.0, 3.0, 6.0, 7.0, 8.0, -1, -1, -1, -1, -1, -1, -1, -1, -1]
            )
        ).all()

    def test_masked_square__5x3_kernel__loses_edge_of_left_and_right_columns(
        self, simple_mask_7x7
    ):
        convolver = aa.Convolver(
            mask=simple_mask_7x7,
            kernel=aa.Kernel.manual_2d(
                [
                    [1.0, 2.0, 3.0],
                    [4.0, 5.0, 6.0],
                    [7.0, 8.0, 9.0],
                    [10.0, 11.0, 12.0],
                    [13.0, 14.0, 15.0],
                ]
            ),
        )

        assert (
            convolver.image_frame_1d_kernels[0]
            == np.array(
                [8.0, 9.0, 11.0, 12.0, 14.0, 15.0, -1, -1, -1, -1, -1, -1, -1, -1, -1]
            )
        ).all()
        assert (
            convolver.image_frame_1d_kernels[1]
            == np.array(
                [
                    7.0,
                    8.0,
                    9.0,
                    10.0,
                    11.0,
                    12.0,
                    13.0,
                    14.0,
                    15.0,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                ]
            )
        ).all()
        assert (
            convolver.image_frame_1d_kernels[2]
            == np.array(
                [7.0, 8.0, 10.0, 11.0, 13.0, 14.0, -1, -1, -1, -1, -1, -1, -1, -1, -1]
            )
        ).all()
        assert (
            convolver.image_frame_1d_kernels[3]
            == np.array(
                [5.0, 6.0, 8.0, 9.0, 11.0, 12.0, -1, -1, -1, -1, -1, -1, -1, -1, -1]
            )
        ).all()
        assert (
            convolver.image_frame_1d_kernels[4]
            == np.array(
                [4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, -1, -1, -1, -1, -1, -1]
            )
        ).all()
        assert (
            convolver.image_frame_1d_kernels[5]
            == np.array(
                [4.0, 5.0, 7.0, 8.0, 10.0, 11.0, -1, -1, -1, -1, -1, -1, -1, -1, -1]
            )
        ).all()
        assert (
            convolver.image_frame_1d_kernels[6]
            == np.array(
                [2.0, 3.0, 5.0, 6.0, 8.0, 9.0, -1, -1, -1, -1, -1, -1, -1, -1, -1]
            )
        ).all()
        assert (
            convolver.image_frame_1d_kernels[7]
            == np.array(
                [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, -1, -1, -1, -1, -1, -1]
            )
        ).all()
        assert (
            convolver.image_frame_1d_kernels[8]
            == np.array(
                [1.0, 2.0, 4.0, 5.0, 7.0, 8.0, -1, -1, -1, -1, -1, -1, -1, -1, -1]
            )
        ).all()


class TestBlurringFrameIndxes:
    def test__blurring_region_3x3_kernel(self, cross_mask):

        convolver = aa.Convolver(
            mask=cross_mask, kernel=aa.Kernel.ones(shape_2d=(3, 3))
        )

        assert (
            convolver.blurring_frame_1d_indexes[4]
            == np.array([0, 1, 2, -1, -1, -1, -1, -1, -1])
        ).all()
        assert (
            convolver.blurring_frame_1d_indexes[5]
            == np.array([0, 2, 3, -1, -1, -1, -1, -1, -1])
        ).all()
        assert (
            convolver.blurring_frame_1d_indexes[10]
            == np.array([1, 2, 4, -1, -1, -1, -1, -1, -1])
        ).all()
        assert (
            convolver.blurring_frame_1d_indexes[11]
            == np.array([2, 3, 4, -1, -1, -1, -1, -1, -1])
        ).all()


class TestBlurringFrameKernels:
    def test__blurring_region_3x3_kernel(self, cross_mask):

        convolver = aa.Convolver(
            mask=cross_mask,
            kernel=aa.Kernel.manual_2d(
                [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]
            ),
        )

        assert (
            convolver.blurring_frame_1d_kernels[4]
            == np.array([6.0, 8.0, 9.0, -1, -1, -1, -1, -1, -1])
        ).all()
        assert (
            convolver.blurring_frame_1d_kernels[5]
            == np.array([4.0, 7.0, 8.0, -1, -1, -1, -1, -1, -1])
        ).all()
        assert (
            convolver.blurring_frame_1d_kernels[10]
            == np.array([2.0, 3.0, 6.0, -1, -1, -1, -1, -1, -1])
        ).all()
        assert (
            convolver.blurring_frame_1d_kernels[11]
            == np.array([1.0, 2.0, 4.0, -1, -1, -1, -1, -1, -1])
        ).all()


class TestFrameLengths:
    def test__frames_are_from_examples_above__lengths_are_right(self, simple_mask_7x7):
        convolver = aa.Convolver(
            mask=simple_mask_7x7, kernel=aa.Kernel.ones(shape_2d=(3, 5))
        )

        # convolver_image.image_frame_indexes[0] == np.array([0, 1, 2, 3, 4, 5])
        # convolver_image.image_frame_indexes[1] == np.array([0, 1, 2, 3, 4, 5])
        # convolver_image.image_frame_indexes[2] == np.array([0, 1, 2, 3, 4, 5])
        # convolver_image.image_frame_indexes[3] == np.array([0, 1, 2, 3, 4, 5, 6, 7, 8])
        # (convolver_image.image_frame_indexes[4] == np.array([0, 1, 2, 3, 4, 5, 6, 7, 8])
        # convolver_image.image_frame_indexes[5] == np.array([0, 1, 2, 3, 4, 5, 6, 7, 8])
        # convolver_image.image_frame_indexes[6] == np.array([3, 4, 5, 6, 7, 8])
        # convolver_image.image_frame_indexes[7] == np.array([3, 4, 5, 6, 7, 8])
        # convolver_image.image_frame_indexes[8] == np.array([3, 4, 5, 6, 7, 8])

        assert (
            convolver.image_frame_1d_lengths == np.array([6, 6, 6, 9, 9, 9, 6, 6, 6])
        ).all()


class TestConvolveMappingMatrix:
    def test__asymetric_convolver__matrix_blurred_correctly(self):

        mask = np.array(
            [
                [True, True, True, True, True, True],
                [True, False, False, False, False, True],
                [True, False, False, False, False, True],
                [True, False, False, False, False, True],
                [True, False, False, False, False, True],
                [True, True, True, True, True, True],
            ]
        )

        asymmetric_kernel = aa.Kernel.manual_2d(
            array=[[0, 0.0, 0], [0.4, 0.2, 0.3], [0, 0.1, 0]]
        )

        convolver = aa.Convolver(mask=mask, kernel=asymmetric_kernel)

        mapping = np.array(
            [
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0],
                [
                    0,
                    1,
                    0,
                ],  # The 0.3 should be 'chopped' from this pixel as it is on the right-most edge
                [0, 0, 0],
                [1, 0, 0],
                [0, 0, 1],
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0],
            ]
        )

        blurred_mapping = convolver.convolve_mapping_matrix(mapping)

        assert (
            blurred_mapping
            == np.array(
                [
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 0.4, 0],
                    [0, 0.2, 0],
                    [0.4, 0, 0],
                    [0.2, 0, 0.4],
                    [0.3, 0, 0.2],
                    [0, 0.1, 0.3],
                    [0, 0, 0],
                    [0.1, 0, 0],
                    [0, 0, 0.1],
                    [0, 0, 0],
                ]
            )
        ).all()

    def test__asymetric_convolver__multiple_overlapping_blurred_entires_in_matrix(self):

        mask = np.array(
            [
                [True, True, True, True, True, True],
                [True, False, False, False, False, True],
                [True, False, False, False, False, True],
                [True, False, False, False, False, True],
                [True, False, False, False, False, True],
                [True, True, True, True, True, True],
            ]
        )

        asymmetric_kernel = aa.Kernel.manual_2d(
            array=[[0, 0.0, 0], [0.4, 0.2, 0.3], [0, 0.1, 0]]
        )

        convolver = aa.Convolver(mask=mask, kernel=asymmetric_kernel)

        mapping = np.array(
            [
                [0, 1, 0],
                [0, 1, 0],
                [0, 1, 0],
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0],
                [
                    0,
                    1,
                    0,
                ],  # The 0.3 should be 'chopped' from this pixel as it is on the right-most edge
                [1, 0, 0],
                [1, 0, 0],
                [0, 0, 1],
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0],
            ]
        )

        blurred_mapping = convolver.convolve_mapping_matrix(mapping)

        assert blurred_mapping == pytest.approx(
            np.array(
                [
                    [0, 0.6, 0],
                    [0, 0.9, 0],
                    [0, 0.5, 0],
                    [0, 0.3, 0],
                    [0, 0.1, 0],
                    [0, 0.1, 0],
                    [0, 0.5, 0],
                    [0, 0.2, 0],
                    [0.6, 0, 0],
                    [0.5, 0, 0.4],
                    [0.3, 0, 0.2],
                    [0, 0.1, 0.3],
                    [0.1, 0, 0],
                    [0.1, 0, 0],
                    [0, 0, 0.1],
                    [0, 0, 0],
                ]
            ),
            1e-4,
        )


class TestConvolution:
    def test__cross_mask_with_blurring_entries__returns_array(self):

        cross_mask = aa.Mask.manual(
            mask=[
                [True, True, True, True, True],
                [True, True, False, True, True],
                [True, False, False, False, True],
                [True, True, False, True, True],
                [True, True, True, True, True],
            ],
            pixel_scales=0.1,
            sub_size=1,
        )

        kernel = aa.Kernel.manual_2d(array=[[0, 0.2, 0], [0.2, 0.4, 0.2], [0, 0.2, 0]])

        convolver = aa.Convolver(mask=cross_mask, kernel=kernel)

        image_array = aa.Array.manual_mask(array=[1, 0, 0, 0, 0], mask=cross_mask)

        blurring_mask = cross_mask.regions.blurring_mask_from_kernel_shape(
            kernel_shape_2d=kernel.shape_2d
        )

        blurring_array = aa.Array.manual_mask(
            array=[0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], mask=blurring_mask
        )

        result = convolver.convolved_image_from_image_and_blurring_image(
            image=image_array, blurring_image=blurring_array
        )

        assert (np.round(result, 1) == np.array([0.6, 0.2, 0.2, 0.0, 0.0])).all()


class TestCompareToFull2dConv:
    def test__compare_convolver_to_2d_convolution(self):
        # Setup a blurred datas_, using the PSF to perform the convolution in 2D, then masks it to make a 1d array.

        import scipy.signal

        kernel = aa.Kernel.manual_2d(array=np.arange(49).reshape(7, 7))
        image = aa.Array.manual_2d(array=np.arange(900).reshape(30, 30))
        blurred_image = scipy.signal.convolve2d(image.in_2d, kernel.in_2d, mode="same")
        blurred_image = aa.Array.manual_2d(array=blurred_image)

        mask = aa.Mask.circular(
            shape_2d=(30, 30), pixel_scales=(1.0, 1.0), sub_size=1, radius=4.0
        )

        masked_image = aa.Array.manual_mask(
            array=image.in_2d, mask=mask, store_in_1d=True
        )

        blurred_masked_image = aa.Array.manual_mask(
            array=blurred_image.in_2d, mask=mask, store_in_1d=True
        )

        # Now reproduce this datas_ using the frame convolver_image

        blurring_mask = mask.regions.blurring_mask_from_kernel_shape(
            kernel_shape_2d=kernel.shape_2d
        )
        convolver = aa.Convolver(mask=mask, kernel=kernel)

        blurring_image = aa.Array.manual_mask(
            array=image.in_2d, mask=blurring_mask, store_in_1d=True
        )

        blurred_masked_im_1 = convolver.convolved_image_from_image_and_blurring_image(
            image=masked_image, blurring_image=blurring_image
        )

        assert blurred_masked_image == pytest.approx(blurred_masked_im_1, 1e-4)
