import numpy as np
import pytest

import autoarray as aa
from autoarray import exc


@pytest.fixture(name="simple_mask_2d_7x7")
def make_simple_mask_2d_7x7():

    mask = [
        [True, True, True, True, True, True, True],
        [True, True, True, True, True, True, True],
        [True, True, False, False, False, True, True],
        [True, True, False, False, False, True, True],
        [True, True, False, False, False, True, True],
        [True, True, True, True, True, True, True],
        [True, True, True, True, True, True, True],
    ]

    return aa.Mask2D(mask=mask, pixel_scales=1.0, sub_size=1)


@pytest.fixture(name="simple_mask_5x5")
def make_simple_mask_5x5():

    mask = [
        [True, True, True, True, True],
        [True, False, False, False, True],
        [True, False, False, False, True],
        [True, False, False, False, True],
        [True, True, True, True, True],
    ]

    return aa.Mask2D(mask=mask, pixel_scales=1.0, sub_size=1)


@pytest.fixture(name="cross_mask")
def make_cross_mask():
    mask = np.full((5, 5), True)

    mask[2, 2] = False
    mask[1, 2] = False
    mask[3, 2] = False
    mask[2, 1] = False
    mask[2, 3] = False

    return aa.Mask2D(mask=mask, pixel_scales=1.0, sub_size=1)


def test__numbering__uses_mask_correctly(simple_mask_5x5, cross_mask):

    convolver = aa.Convolver(
        mask=simple_mask_5x5,
        kernel=aa.Kernel2D.ones(shape_native=(1, 1), pixel_scales=1.0),
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

    convolver = aa.Convolver(
        mask=cross_mask, kernel=aa.Kernel2D.ones(shape_native=(1, 1), pixel_scales=1.0)
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


def test__even_kernel_failure():

    with pytest.raises(exc.KernelException):
        aa.Convolver(
            mask=np.full((3, 3), False),
            kernel=aa.Kernel2D.ones(shape_native=(2, 2), pixel_scales=1.0),
        )


def test__frame_extraction__frame_and_kernel_frame_at_coords(simple_mask_5x5):

    convolver = aa.Convolver(
        mask=simple_mask_5x5,
        kernel=aa.Kernel2D.manual_native(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], pixel_scales=1.0
        ),
    )

    frame, kernel_frame = convolver.frame_at_coordinates_jit(
        coordinates=(2, 2),
        mask=simple_mask_5x5,
        mask_index_array=convolver.mask_index_array,
        kernel_2d=convolver.kernel.native,
    )

    assert (frame == np.array([i for i in range(9)])).all()

    assert (
        kernel_frame == np.array([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]])
    ).all()

    corner_frame = np.array([0, 1, 3, 4, -1, -1, -1, -1, -1])

    frame, kernel_frame = convolver.frame_at_coordinates_jit(
        coordinates=(1, 1),
        mask=simple_mask_5x5,
        mask_index_array=convolver.mask_index_array,
        kernel_2d=convolver.kernel.native,
    )

    assert (frame == corner_frame).all()

    frame, kernel_frame = convolver.frame_at_coordinates_jit(
        coordinates=(1, 1),
        mask=simple_mask_5x5,
        mask_index_array=convolver.mask_index_array,
        kernel_2d=convolver.kernel.native,
    )

    assert (kernel_frame == np.array([5.0, 6.0, 8.0, 9.0, -1, -1, -1, -1, -1])).all()

    assert 9 == len(convolver.image_frame_1d_indexes)

    assert (
        convolver.image_frame_1d_indexes[4] == np.array([i for i in range(9)])
    ).all()


def test__frame_extraction__more_complicated_frames(simple_mask_2d_7x7):

    convolver = aa.Convolver(
        mask=simple_mask_2d_7x7,
        kernel=aa.Kernel2D.manual_native(
            [
                [1.0, 2.0, 3.0, 4.0, 5.0],
                [6.0, 7.0, 8.0, 9.0, 10.0],
                [11.0, 12.0, 13.0, 14.0, 15.0],
                [16.0, 17.0, 18.0, 19.0, 20.0],
                [21.0, 22.0, 23.0, 24.0, 25.0],
            ],
            pixel_scales=1.0,
        ),
    )

    frame, kernel_frame = convolver.frame_at_coordinates_jit(
        coordinates=(2, 2),
        mask=simple_mask_2d_7x7,
        mask_index_array=convolver.mask_index_array,
        kernel_2d=convolver.kernel.native,
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
        mask=simple_mask_2d_7x7,
        mask_index_array=convolver.mask_index_array,
        kernel_2d=convolver.kernel.native,
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
        mask=simple_mask_2d_7x7,
        mask_index_array=convolver.mask_index_array,
        kernel_2d=convolver.kernel.native,
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


def test__image_frame_indexes__for_different_masks(cross_mask, simple_mask_2d_7x7):

    convolver = aa.Convolver(
        mask=cross_mask,
        kernel=aa.Kernel2D.manual_native(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], pixel_scales=1.0
        ),
    )

    assert 5 == len(convolver.image_frame_1d_indexes)

    assert (
        convolver.image_frame_1d_indexes[0]
        == np.array([0, 1, 2, 3, -1, -1, -1, -1, -1])
    ).all()
    assert (
        convolver.image_frame_1d_indexes[1]
        == np.array([0, 1, 2, 4, -1, -1, -1, -1, -1])
    ).all()
    assert (
        convolver.image_frame_1d_indexes[2] == np.array([0, 1, 2, 3, 4, -1, -1, -1, -1])
    ).all()
    assert (
        convolver.image_frame_1d_indexes[3]
        == np.array([0, 2, 3, 4, -1, -1, -1, -1, -1])
    ).all()
    assert (
        convolver.image_frame_1d_indexes[4]
        == np.array([1, 2, 3, 4, -1, -1, -1, -1, -1])
    ).all()

    convolver = aa.Convolver(
        mask=simple_mask_2d_7x7,
        kernel=aa.Kernel2D.ones(shape_native=(3, 5), pixel_scales=1.0),
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

    convolver = aa.Convolver(
        mask=simple_mask_2d_7x7,
        kernel=aa.Kernel2D.ones(shape_native=(5, 3), pixel_scales=1.0),
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

    convolver = aa.Convolver(
        mask=simple_mask_2d_7x7,
        kernel=aa.Kernel2D.ones(shape_native=(5, 5), pixel_scales=1.0),
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


def test_image_frame_kernels__different_shape_masks(
    simple_mask_5x5, simple_mask_2d_7x7
):

    convolver = aa.Convolver(
        mask=simple_mask_5x5,
        kernel=aa.Kernel2D.manual_native(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], pixel_scales=1.0
        ),
    )

    assert 9 == len(convolver.image_frame_1d_indexes)

    assert (
        convolver.image_frame_1d_kernels[4]
        == np.array([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]])
    ).all()

    convolver = aa.Convolver(
        mask=simple_mask_2d_7x7,
        kernel=aa.Kernel2D.manual_native(
            [
                [1.0, 2.0, 3.0, 4.0, 5.0],
                [6.0, 7.0, 8.0, 9.0, 10.0],
                [11.0, 12.0, 13.0, 14.0, 15.0],
            ],
            pixel_scales=1.0,
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
            [3.0, 4.0, 5.0, 8.0, 9.0, 10.0, 13.0, 14.0, 15.0, -1, -1, -1, -1, -1, -1]
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
        == np.array([3.0, 4.0, 5.0, 8.0, 9.0, 10.0, -1, -1, -1, -1, -1, -1, -1, -1, -1])
    ).all()
    assert (
        convolver.image_frame_1d_kernels[7]
        == np.array([2.0, 3.0, 4.0, 7.0, 8.0, 9.0, -1, -1, -1, -1, -1, -1, -1, -1, -1])
    ).all()
    assert (
        convolver.image_frame_1d_kernels[8]
        == np.array([1.0, 2.0, 3.0, 6.0, 7.0, 8.0, -1, -1, -1, -1, -1, -1, -1, -1, -1])
    ).all()

    convolver = aa.Convolver(
        mask=simple_mask_2d_7x7,
        kernel=aa.Kernel2D.manual_native(
            [
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
                [7.0, 8.0, 9.0],
                [10.0, 11.0, 12.0],
                [13.0, 14.0, 15.0],
            ],
            pixel_scales=1.0,
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
            [7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, -1, -1, -1, -1, -1, -1]
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
        == np.array([2.0, 3.0, 5.0, 6.0, 8.0, 9.0, -1, -1, -1, -1, -1, -1, -1, -1, -1])
    ).all()
    assert (
        convolver.image_frame_1d_kernels[7]
        == np.array(
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, -1, -1, -1, -1, -1, -1]
        )
    ).all()
    assert (
        convolver.image_frame_1d_kernels[8]
        == np.array([1.0, 2.0, 4.0, 5.0, 7.0, 8.0, -1, -1, -1, -1, -1, -1, -1, -1, -1])
    ).all()


def test__blurring_frame_indexes__blurring_region_3x3_kernel(cross_mask):

    convolver = aa.Convolver(
        mask=cross_mask, kernel=aa.Kernel2D.ones(shape_native=(3, 3), pixel_scales=1.0)
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


def test__blurring_frame_kernels__blurring_region_3x3_kernel(cross_mask):

    convolver = aa.Convolver(
        mask=cross_mask,
        kernel=aa.Kernel2D.manual_native(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], pixel_scales=1.0
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


def test__frame_lengths__frames_are_from_examples_above__lengths_are_right(
    simple_mask_2d_7x7,
):

    convolver = aa.Convolver(
        mask=simple_mask_2d_7x7,
        kernel=aa.Kernel2D.ones(shape_native=(3, 5), pixel_scales=1.0),
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


def test__convolve_mapping_matrix__asymetric_convolver__matrix_blurred_correctly():

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

    asymmetric_kernel = aa.Kernel2D.manual_native(
        array=[[0, 0.0, 0], [0.4, 0.2, 0.3], [0, 0.1, 0]], pixel_scales=1.0
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

    asymmetric_kernel = aa.Kernel2D.manual_native(
        array=[[0, 0.0, 0], [0.4, 0.2, 0.3], [0, 0.1, 0]], pixel_scales=1.0
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


def test__convolution__cross_mask_with_blurring_entries__returns_array():

    cross_mask = aa.Mask2D(
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

    kernel = aa.Kernel2D.manual_native(
        array=[[0, 0.2, 0], [0.2, 0.4, 0.2], [0, 0.2, 0]], pixel_scales=0.1
    )

    convolver = aa.Convolver(mask=cross_mask, kernel=kernel)

    image_array = aa.Array2D(array=[1, 0, 0, 0, 0], mask=cross_mask)

    blurring_mask = cross_mask.derive_mask.blurring_from(
        kernel_shape_native=kernel.shape_native
    )

    blurring_array = aa.Array2D(
        array=[0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], mask=blurring_mask
    )

    result = convolver.convolve_image(image=image_array, blurring_image=blurring_array)

    assert (np.round(result, 1) == np.array([0.6, 0.2, 0.2, 0.0, 0.0])).all()


def test__compare_to_full_2d_convolution():
    # Setup a blurred data, using the PSF to perform the convolution in 2D, then masks it to make a 1d array.

    import scipy.signal

    mask = aa.Mask2D.circular(
        shape_native=(30, 30), pixel_scales=(1.0, 1.0), sub_size=1, radius=4.0
    )
    kernel = aa.Kernel2D.manual_native(
        array=np.arange(49).reshape(7, 7), pixel_scales=1.0
    )
    image = aa.Array2D.without_mask(
        array=np.arange(900).reshape(30, 30), pixel_scales=1.0
    )

    blurred_image_via_scipy = scipy.signal.convolve2d(
        image.native, kernel.native, mode="same"
    )
    blurred_image_via_scipy = aa.Array2D.without_mask(
        array=blurred_image_via_scipy, pixel_scales=1.0
    )
    blurred_masked_image_via_scipy = aa.Array2D(
        array=blurred_image_via_scipy.native, mask=mask
    )

    # Now reproduce this data using the frame convolver_image

    masked_image = aa.Array2D(array=image.native, mask=mask)

    blurring_mask = mask.derive_mask.blurring_from(
        kernel_shape_native=kernel.shape_native
    )

    convolver = aa.Convolver(mask=mask, kernel=kernel)

    blurring_image = aa.Array2D(array=image.native, mask=blurring_mask)

    blurred_masked_im_1 = convolver.convolve_image(
        image=masked_image, blurring_image=blurring_image
    )

    assert blurred_masked_image_via_scipy == pytest.approx(blurred_masked_im_1, 1e-4)


def test__compare_to_full_2d_convolution__no_blurring_image():
    # Setup a blurred data, using the PSF to perform the convolution in 2D, then masks it to make a 1d array.

    import scipy.signal

    mask = aa.Mask2D.circular(
        shape_native=(30, 30), pixel_scales=(1.0, 1.0), sub_size=1, radius=4.0
    )
    kernel = aa.Kernel2D.manual_native(
        array=np.arange(49).reshape(7, 7), pixel_scales=1.0
    )
    image = aa.Array2D.without_mask(
        array=np.arange(900).reshape(30, 30), pixel_scales=1.0
    )

    blurring_mask = mask.derive_mask.blurring_from(
        kernel_shape_native=kernel.shape_native
    )
    blurred_image_via_scipy = scipy.signal.convolve2d(
        image.native * blurring_mask, kernel.native, mode="same"
    )
    blurred_image_via_scipy = aa.Array2D.without_mask(
        array=blurred_image_via_scipy, pixel_scales=1.0
    )
    blurred_masked_image_via_scipy = aa.Array2D(
        array=blurred_image_via_scipy.native, mask=mask
    )

    # Now reproduce this data using the frame convolver_image

    masked_image = aa.Array2D(array=image.native, mask=mask)

    convolver = aa.Convolver(mask=mask, kernel=kernel)

    blurred_masked_im_1 = convolver.convolve_image_no_blurring(image=masked_image)

    assert blurred_masked_image_via_scipy == pytest.approx(blurred_masked_im_1, 1e-4)
