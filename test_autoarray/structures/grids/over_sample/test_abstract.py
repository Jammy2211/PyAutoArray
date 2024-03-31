import numpy as np
import autoarray as aa


def test__sub_pixels_in_mask():

    over_sample = aa.OverSampleUniform()

    mask = aa.Mask2D.all_false(shape_native=(5, 5), pixel_scales=1.0)

    assert over_sample.sub_pixels_in_mask_from(mask=mask, sub_size=1) == 25

    mask = aa.Mask2D.all_false(shape_native=(5, 5), pixel_scales=1.0)

    assert over_sample.sub_pixels_in_mask_from(mask=mask, sub_size=2) == 100

    mask = aa.Mask2D.all_false(shape_native=(10, 10), pixel_scales=1.0)

    assert over_sample.sub_pixels_in_mask_from(mask=mask, sub_size=3) == 900


def test__structure_2d_from():
    mask = np.array(
        [
            [True, True, True, True],
            [True, False, False, True],
            [True, False, False, True],
            [True, True, True, True],
        ]
    )

    mask = aa.Mask2D(mask=mask, pixel_scales=(1.0, 1.0))

    over_sample = aa.OverSampleUniform()

    result = over_sample.structure_2d_from(
        result=np.array([1.0, 2.0, 3.0, 4.0]), mask=mask
    )

    assert isinstance(result, aa.Array2D)
    assert (
        result.native
        == np.array(
            [
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 2.0, 0.0],
                [0.0, 3.0, 4.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
            ]
        )
    ).all()

    result = over_sample.structure_2d_from(
        result=np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]]), mask=mask
    )

    assert isinstance(result, aa.Grid2D)
    assert (
        result.native
        == np.array(
            [
                [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
                [[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [0.0, 0.0]],
                [[0.0, 0.0], [3.0, 3.0], [4.0, 4.0], [0.0, 0.0]],
                [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
            ]
        )
    ).all()


def test__structure_2d_list_from():
    mask = np.array(
        [
            [True, True, True, True],
            [True, False, False, True],
            [True, False, False, True],
            [True, True, True, True],
        ]
    )

    mask = aa.Mask2D(mask=mask, pixel_scales=(1.0, 1.0))

    over_sample = aa.OverSampleUniform()

    result = over_sample.structure_2d_list_from(
        result_list=[np.array([1.0, 2.0, 3.0, 4.0])], mask=mask
    )

    assert isinstance(result[0], aa.Array2D)
    assert (
        result[0].native
        == np.array(
            [
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 2.0, 0.0],
                [0.0, 3.0, 4.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
            ]
        )
    ).all()

    result = over_sample.structure_2d_list_from(
        result_list=[np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]])],
        mask=mask,
    )

    assert isinstance(result[0], aa.Grid2D)
    assert (
        result[0].native
        == np.array(
            [
                [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
                [[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [0.0, 0.0]],
                [[0.0, 0.0], [3.0, 3.0], [4.0, 4.0], [0.0, 0.0]],
                [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
            ]
        )
    ).all()
