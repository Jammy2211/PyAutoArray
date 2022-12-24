import os
from os import path
import numpy as np
import pytest
import shutil

import autoarray as aa
from autoarray import exc

test_data_dir = path.join(
    "{}".format(path.dirname(path.realpath(__file__))), "files", "mask"
)


def test__manual():

    mask = aa.Mask2D.manual(
        mask=[[False, False], [True, True]], pixel_scales=1.0, sub_size=1
    )

    assert type(mask) == aa.Mask2D
    assert (mask == np.array([[False, False], [True, True]])).all()
    assert mask.pixel_scales == (1.0, 1.0)
    assert mask.origin == (0.0, 0.0)
    assert mask.sub_size == 1
    assert (mask.geometry.extent == np.array([-1.0, 1.0, -1.0, 1.0])).all()

    mask = aa.Mask2D.manual(
        mask=[[False, False], [True, True]],
        pixel_scales=(2.0, 3.0),
        sub_size=2,
        origin=(0.0, 1.0),
    )

    assert type(mask) == aa.Mask2D
    assert (mask == np.array([[False, False], [True, True]])).all()
    assert mask.pixel_scales == (2.0, 3.0)
    assert mask.origin == (0.0, 1.0)
    assert mask.sub_size == 2


def test__manual__invert_is_true():

    mask = aa.Mask2D.manual(
        mask=[[False, False, True], [True, True, False]],
        pixel_scales=1.0,
        invert=True,
    )

    assert type(mask) == aa.Mask2D
    assert (mask == np.array([[True, True, False], [False, False, True]])).all()


def test__unmasked():

    mask = aa.Mask2D.unmasked(shape_native=(5, 5), pixel_scales=1.0, invert=False)

    assert mask.shape == (5, 5)
    assert (
        mask
        == np.array(
            [
                [False, False, False, False, False],
                [False, False, False, False, False],
                [False, False, False, False, False],
                [False, False, False, False, False],
                [False, False, False, False, False],
            ]
        )
    ).all()

    mask = aa.Mask2D.unmasked(
        shape_native=(3, 3),
        pixel_scales=(2.0, 2.5),
        invert=True,
        sub_size=4,
        origin=(1.0, 2.0),
    )

    assert mask.shape == (3, 3)
    assert (
        mask == np.array([[True, True, True], [True, True, True], [True, True, True]])
    ).all()

    assert mask.sub_size == 4
    assert mask.pixel_scales == (2.0, 2.5)
    assert mask.origin == (1.0, 2.0)


def test__circular():

    mask_via_util = aa.util.mask_2d.mask_2d_circular_from(
        shape_native=(5, 4), pixel_scales=(2.7, 2.7), radius=3.5, centre=(0.0, 0.0)
    )

    mask = aa.Mask2D.circular(
        shape_native=(5, 4),
        pixel_scales=(2.7, 2.7),
        sub_size=1,
        radius=3.5,
        centre=(0.0, 0.0),
    )

    assert (mask == mask_via_util).all()
    assert mask.origin == (0.0, 0.0)
    assert mask.mask_centre == pytest.approx((0.0, 0.0), 1.0e-8)

    mask = aa.Mask2D.circular(
        shape_native=(5, 4),
        pixel_scales=(2.7, 2.7),
        sub_size=1,
        radius=3.5,
        centre=(0.0, 0.0),
        invert=True,
    )

    assert (mask == np.invert(mask_via_util)).all()
    assert mask.origin == (0.0, 0.0)
    assert mask.mask_centre == (0.0, 0.0)


def test__circular_annular():
    mask_via_util = aa.util.mask_2d.mask_2d_circular_annular_from(
        shape_native=(5, 4),
        pixel_scales=(2.7, 2.7),
        inner_radius=0.8,
        outer_radius=3.5,
        centre=(0.0, 0.0),
    )

    mask = aa.Mask2D.circular_annular(
        shape_native=(5, 4),
        pixel_scales=(2.7, 2.7),
        sub_size=1,
        inner_radius=0.8,
        outer_radius=3.5,
        centre=(0.0, 0.0),
    )

    assert (mask == mask_via_util).all()
    assert mask.origin == (0.0, 0.0)
    assert mask.mask_centre == pytest.approx((0.0, 0.0), 1.0e-8)

    mask = aa.Mask2D.circular_annular(
        shape_native=(5, 4),
        pixel_scales=(2.7, 2.7),
        sub_size=1,
        inner_radius=0.8,
        outer_radius=3.5,
        centre=(0.0, 0.0),
        invert=True,
    )

    assert (mask == np.invert(mask_via_util)).all()
    assert mask.origin == (0.0, 0.0)
    assert mask.mask_centre == (0.0, 0.0)


def test__circular_anti_annular():
    mask_via_util = aa.util.mask_2d.mask_2d_circular_anti_annular_from(
        shape_native=(9, 9),
        pixel_scales=(1.2, 1.2),
        inner_radius=0.8,
        outer_radius=2.2,
        outer_radius_2_scaled=3.0,
        centre=(0.0, 0.0),
    )

    mask = aa.Mask2D.circular_anti_annular(
        shape_native=(9, 9),
        pixel_scales=(1.2, 1.2),
        sub_size=1,
        inner_radius=0.8,
        outer_radius=2.2,
        outer_radius_2=3.0,
        centre=(0.0, 0.0),
    )

    assert (mask == mask_via_util).all()
    assert mask.origin == (0.0, 0.0)
    assert mask.mask_centre == (0.0, 0.0)

    mask = aa.Mask2D.circular_anti_annular(
        shape_native=(9, 9),
        pixel_scales=(1.2, 1.2),
        sub_size=1,
        inner_radius=0.8,
        outer_radius=2.2,
        outer_radius_2=3.0,
        centre=(0.0, 0.0),
        invert=True,
    )

    assert (mask == np.invert(mask_via_util)).all()
    assert mask.origin == (0.0, 0.0)
    assert mask.mask_centre == (0.0, 0.0)


def test__elliptical():
    mask_via_util = aa.util.mask_2d.mask_2d_elliptical_from(
        shape_native=(8, 5),
        pixel_scales=(2.7, 2.7),
        major_axis_radius=5.7,
        axis_ratio=0.4,
        angle=40.0,
        centre=(0.0, 0.0),
    )

    mask = aa.Mask2D.elliptical(
        shape_native=(8, 5),
        pixel_scales=(2.7, 2.7),
        sub_size=1,
        major_axis_radius=5.7,
        axis_ratio=0.4,
        angle=40.0,
        centre=(0.0, 0.0),
    )

    assert (mask == mask_via_util).all()
    assert mask.origin == (0.0, 0.0)
    assert mask.mask_centre == pytest.approx((0.0, 0.0), 1.0e-8)

    mask = aa.Mask2D.elliptical(
        shape_native=(8, 5),
        pixel_scales=(2.7, 2.7),
        sub_size=1,
        major_axis_radius=5.7,
        axis_ratio=0.4,
        angle=40.0,
        centre=(0.0, 0.0),
        invert=True,
    )

    assert (mask == np.invert(mask_via_util)).all()
    assert mask.origin == (0.0, 0.0)
    assert mask.mask_centre == (0.0, 0.0)


def test__elliptical_annular():
    mask_via_util = aa.util.mask_2d.mask_2d_elliptical_annular_from(
        shape_native=(8, 5),
        pixel_scales=(2.7, 2.7),
        inner_major_axis_radius=2.1,
        inner_axis_ratio=0.6,
        inner_phi=20.0,
        outer_major_axis_radius=5.7,
        outer_axis_ratio=0.4,
        outer_phi=40.0,
        centre=(0.0, 0.0),
    )

    mask = aa.Mask2D.elliptical_annular(
        shape_native=(8, 5),
        pixel_scales=(2.7, 2.7),
        sub_size=1,
        inner_major_axis_radius=2.1,
        inner_axis_ratio=0.6,
        inner_phi=20.0,
        outer_major_axis_radius=5.7,
        outer_axis_ratio=0.4,
        outer_phi=40.0,
        centre=(0.0, 0.0),
    )

    assert (mask == mask_via_util).all()
    assert mask.origin == (0.0, 0.0)
    assert mask.mask_centre == pytest.approx((0.0, 0.0), 1.0e-8)

    mask = aa.Mask2D.elliptical_annular(
        shape_native=(8, 5),
        pixel_scales=(2.7, 2.7),
        sub_size=1,
        inner_major_axis_radius=2.1,
        inner_axis_ratio=0.6,
        inner_phi=20.0,
        outer_major_axis_radius=5.7,
        outer_axis_ratio=0.4,
        outer_phi=40.0,
        centre=(0.0, 0.0),
        invert=True,
    )

    assert (mask == np.invert(mask_via_util)).all()
    assert mask.origin == (0.0, 0.0)
    assert mask.mask_centre == (0.0, 0.0)


def test__from_pixel_coordinates():

    mask = aa.Mask2D.from_pixel_coordinates(
        shape_native=(5, 5), pixel_coordinates=[[2, 2]], pixel_scales=1.0, buffer=0
    )

    assert (
        mask
        == np.array(
            [
                [True, True, True, True, True],
                [True, True, True, True, True],
                [True, True, False, True, True],
                [True, True, True, True, True],
                [True, True, True, True, True],
            ]
        )
    ).all()

    mask = aa.Mask2D.from_pixel_coordinates(
        shape_native=(5, 5), pixel_coordinates=[[2, 2]], pixel_scales=1.0, buffer=1
    )

    assert (
        mask
        == np.array(
            [
                [True, True, True, True, True],
                [True, False, False, False, True],
                [True, False, False, False, True],
                [True, False, False, False, True],
                [True, True, True, True, True],
            ]
        )
    ).all()

    mask = aa.Mask2D.from_pixel_coordinates(
        shape_native=(7, 7),
        pixel_coordinates=[[2, 2], [5, 5]],
        pixel_scales=1.0,
        buffer=1,
    )

    assert (
        mask
        == np.array(
            [
                [True, True, True, True, True, True, True],
                [True, False, False, False, True, True, True],
                [True, False, False, False, True, True, True],
                [True, False, False, False, True, True, True],
                [True, True, True, True, False, False, False],
                [True, True, True, True, False, False, False],
                [True, True, True, True, False, False, False],
            ]
        )
    ).all()


def test__from_fits__output_to_fits():

    mask = aa.Mask2D.from_fits(
        file_path=path.join(test_data_dir, "3x3_ones.fits"),
        hdu=0,
        sub_size=1,
        pixel_scales=(1.0, 1.0),
    )

    output_data_dir = path.join(
        "{}".format(path.dirname(path.realpath(__file__))),
        "files",
        "array",
        "output_test",
    )

    if path.exists(output_data_dir):
        shutil.rmtree(output_data_dir)

    os.makedirs(output_data_dir)

    mask.output_to_fits(file_path=path.join(output_data_dir, "mask.fits"))

    mask = aa.Mask2D.from_fits(
        file_path=path.join(output_data_dir, "mask.fits"),
        hdu=0,
        sub_size=1,
        pixel_scales=(1.0, 1.0),
        origin=(2.0, 2.0),
    )

    assert (mask == np.ones((3, 3))).all()
    assert mask.pixel_scales == (1.0, 1.0)
    assert mask.origin == (2.0, 2.0)


def test__from_fits__with_resized_mask_shape():

    mask = aa.Mask2D.from_fits(
        file_path=path.join(test_data_dir, "3x3_ones.fits"),
        hdu=0,
        sub_size=1,
        pixel_scales=(1.0, 1.0),
        resized_mask_shape=(1, 1),
    )

    assert mask.shape_native == (1, 1)

    mask = aa.Mask2D.from_fits(
        file_path=path.join(test_data_dir, "3x3_ones.fits"),
        hdu=0,
        sub_size=1,
        pixel_scales=(1.0, 1.0),
        resized_mask_shape=(5, 5),
    )

    assert mask.shape_native == (5, 5)


def test__mask__input_is_1d_mask__no_shape_native__raises_exception():

    with pytest.raises(exc.MaskException):

        aa.Mask2D.manual(mask=[False, False, True], pixel_scales=1.0)

    with pytest.raises(exc.MaskException):

        aa.Mask2D.manual(mask=[False, False, True], pixel_scales=False)

    with pytest.raises(exc.MaskException):

        aa.Mask2D.manual(mask=[False, False, True], pixel_scales=1.0, sub_size=1)

    with pytest.raises(exc.MaskException):

        aa.Mask2D.manual(mask=[False, False, True], pixel_scales=False, sub_size=1)


def test__is_all_true():

    mask = aa.Mask2D.manual(mask=[[False, False], [False, False]], pixel_scales=1.0)

    assert mask.is_all_true is False

    mask = aa.Mask2D.manual(mask=[[False, False]], pixel_scales=1.0)

    assert mask.is_all_true is False

    mask = aa.Mask2D.manual(mask=[[False, True], [False, False]], pixel_scales=1.0)

    assert mask.is_all_true is False

    mask = aa.Mask2D.manual(mask=[[True, True], [True, True]], pixel_scales=1.0)

    assert mask.is_all_true is True


def test__is_all_false():

    mask = aa.Mask2D.manual(mask=[[False, False], [False, False]], pixel_scales=1.0)

    assert mask.is_all_false is True

    mask = aa.Mask2D.manual(mask=[[False, False]], pixel_scales=1.0)

    assert mask.is_all_false is True

    mask = aa.Mask2D.manual(mask=[[False, True], [False, False]], pixel_scales=1.0)

    assert mask.is_all_false is False

    mask = aa.Mask2D.manual(mask=[[True, True], [False, False]], pixel_scales=1.0)

    assert mask.is_all_false is False


def test__sub_shape_native():

    mask = aa.Mask2D.unmasked(shape_native=(5, 5), pixel_scales=1.0, sub_size=1)

    assert mask.sub_shape_native == (5, 5)

    mask = aa.Mask2D.unmasked(shape_native=(5, 5), pixel_scales=1.0, sub_size=2)

    assert mask.sub_shape_native == (10, 10)

    mask = aa.Mask2D.unmasked(shape_native=(10, 5), pixel_scales=1.0, sub_size=3)

    assert mask.sub_shape_native == (30, 15)


def test__resized_mask_from():

    mask = aa.Mask2D.unmasked(shape_native=(5, 5), pixel_scales=1.0)
    mask[2, 2] = True

    mask_resized = mask.resized_mask_from(new_shape=(7, 7))

    mask_resized_manual = np.full(fill_value=False, shape=(7, 7))
    mask_resized_manual[3, 3] = True

    assert (mask_resized == mask_resized_manual).all()

    mask = aa.Mask2D.unmasked(shape_native=(5, 5), pixel_scales=1.0)
    mask[2, 2] = True

    mask_resized = mask.resized_mask_from(new_shape=(3, 3))

    mask_resized_manual = np.full(fill_value=False, shape=(3, 3))
    mask_resized_manual[1, 1] = True

    assert (mask_resized == mask_resized_manual).all()





def test__shape_native_masked_pixels():

    mask = aa.Mask2D.manual(
        mask=[
            [True, True, True, True, True, True, True, True, True],
            [True, False, False, False, False, False, False, False, True],
            [True, False, True, True, True, True, True, False, True],
            [True, False, True, False, False, False, True, False, True],
            [True, False, True, False, True, False, True, False, True],
            [True, False, True, False, False, False, True, False, True],
            [True, False, True, True, True, True, True, False, True],
            [True, False, False, False, False, False, False, False, True],
            [True, True, True, True, True, True, True, True, True],
        ],
        pixel_scales=1.0,
    )

    assert mask.shape_native_masked_pixels == (7, 7)

    mask = aa.Mask2D.manual(
        mask=[
            [True, True, True, True, True, True, True, True, False],
            [True, False, False, False, False, False, False, False, True],
            [True, False, True, True, True, True, True, False, True],
            [True, False, True, False, False, False, True, False, True],
            [True, False, True, False, True, False, True, False, True],
            [True, False, True, False, False, False, True, False, True],
            [True, False, True, True, True, True, True, False, True],
            [True, False, False, False, False, False, False, False, True],
            [True, True, True, True, True, True, True, True, True],
        ],
        pixel_scales=1.0,
    )

    assert mask.shape_native_masked_pixels == (8, 8)

    mask = aa.Mask2D.manual(
        mask=[
            [True, True, True, True, True, True, True, False, True],
            [True, False, False, False, False, False, False, False, True],
            [True, False, True, True, True, True, True, False, True],
            [True, False, True, False, False, False, True, False, True],
            [True, False, True, False, True, False, True, False, True],
            [True, False, True, False, False, False, True, False, True],
            [True, False, True, True, True, True, True, False, True],
            [True, False, False, False, False, False, False, False, True],
            [True, True, True, True, True, True, True, True, True],
        ],
        pixel_scales=1.0,
    )

    assert mask.shape_native_masked_pixels == (8, 7)


def test__zoom_quantities():

    mask = aa.Mask2D.unmasked(shape_native=(3, 5), pixel_scales=(1.0, 1.0))
    assert mask.zoom_centre == (1.0, 2.0)
    assert mask.zoom_offset_pixels == (0, 0)
    assert mask.zoom_shape_native == (5, 5)

    mask = aa.Mask2D.unmasked(shape_native=(5, 3), pixel_scales=(1.0, 1.0))
    assert mask.zoom_centre == (2.0, 1.0)
    assert mask.zoom_offset_pixels == (0, 0)
    assert mask.zoom_shape_native == (5, 5)

    mask = aa.Mask2D.unmasked(shape_native=(4, 6), pixel_scales=(1.0, 1.0))
    assert mask.zoom_centre == (1.5, 2.5)
    assert mask.zoom_offset_pixels == (0, 0)
    assert mask.zoom_shape_native == (6, 6)

    mask = aa.Mask2D.unmasked(shape_native=(6, 4), pixel_scales=(1.0, 1.0))
    assert mask.zoom_centre == (2.5, 1.5)
    assert mask.zoom_offset_pixels == (0, 0)
    assert mask.zoom_shape_native == (6, 6)


def test__mask_is_single_false__extraction_centre_is_central_pixel():

    mask = aa.Mask2D.manual(
        mask=np.array([[False, True, True], [True, True, True], [True, True, True]]),
        pixel_scales=(1.0, 1.0),
    )
    assert mask.zoom_centre == (0, 0)
    assert mask.zoom_offset_pixels == (-1, -1)
    assert mask.zoom_shape_native == (1, 1)

    mask = aa.Mask2D.manual(
        mask=np.array([[True, True, False], [True, True, True], [True, True, True]]),
        pixel_scales=(1.0, 1.0),
    )
    assert mask.zoom_centre == (0, 2)
    assert mask.zoom_offset_pixels == (-1, 1)
    assert mask.zoom_shape_native == (1, 1)

    mask = aa.Mask2D.manual(
        mask=np.array([[True, True, True], [True, True, True], [False, True, True]]),
        pixel_scales=(1.0, 1.0),
    )
    assert mask.zoom_centre == (2, 0)
    assert mask.zoom_offset_pixels == (1, -1)
    assert mask.zoom_shape_native == (1, 1)

    mask = aa.Mask2D.manual(
        mask=np.array([[True, True, True], [True, True, True], [True, True, False]]),
        pixel_scales=(1.0, 1.0),
    )
    assert mask.zoom_centre == (2, 2)
    assert mask.zoom_offset_pixels == (1, 1)
    assert mask.zoom_shape_native == (1, 1)

    mask = aa.Mask2D.manual(
        mask=np.array([[True, False, True], [True, True, True], [True, True, True]]),
        pixel_scales=(1.0, 1.0),
    )
    assert mask.zoom_centre == (0, 1)
    assert mask.zoom_offset_pixels == (-1, 0)
    assert mask.zoom_shape_native == (1, 1)

    mask = aa.Mask2D.manual(
        mask=np.array([[True, True, True], [False, True, True], [True, True, True]]),
        pixel_scales=(1.0, 1.0),
    )
    assert mask.zoom_centre == (1, 0)
    assert mask.zoom_offset_pixels == (0, -1)
    assert mask.zoom_shape_native == (1, 1)

    mask = aa.Mask2D.manual(
        mask=np.array([[True, True, True], [True, True, False], [True, True, True]]),
        pixel_scales=(1.0, 1.0),
    )
    assert mask.zoom_centre == (1, 2)
    assert mask.zoom_offset_pixels == (0, 1)
    assert mask.zoom_shape_native == (1, 1)

    mask = aa.Mask2D.manual(
        mask=np.array([[True, True, True], [True, True, True], [True, False, True]]),
        pixel_scales=(1.0, 1.0),
    )
    assert mask.zoom_centre == (2, 1)
    assert mask.zoom_offset_pixels == (1, 0)
    assert mask.zoom_shape_native == (1, 1)


def test__mask_is_x2_false__extraction_centre_is_central_pixel():
    mask = aa.Mask2D.manual(
        mask=np.array([[False, True, True], [True, True, True], [True, True, False]]),
        pixel_scales=(1.0, 1.0),
    )
    assert mask.zoom_centre == (1, 1)
    assert mask.zoom_offset_pixels == (0, 0)
    assert mask.zoom_shape_native == (3, 3)

    mask = aa.Mask2D.manual(
        mask=np.array([[False, True, True], [True, True, True], [False, True, True]]),
        pixel_scales=(1.0, 1.0),
    )
    assert mask.zoom_centre == (1, 0)
    assert mask.zoom_offset_pixels == (0, -1)
    assert mask.zoom_shape_native == (3, 3)

    mask = aa.Mask2D.manual(
        mask=np.array([[False, True, False], [True, True, True], [True, True, True]]),
        pixel_scales=(1.0, 1.0),
    )
    assert mask.zoom_centre == (0, 1)
    assert mask.zoom_offset_pixels == (-1, 0)
    assert mask.zoom_shape_native == (3, 3)

    mask = aa.Mask2D.manual(
        mask=np.array([[False, False, True], [True, True, True], [True, True, True]]),
        pixel_scales=(1.0, 1.0),
    )
    assert mask.zoom_centre == (0, 0.5)
    assert mask.zoom_offset_pixels == (-1, -0.5)
    assert mask.zoom_shape_native == (1, 2)


def test__rectangular_mask():
    mask = aa.Mask2D.manual(
        mask=np.array(
            [
                [False, True, True, True],
                [True, True, True, True],
                [True, True, True, True],
            ]
        ),
        pixel_scales=(1.0, 1.0),
    )

    assert mask.zoom_centre == (0, 0)
    assert mask.zoom_offset_pixels == (-1.0, -1.5)

    mask = aa.Mask2D.manual(
        mask=np.array(
            [
                [True, True, True, True],
                [True, True, True, True],
                [True, True, True, False],
            ]
        ),
        pixel_scales=(1.0, 1.0),
    )

    assert mask.zoom_centre == (2, 3)
    assert mask.zoom_offset_pixels == (1.0, 1.5)

    mask = aa.Mask2D.manual(
        mask=np.array(
            [
                [True, True, True, True, True],
                [True, True, True, True, True],
                [True, True, True, True, False],
            ]
        ),
        pixel_scales=(1.0, 1.0),
    )

    assert mask.zoom_centre == (2, 4)
    assert mask.zoom_offset_pixels == (1, 2)

    mask = aa.Mask2D.manual(
        mask=np.array(
            [
                [True, True, True, True, True, True, True],
                [True, True, True, True, True, True, True],
                [True, True, True, True, True, True, False],
            ]
        ),
        pixel_scales=(1.0, 1.0),
    )

    assert mask.zoom_centre == (2, 6)
    assert mask.zoom_offset_pixels == (1, 3)

    mask = aa.Mask2D.manual(
        mask=np.array(
            [
                [True, True, True],
                [True, True, True],
                [True, True, True],
                [True, True, True],
                [True, True, False],
            ]
        ),
        pixel_scales=(1.0, 1.0),
    )

    assert mask.zoom_centre == (4, 2)
    assert mask.zoom_offset_pixels == (2, 1)

    mask = aa.Mask2D.manual(
        mask=np.array(
            [
                [True, True, True],
                [True, True, True],
                [True, True, True],
                [True, True, True],
                [True, True, True],
                [True, True, True],
                [True, True, False],
            ]
        ),
        pixel_scales=(1.0, 1.0),
    )

    assert mask.zoom_centre == (6, 2)
    assert mask.zoom_offset_pixels == (3, 1)


def test__zoom_mask_unmasked():

    mask = aa.Mask2D.manual(
        mask=np.array(
            [
                [False, True, True, True],
                [True, False, True, True],
                [True, True, True, True],
            ]
        ),
        pixel_scales=(1.0, 1.0),
    )

    zoom_mask = mask.zoom_mask_unmasked

    assert (zoom_mask == np.array([[False, False], [False, False]])).all()
    assert zoom_mask.origin == (0.5, -1.0)

    mask = aa.Mask2D.manual(
        mask=np.array(
            [
                [False, True, True, True],
                [True, False, True, True],
                [True, False, True, True],
            ]
        ),
        pixel_scales=(1.0, 2.0),
    )

    zoom_mask = mask.zoom_mask_unmasked

    assert (
        zoom_mask == np.array([[False, False], [False, False], [False, False]])
    ).all()
    assert zoom_mask.origin == (0.0, -2.0)


### GEOMETRY ###


def test__mask_centre():
    mask = np.array(
        [
            [True, True, True, True],
            [True, False, False, True],
            [True, True, True, True],
        ]
    )

    mask = aa.Mask2D.manual(mask=mask, pixel_scales=(1.0, 1.0))

    assert mask.mask_centre == (0.0, 0.0)

    mask = np.array(
        [
            [True, True, True, True],
            [True, False, False, False],
            [True, True, True, True],
        ]
    )

    mask = aa.Mask2D.manual(mask=mask, pixel_scales=(1.0, 1.0))

    assert mask.mask_centre == (0.0, 0.5)

    mask = np.array(
        [
            [True, True, False, True],
            [True, False, False, True],
            [True, True, True, True],
        ]
    )

    mask = aa.Mask2D.manual(mask=mask, pixel_scales=(1.0, 1.0))

    assert mask.mask_centre == (0.5, 0.0)

    mask = np.array(
        [
            [True, True, True, True],
            [False, False, False, True],
            [True, True, True, True],
        ]
    )

    mask = aa.Mask2D.manual(mask=mask, pixel_scales=(1.0, 1.0))

    assert mask.mask_centre == (0.0, -0.5)

    mask = np.array(
        [
            [True, True, True, True],
            [True, False, False, True],
            [True, False, True, True],
        ]
    )

    mask = aa.Mask2D.manual(mask=mask, pixel_scales=(1.0, 1.0))

    assert mask.mask_centre == (-0.5, 0.0)

    mask = np.array(
        [
            [True, True, True, True],
            [True, False, False, True],
            [False, True, True, True],
        ]
    )

    mask = aa.Mask2D.manual(mask=mask, pixel_scales=(1.0, 1.0))

    assert mask.mask_centre == (-0.5, -0.5)


def test__unmasked_grid_sub_1():

    grid_2d_util = aa.util.grid_2d.grid_2d_via_shape_native_from(
        shape_native=(4, 7), pixel_scales=(0.56, 0.56), sub_size=1
    )

    grid_1d_util = aa.util.grid_2d.grid_2d_slim_via_shape_native_from(
        shape_native=(4, 7), pixel_scales=(0.56, 0.56), sub_size=1
    )

    mask = aa.Mask2D.unmasked(shape_native=(4, 7), pixel_scales=(0.56, 0.56))
    mask[0, 0] = True

    assert mask.unmasked_grid_sub_1.slim == pytest.approx(grid_1d_util, 1e-4)
    assert mask.unmasked_grid_sub_1.native == pytest.approx(grid_2d_util, 1e-4)
    assert (
        mask.unmasked_grid_sub_1.mask == np.full(fill_value=False, shape=(4, 7))
    ).all()

    mask = aa.Mask2D.unmasked(shape_native=(3, 3), pixel_scales=(1.0, 1.0))

    assert (
        mask.unmasked_grid_sub_1.native
        == np.array(
            [
                [[1.0, -1.0], [1.0, 0.0], [1.0, 1.0]],
                [[0.0, -1.0], [0.0, 0.0], [0.0, 1.0]],
                [[-1.0, -1.0], [-1.0, 0.0], [-1.0, 1.0]],
            ]
        )
    ).all()

    grid_2d_util = aa.util.grid_2d.grid_2d_via_shape_native_from(
        shape_native=(4, 7), pixel_scales=(0.8, 0.56), sub_size=1
    )

    grid_1d_util = aa.util.grid_2d.grid_2d_slim_via_shape_native_from(
        shape_native=(4, 7), pixel_scales=(0.8, 0.56), sub_size=1
    )

    mask = aa.Mask2D.unmasked(shape_native=(4, 7), pixel_scales=(0.8, 0.56))

    assert mask.unmasked_grid_sub_1.slim == pytest.approx(grid_1d_util, 1e-4)
    assert mask.unmasked_grid_sub_1.native == pytest.approx(grid_2d_util, 1e-4)

    mask = aa.Mask2D.unmasked(shape_native=(3, 3), pixel_scales=(1.0, 2.0))

    assert (
        mask.unmasked_grid_sub_1.native
        == np.array(
            [
                [[1.0, -2.0], [1.0, 0.0], [1.0, 2.0]],
                [[0.0, -2.0], [0.0, 0.0], [0.0, 2.0]],
                [[-1.0, -2.0], [-1.0, 0.0], [-1.0, 2.0]],
            ]
        )
    ).all()


def test__masked_grid_sub_1():

    mask = aa.Mask2D.unmasked(shape_native=(3, 3), pixel_scales=(1.0, 1.0))

    assert (
        mask.masked_grid_sub_1.slim
        == np.array(
            [
                [1.0, -1.0],
                [1.0, 0.0],
                [1.0, 1.0],
                [0.0, -1.0],
                [0.0, 0.0],
                [0.0, 1.0],
                [-1.0, -1.0],
                [-1.0, 0.0],
                [-1.0, 1.0],
            ]
        )
    ).all()

    mask = aa.Mask2D.unmasked(shape_native=(3, 3), pixel_scales=(1.0, 1.0))
    mask[1, 1] = True

    assert (
        mask.masked_grid_sub_1.slim
        == np.array(
            [
                [1.0, -1.0],
                [1.0, 0.0],
                [1.0, 1.0],
                [0.0, -1.0],
                [0.0, 1.0],
                [-1.0, -1.0],
                [-1.0, 0.0],
                [-1.0, 1.0],
            ]
        )
    ).all()

    mask = aa.Mask2D.manual(
        mask=np.array([[False, True], [True, False], [True, False]]),
        pixel_scales=(1.0, 1.0),
        origin=(3.0, -2.0),
    )

    assert (
        mask.masked_grid_sub_1.slim == np.array([[4.0, -2.5], [3.0, -1.5], [2.0, -1.5]])
    ).all()


def test__edge_grid_sub_1():
    mask = np.array(
        [
            [True, True, True, True, True, True, True, True, True],
            [True, False, False, False, False, False, False, False, True],
            [True, False, True, True, True, True, True, False, True],
            [True, False, True, False, False, False, True, False, True],
            [True, False, True, False, True, False, True, False, True],
            [True, False, True, False, False, False, True, False, True],
            [True, False, True, True, True, True, True, False, True],
            [True, False, False, False, False, False, False, False, True],
            [True, True, True, True, True, True, True, True, True],
        ]
    )

    mask = aa.Mask2D.manual(mask=mask, pixel_scales=(1.0, 1.0))

    assert mask.edge_grid_sub_1.slim[0:11] == pytest.approx(
        np.array(
            [
                [3.0, -3.0],
                [3.0, -2.0],
                [3.0, -1.0],
                [3.0, -0.0],
                [3.0, 1.0],
                [3.0, 2.0],
                [3.0, 3.0],
                [2.0, -3.0],
                [2.0, 3.0],
                [1.0, -3.0],
                [1.0, -1.0],
            ]
        ),
        1e-4,
    )


def test__border_grid_sub_1():
    mask = np.array(
        [
            [True, True, True, True, True, True, True, True, True],
            [True, False, False, False, False, False, False, False, True],
            [True, False, True, True, True, True, True, False, True],
            [True, False, True, False, False, False, True, False, True],
            [True, False, True, False, True, False, True, False, True],
            [True, False, True, False, False, False, True, False, True],
            [True, False, True, True, True, True, True, False, True],
            [True, False, False, False, False, False, False, False, True],
            [True, True, True, True, True, True, True, True, True],
        ]
    )

    mask = aa.Mask2D.manual(mask=mask, pixel_scales=(1.0, 1.0))

    assert mask.border_grid_sub_1.slim[0:11] == pytest.approx(
        np.array(
            [
                [3.0, -3.0],
                [3.0, -2.0],
                [3.0, -1.0],
                [3.0, -0.0],
                [3.0, 1.0],
                [3.0, 2.0],
                [3.0, 3.0],
                [2.0, -3.0],
                [2.0, 3.0],
                [1.0, -3.0],
                [1.0, 3.0],
            ]
        ),
        1e-4,
    )


def test__masked_grid():
    mask = aa.Mask2D.unmasked(shape_native=(3, 3), pixel_scales=(1.0, 1.0), sub_size=1)

    assert (
        mask.masked_grid
        == np.array(
            [
                [1.0, -1.0],
                [1.0, 0.0],
                [1.0, 1.0],
                [0.0, -1.0],
                [0.0, 0.0],
                [0.0, 1.0],
                [-1.0, -1.0],
                [-1.0, 0.0],
                [-1.0, 1.0],
            ]
        )
    ).all()

    mask = aa.Mask2D.unmasked(shape_native=(2, 2), pixel_scales=(1.0, 1.0), sub_size=2)

    assert (
        mask.masked_grid
        == np.array(
            [
                [0.75, -0.75],
                [0.75, -0.25],
                [0.25, -0.75],
                [0.25, -0.25],
                [0.75, 0.25],
                [0.75, 0.75],
                [0.25, 0.25],
                [0.25, 0.75],
                [-0.25, -0.75],
                [-0.25, -0.25],
                [-0.75, -0.75],
                [-0.75, -0.25],
                [-0.25, 0.25],
                [-0.25, 0.75],
                [-0.75, 0.25],
                [-0.75, 0.75],
            ]
        )
    ).all()

    mask = aa.Mask2D.unmasked(shape_native=(3, 3), pixel_scales=(1.0, 1.0), sub_size=1)
    mask[1, 1] = True

    assert (
        mask.masked_grid
        == np.array(
            [
                [1.0, -1.0],
                [1.0, 0.0],
                [1.0, 1.0],
                [0.0, -1.0],
                [0.0, 1.0],
                [-1.0, -1.0],
                [-1.0, 0.0],
                [-1.0, 1.0],
            ]
        )
    ).all()

    mask = aa.Mask2D.manual(
        mask=np.array([[False, True], [True, False], [True, False]]),
        pixel_scales=(1.0, 1.0),
        sub_size=5,
        origin=(3.0, -2.0),
    )

    masked_grid_util = aa.util.grid_2d.grid_2d_slim_via_mask_from(
        mask_2d=mask, pixel_scales=(1.0, 1.0), sub_size=5, origin=(3.0, -2.0)
    )

    assert (mask.masked_grid == masked_grid_util).all()


def test__border_1d_grid():

    mask = np.array(
        [
            [True, True, True, True, True, True, True],
            [True, True, True, True, True, True, True],
            [True, False, False, True, True, True, True],
            [True, True, True, True, False, True, True],
            [True, True, True, True, True, True, True],
            [True, True, True, True, True, True, True],
            [True, True, True, True, True, True, True],
        ]
    )

    mask = aa.Mask2D.manual(mask=mask, pixel_scales=(1.0, 1.0), sub_size=2)

    assert (
        mask.border_grid_1d == np.array([[1.25, -2.25], [1.25, -1.25], [-0.25, 1.25]])
    ).all()

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

    mask = aa.Mask2D.manual(mask=mask, pixel_scales=(1.0, 1.0), sub_size=2)

    assert (
        mask.border_grid_1d
        == np.array(
            [
                [1.25, -1.25],
                [1.25, 0.25],
                [1.25, 1.25],
                [-0.25, -1.25],
                [-0.25, 1.25],
                [-1.25, -1.25],
                [-1.25, 0.25],
                [-1.25, 1.25],
            ]
        )
    ).all()
