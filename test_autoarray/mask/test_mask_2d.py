from astropy.io import fits
import os
from os import path
import numpy as np
import pytest
import shutil

import autoarray as aa
from autoarray import exc

test_data_path = path.join(
    "{}".format(path.dirname(path.realpath(__file__))), "files", "mask"
)


def test__constructor():
    mask = aa.Mask2D(mask=[[False, False], [True, True]], pixel_scales=1.0)

    assert type(mask) == aa.Mask2D
    assert (mask == np.array([[False, False], [True, True]])).all()
    assert mask.pixel_scales == (1.0, 1.0)
    assert mask.origin == (0.0, 0.0)
    assert (mask.geometry.extent == np.array([-1.0, 1.0, -1.0, 1.0])).all()

    mask = aa.Mask2D(
        mask=[[False, False], [True, True]],
        pixel_scales=(2.0, 3.0),
        origin=(0.0, 1.0),
    )

    assert type(mask) == aa.Mask2D
    assert (mask == np.array([[False, False], [True, True]])).all()
    assert mask.pixel_scales == (2.0, 3.0)
    assert mask.origin == (0.0, 1.0)


def test__constructor__invert_is_true():
    mask = aa.Mask2D(
        mask=[[False, False, True], [True, True, False]],
        pixel_scales=1.0,
        invert=True,
    )

    assert type(mask) == aa.Mask2D
    assert (mask == np.array([[True, True, False], [False, False, True]])).all()


def test__all_false():
    mask = aa.Mask2D.all_false(shape_native=(5, 5), pixel_scales=1.0, invert=False)

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

    mask = aa.Mask2D.all_false(
        shape_native=(3, 3),
        pixel_scales=(2.0, 2.5),
        invert=True,
        origin=(1.0, 2.0),
    )

    assert mask.shape == (3, 3)
    assert (
        mask == np.array([[True, True, True], [True, True, True], [True, True, True]])
    ).all()

    assert mask.pixel_scales == (2.0, 2.5)
    assert mask.origin == (1.0, 2.0)


def test__circular():
    mask_via_util = aa.util.mask_2d.mask_2d_circular_from(
        shape_native=(5, 4), pixel_scales=(2.7, 2.7), radius=3.5, centre=(0.0, 0.0)
    )

    mask = aa.Mask2D.circular(
        shape_native=(5, 4),
        pixel_scales=(2.7, 2.7),
        radius=3.5,
        centre=(0.0, 0.0),
    )

    assert (mask == mask_via_util).all()
    assert mask.origin == (0.0, 0.0)
    assert mask.mask_centre == pytest.approx((0.0, 0.0), 1.0e-8)

    mask = aa.Mask2D.circular(
        shape_native=(5, 4),
        pixel_scales=(2.7, 2.7),
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
        inner_radius=0.8,
        outer_radius=3.5,
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
    test_data_path = path.join(
        "{}".format(path.dirname(path.realpath(__file__))), "files", "mask"
    )

    mask = aa.Mask2D.from_fits(
        file_path=path.join(test_data_path, "3x3_ones.fits"),
        hdu=0,
        pixel_scales=(1.0, 1.0),
    )

    test_data_path = path.join(
        "{}".format(path.dirname(path.realpath(__file__))),
        "files",
        "array",
        "output_test",
    )

    if path.exists(test_data_path):
        shutil.rmtree(test_data_path)

    os.makedirs(test_data_path)

    mask.output_to_fits(file_path=path.join(test_data_path, "mask.fits"))

    mask = aa.Mask2D.from_fits(
        file_path=path.join(test_data_path, "mask.fits"),
        hdu=0,
        pixel_scales=(1.0, 1.0),
        origin=(2.0, 2.0),
    )

    assert (mask == np.ones((3, 3))).all()
    assert mask.pixel_scales == (1.0, 1.0)
    assert mask.origin == (2.0, 2.0)

    header = aa.header_obj_from(file_path=path.join(test_data_path, "mask.fits"), hdu=0)

    assert header["PIXSCAY"] == 1.0
    assert header["PIXSCAX"] == 1.0
    assert header["ORIGINY"] == 0.0
    assert header["ORIGINX"] == 0.0


def test__from_fits__with_resized_mask_shape():
    mask = aa.Mask2D.from_fits(
        file_path=path.join(test_data_path, "3x3_ones.fits"),
        hdu=0,
        pixel_scales=(1.0, 1.0),
        resized_mask_shape=(1, 1),
    )

    assert mask.shape_native == (1, 1)

    mask = aa.Mask2D.from_fits(
        file_path=path.join(test_data_path, "3x3_ones.fits"),
        hdu=0,
        pixel_scales=(1.0, 1.0),
        resized_mask_shape=(5, 5),
    )

    assert mask.shape_native == (5, 5)


def test__mask__input_is_1d_mask__no_shape_native__raises_exception():
    with pytest.raises(exc.MaskException):
        aa.Mask2D(mask=[False, False, True], pixel_scales=1.0)

    with pytest.raises(exc.MaskException):
        aa.Mask2D(mask=[False, False, True], pixel_scales=False)

    with pytest.raises(exc.MaskException):
        aa.Mask2D(mask=[False, False, True], pixel_scales=1.0)

    with pytest.raises(exc.MaskException):
        aa.Mask2D(mask=[False, False, True], pixel_scales=False)


def test__is_all_true():
    mask = aa.Mask2D(mask=[[False, False], [False, False]], pixel_scales=1.0)

    assert mask.is_all_true is False

    mask = aa.Mask2D(mask=[[False, False]], pixel_scales=1.0)

    assert mask.is_all_true is False

    mask = aa.Mask2D(mask=[[False, True], [False, False]], pixel_scales=1.0)

    assert mask.is_all_true is False

    mask = aa.Mask2D(mask=[[True, True], [True, True]], pixel_scales=1.0)

    assert mask.is_all_true is True


def test__is_all_false():
    mask = aa.Mask2D(mask=[[False, False], [False, False]], pixel_scales=1.0)

    assert mask.is_all_false is True

    mask = aa.Mask2D(mask=[[False, False]], pixel_scales=1.0)

    assert mask.is_all_false is True

    mask = aa.Mask2D(mask=[[False, True], [False, False]], pixel_scales=1.0)

    assert mask.is_all_false is False

    mask = aa.Mask2D(mask=[[True, True], [False, False]], pixel_scales=1.0)

    assert mask.is_all_false is False


def test__shape_native_masked_pixels():
    mask = aa.Mask2D(
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

    mask = aa.Mask2D(
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

    mask = aa.Mask2D(
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


def test__rescaled_from():
    mask = aa.Mask2D.all_false(shape_native=(5, 5), pixel_scales=1.0)
    mask[2, 2] = True

    mask_rescaled = mask.rescaled_from(rescale_factor=2.0)

    mask_rescaled_manual = np.full(fill_value=False, shape=(3, 3))
    mask_rescaled_manual[1, 1] = True

    mask_rescaled_manual = aa.util.mask_2d.rescaled_mask_2d_from(
        mask_2d=mask, rescale_factor=2.0
    )

    assert (mask_rescaled == mask_rescaled_manual).all()


def test__resized_from():
    mask = aa.Mask2D.all_false(shape_native=(5, 5), pixel_scales=1.0)
    mask[2, 2] = True

    mask_resized = mask.resized_from(new_shape=(7, 7))

    mask_resized_manual = np.full(fill_value=False, shape=(7, 7))
    mask_resized_manual[3, 3] = True

    assert (mask_resized == mask_resized_manual).all()

    mask = aa.Mask2D.all_false(shape_native=(5, 5), pixel_scales=1.0)
    mask[2, 2] = True

    mask_resized = mask.resized_from(new_shape=(3, 3))

    mask_resized_manual = np.full(fill_value=False, shape=(3, 3))
    mask_resized_manual[1, 1] = True

    assert (mask_resized == mask_resized_manual).all()


def test__zoom_quantities():
    mask = aa.Mask2D.all_false(shape_native=(3, 5), pixel_scales=(1.0, 1.0))
    assert mask.zoom_centre == (1.0, 2.0)
    assert mask.zoom_offset_pixels == (0, 0)
    assert mask.zoom_shape_native == (5, 5)

    mask = aa.Mask2D.all_false(shape_native=(5, 3), pixel_scales=(1.0, 1.0))
    assert mask.zoom_centre == (2.0, 1.0)
    assert mask.zoom_offset_pixels == (0, 0)
    assert mask.zoom_shape_native == (5, 5)

    mask = aa.Mask2D.all_false(shape_native=(4, 6), pixel_scales=(1.0, 1.0))
    assert mask.zoom_centre == (1.5, 2.5)
    assert mask.zoom_offset_pixels == (0, 0)
    assert mask.zoom_shape_native == (6, 6)

    mask = aa.Mask2D.all_false(shape_native=(6, 4), pixel_scales=(1.0, 1.0))
    assert mask.zoom_centre == (2.5, 1.5)
    assert mask.zoom_offset_pixels == (0, 0)
    assert mask.zoom_shape_native == (6, 6)


def test__mask_is_single_false__extraction_centre_is_central_pixel():
    mask = aa.Mask2D(
        mask=np.array([[False, True, True], [True, True, True], [True, True, True]]),
        pixel_scales=(1.0, 1.0),
    )
    assert mask.zoom_centre == (0, 0)
    assert mask.zoom_offset_pixels == (-1, -1)
    assert mask.zoom_shape_native == (1, 1)

    mask = aa.Mask2D(
        mask=np.array([[True, True, False], [True, True, True], [True, True, True]]),
        pixel_scales=(1.0, 1.0),
    )
    assert mask.zoom_centre == (0, 2)
    assert mask.zoom_offset_pixels == (-1, 1)
    assert mask.zoom_shape_native == (1, 1)

    mask = aa.Mask2D(
        mask=np.array([[True, True, True], [True, True, True], [False, True, True]]),
        pixel_scales=(1.0, 1.0),
    )
    assert mask.zoom_centre == (2, 0)
    assert mask.zoom_offset_pixels == (1, -1)
    assert mask.zoom_shape_native == (1, 1)

    mask = aa.Mask2D(
        mask=np.array([[True, True, True], [True, True, True], [True, True, False]]),
        pixel_scales=(1.0, 1.0),
    )
    assert mask.zoom_centre == (2, 2)
    assert mask.zoom_offset_pixels == (1, 1)
    assert mask.zoom_shape_native == (1, 1)

    mask = aa.Mask2D(
        mask=np.array([[True, False, True], [True, True, True], [True, True, True]]),
        pixel_scales=(1.0, 1.0),
    )
    assert mask.zoom_centre == (0, 1)
    assert mask.zoom_offset_pixels == (-1, 0)
    assert mask.zoom_shape_native == (1, 1)

    mask = aa.Mask2D(
        mask=np.array([[True, True, True], [False, True, True], [True, True, True]]),
        pixel_scales=(1.0, 1.0),
    )
    assert mask.zoom_centre == (1, 0)
    assert mask.zoom_offset_pixels == (0, -1)
    assert mask.zoom_shape_native == (1, 1)

    mask = aa.Mask2D(
        mask=np.array([[True, True, True], [True, True, False], [True, True, True]]),
        pixel_scales=(1.0, 1.0),
    )
    assert mask.zoom_centre == (1, 2)
    assert mask.zoom_offset_pixels == (0, 1)
    assert mask.zoom_shape_native == (1, 1)

    mask = aa.Mask2D(
        mask=np.array([[True, True, True], [True, True, True], [True, False, True]]),
        pixel_scales=(1.0, 1.0),
    )
    assert mask.zoom_centre == (2, 1)
    assert mask.zoom_offset_pixels == (1, 0)
    assert mask.zoom_shape_native == (1, 1)


def test__mask_is_x2_false__extraction_centre_is_central_pixel():
    mask = aa.Mask2D(
        mask=np.array([[False, True, True], [True, True, True], [True, True, False]]),
        pixel_scales=(1.0, 1.0),
    )
    assert mask.zoom_centre == (1, 1)
    assert mask.zoom_offset_pixels == (0, 0)
    assert mask.zoom_shape_native == (3, 3)

    mask = aa.Mask2D(
        mask=np.array([[False, True, True], [True, True, True], [False, True, True]]),
        pixel_scales=(1.0, 1.0),
    )
    assert mask.zoom_centre == (1, 0)
    assert mask.zoom_offset_pixels == (0, -1)
    assert mask.zoom_shape_native == (3, 3)

    mask = aa.Mask2D(
        mask=np.array([[False, True, False], [True, True, True], [True, True, True]]),
        pixel_scales=(1.0, 1.0),
    )
    assert mask.zoom_centre == (0, 1)
    assert mask.zoom_offset_pixels == (-1, 0)
    assert mask.zoom_shape_native == (3, 3)

    mask = aa.Mask2D(
        mask=np.array([[False, False, True], [True, True, True], [True, True, True]]),
        pixel_scales=(1.0, 1.0),
    )
    assert mask.zoom_centre == (0, 0.5)
    assert mask.zoom_offset_pixels == (-1, -0.5)
    assert mask.zoom_shape_native == (1, 2)


def test__rectangular_mask():
    mask = aa.Mask2D(
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

    mask = aa.Mask2D(
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

    mask = aa.Mask2D(
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

    mask = aa.Mask2D(
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

    mask = aa.Mask2D(
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

    mask = aa.Mask2D(
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
    mask = aa.Mask2D(
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

    mask = aa.Mask2D(
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


def test__mask_centre():
    mask = np.array(
        [
            [True, True, True, True],
            [True, False, False, True],
            [True, True, True, True],
        ]
    )

    mask = aa.Mask2D(mask=mask, pixel_scales=(1.0, 1.0))

    assert mask.mask_centre == (0.0, 0.0)

    mask = np.array(
        [
            [True, True, True, True],
            [True, False, False, False],
            [True, True, True, True],
        ]
    )

    mask = aa.Mask2D(mask=mask, pixel_scales=(1.0, 1.0))

    assert mask.mask_centre == (0.0, 0.5)

    mask = np.array(
        [
            [True, True, False, True],
            [True, False, False, True],
            [True, True, True, True],
        ]
    )

    mask = aa.Mask2D(mask=mask, pixel_scales=(1.0, 1.0))

    assert mask.mask_centre == (0.5, 0.0)

    mask = np.array(
        [
            [True, True, True, True],
            [False, False, False, True],
            [True, True, True, True],
        ]
    )

    mask = aa.Mask2D(mask=mask, pixel_scales=(1.0, 1.0))

    assert mask.mask_centre == (0.0, -0.5)

    mask = np.array(
        [
            [True, True, True, True],
            [True, False, False, True],
            [True, False, True, True],
        ]
    )

    mask = aa.Mask2D(mask=mask, pixel_scales=(1.0, 1.0))

    assert mask.mask_centre == (-0.5, 0.0)

    mask = np.array(
        [
            [True, True, True, True],
            [True, False, False, True],
            [False, True, True, True],
        ]
    )

    mask = aa.Mask2D(mask=mask, pixel_scales=(1.0, 1.0))

    assert mask.mask_centre == (-0.5, -0.5)


def test__is_circular():
    mask = np.array(
        [
            [True, True, True, True],
            [True, False, False, True],
            [True, True, True, True],
        ]
    )

    mask = aa.Mask2D(mask=mask, pixel_scales=(1.0, 1.0))

    assert mask.is_circular == False

    mask = aa.Mask2D.circular(shape_native=(5, 5), radius=1.0, pixel_scales=(1.0, 1.0))

    assert mask.is_circular == True

    mask = aa.Mask2D.circular(
        shape_native=(10, 10), radius=3.0, pixel_scales=(1.0, 1.0)
    )

    assert mask.is_circular == True

    mask = aa.Mask2D.circular(
        shape_native=(10, 10), radius=4.0, pixel_scales=(1.0, 1.0)
    )

    assert mask.is_circular == True


def test__circular_radius():
    mask = aa.Mask2D.circular(
        shape_native=(10, 10), radius=3.0, pixel_scales=(1.0, 1.0)
    )

    assert mask.circular_radius == pytest.approx(3.0, 1e-4)

    mask = aa.Mask2D.circular(
        shape_native=(30, 30), radius=5.5, pixel_scales=(0.5, 0.5)
    )

    assert mask.circular_radius == pytest.approx(5.5, 1e-4)

    mask = aa.Mask2D.circular(
        shape_native=(30, 30), radius=5.75, pixel_scales=(0.5, 0.5)
    )

    assert mask.circular_radius == pytest.approx(5.5, 1e-4)
