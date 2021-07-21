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


class TestMask:
    def test__manual(self):

        mask = aa.Mask2D.manual(
            mask=[[False, False], [True, True]], pixel_scales=1.0, sub_size=1
        )

        assert type(mask) == aa.Mask2D
        assert (mask == np.array([[False, False], [True, True]])).all()
        assert mask.pixel_scales == (1.0, 1.0)
        assert mask.origin == (0.0, 0.0)
        assert mask.sub_size == 1
        assert (mask.extent == np.array([-1.0, 1.0, -1.0, 1.0])).all()

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

        mask = aa.Mask2D.manual(
            mask=[[False, False], [True, True], [True, False], [False, True]],
            pixel_scales=1.0,
            sub_size=2,
        )

        assert type(mask) == aa.Mask2D
        assert (
            mask
            == np.array([[False, False], [True, True], [True, False], [False, True]])
        ).all()
        assert mask.pixel_scales == (1.0, 1.0)
        assert mask.origin == (0.0, 0.0)
        assert mask.sub_size == 2

    def test__mask__invert_is_true_inverts_the_mask(self):

        mask = aa.Mask2D.manual(
            mask=[[False, False, True], [True, True, False]],
            pixel_scales=1.0,
            invert=True,
        )

        assert type(mask) == aa.Mask2D
        assert (mask == np.array([[True, True, False], [False, False, True]])).all()

    def test__mask__input_is_1d_mask__no_shape_native__raises_exception(self):

        with pytest.raises(exc.MaskException):

            aa.Mask2D.manual(mask=[False, False, True], pixel_scales=1.0)

        with pytest.raises(exc.MaskException):

            aa.Mask2D.manual(mask=[False, False, True], pixel_scales=False)

        with pytest.raises(exc.MaskException):

            aa.Mask2D.manual(mask=[False, False, True], pixel_scales=1.0, sub_size=1)

        with pytest.raises(exc.MaskException):

            aa.Mask2D.manual(mask=[False, False, True], pixel_scales=False, sub_size=1)

    def test__is_all_true(self):

        mask = aa.Mask2D.manual(mask=[[False, False], [False, False]], pixel_scales=1.0)

        assert mask.is_all_true is False

        mask = aa.Mask2D.manual(mask=[[False, False]], pixel_scales=1.0)

        assert mask.is_all_true is False

        mask = aa.Mask2D.manual(mask=[[False, True], [False, False]], pixel_scales=1.0)

        assert mask.is_all_true is False

        mask = aa.Mask2D.manual(mask=[[True, True], [True, True]], pixel_scales=1.0)

        assert mask.is_all_true is True

    def test__is_all_false(self):

        mask = aa.Mask2D.manual(mask=[[False, False], [False, False]], pixel_scales=1.0)

        assert mask.is_all_false is True

        mask = aa.Mask2D.manual(mask=[[False, False]], pixel_scales=1.0)

        assert mask.is_all_false is True

        mask = aa.Mask2D.manual(mask=[[False, True], [False, False]], pixel_scales=1.0)

        assert mask.is_all_false is False

        mask = aa.Mask2D.manual(mask=[[True, True], [False, False]], pixel_scales=1.0)

        assert mask.is_all_false is False


class TestClassMethods:
    def test__mask_all_unmasked__5x5__input__all_are_false(self):

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
            shape_native=(3, 3), pixel_scales=(1.5, 1.5), invert=False, sub_size=2
        )

        assert mask.shape == (3, 3)
        assert (
            mask
            == np.array(
                [[False, False, False], [False, False, False], [False, False, False]]
            )
        ).all()

        assert mask.sub_size == 2
        assert mask.pixel_scales == (1.5, 1.5)
        assert mask.origin == (0.0, 0.0)
        assert mask.mask_centre == (0.0, 0.0)

        mask = aa.Mask2D.unmasked(
            shape_native=(3, 3),
            pixel_scales=(2.0, 2.5),
            invert=True,
            sub_size=4,
            origin=(1.0, 2.0),
        )

        assert mask.shape == (3, 3)
        assert (
            mask
            == np.array([[True, True, True], [True, True, True], [True, True, True]])
        ).all()

        assert mask.sub_size == 4
        assert mask.pixel_scales == (2.0, 2.5)
        assert mask.origin == (1.0, 2.0)

    def test__mask_circular__compare_to_array_util(self):

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

    def test__mask_circular__inverted__compare_to_array_util(self):
        mask_via_util = aa.util.mask_2d.mask_2d_circular_from(
            shape_native=(5, 4), pixel_scales=(2.7, 2.7), radius=3.5, centre=(0.0, 0.0)
        )

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

    def test__mask_annulus__compare_to_array_util(self):
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

    def test__mask_annulus_inverted__compare_to_array_util(self):
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
            invert=True,
        )

        assert (mask == np.invert(mask_via_util)).all()
        assert mask.origin == (0.0, 0.0)
        assert mask.mask_centre == (0.0, 0.0)

    def test__mask_anti_annulus__compare_to_array_util(self):
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

    def test__mask_anti_annulus_inverted__compare_to_array_util(self):
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
            invert=True,
        )

        assert (mask == np.invert(mask_via_util)).all()
        assert mask.origin == (0.0, 0.0)
        assert mask.mask_centre == (0.0, 0.0)

    def test__mask_elliptical__compare_to_array_util(self):
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

    def test__mask_elliptical_inverted__compare_to_array_util(self):
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
            invert=True,
        )

        assert (mask == np.invert(mask_via_util)).all()
        assert mask.origin == (0.0, 0.0)
        assert mask.mask_centre == (0.0, 0.0)

    def test__mask_elliptical_annular__compare_to_array_util(self):
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

    def test__mask_elliptical_annular_inverted__compare_to_array_util(self):

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
            invert=True,
        )

        assert (mask == np.invert(mask_via_util)).all()
        assert mask.origin == (0.0, 0.0)
        assert mask.mask_centre == (0.0, 0.0)

    def test__from_pixel_coordinates__mask_with_or_without_buffer__false_at_buffed_coordinates(
        self,
    ):

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


class TestToFromFits:
    def test__load_and_output_mask_to_fits(self):

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

    def test__load_from_fits_with_resized_mask_shape(self):

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


class TestSubQuantities:
    def test__sub_shape_is_shape_times_sub_size(self):

        mask = aa.Mask2D.unmasked(shape_native=(5, 5), pixel_scales=1.0, sub_size=1)

        assert mask.sub_shape_native == (5, 5)

        mask = aa.Mask2D.unmasked(shape_native=(5, 5), pixel_scales=1.0, sub_size=2)

        assert mask.sub_shape_native == (10, 10)

        mask = aa.Mask2D.unmasked(shape_native=(10, 5), pixel_scales=1.0, sub_size=3)

        assert mask.sub_shape_native == (30, 15)


class TestNewMasksFromMask:
    def test__sub_mask__is_mask_at_sub_grid_resolution(self):

        mask = aa.Mask2D.manual(
            mask=[[False, True], [False, False]], pixel_scales=1.0, sub_size=2
        )

        assert (
            mask.sub_mask
            == np.array(
                [
                    [False, False, True, True],
                    [False, False, True, True],
                    [False, False, False, False],
                    [False, False, False, False],
                ]
            )
        ).all()

        mask = aa.Mask2D.manual(
            mask=[[False, False, True], [False, True, False]],
            pixel_scales=1.0,
            sub_size=2,
        )

        assert (
            mask.sub_mask
            == np.array(
                [
                    [False, False, False, False, True, True],
                    [False, False, False, False, True, True],
                    [False, False, True, True, False, False],
                    [False, False, True, True, False, False],
                ]
            )
        ).all()

    def test__resized_mask__pad__compare_to_manual_mask(self):

        mask = aa.Mask2D.unmasked(shape_native=(5, 5), pixel_scales=1.0)
        mask[2, 2] = True

        mask_resized = mask.resized_mask_from_new_shape(new_shape=(7, 7))

        mask_resized_manual = np.full(fill_value=False, shape=(7, 7))
        mask_resized_manual[3, 3] = True

        assert (mask_resized == mask_resized_manual).all()

    def test__resized_mask__trim__compare_to_manual_mask(self):

        mask = aa.Mask2D.unmasked(shape_native=(5, 5), pixel_scales=1.0)
        mask[2, 2] = True

        mask_resized = mask.resized_mask_from_new_shape(new_shape=(3, 3))

        mask_resized_manual = np.full(fill_value=False, shape=(3, 3))
        mask_resized_manual[1, 1] = True

        assert (mask_resized == mask_resized_manual).all()

    def test__rescaled_mask_from_rescale_factor__compare_to_manual_mask(self):

        mask = aa.Mask2D.unmasked(shape_native=(5, 5), pixel_scales=1.0)
        mask[2, 2] = True

        mask_rescaled = mask.rescaled_mask_from_rescale_factor(rescale_factor=2.0)

        mask_rescaled_manual = np.full(fill_value=False, shape=(3, 3))
        mask_rescaled_manual[1, 1] = True

        mask_rescaled_manual = aa.util.mask_2d.rescaled_mask_2d_from(
            mask_2d=mask, rescale_factor=2.0
        )

        assert (mask_rescaled == mask_rescaled_manual).all()

    def test__edged_buffed_mask__compare_to_manual_mask(self):

        mask = aa.Mask2D.unmasked(shape_native=(5, 5), pixel_scales=1.0)
        mask[2, 2] = True

        edge_buffed_mask_manual = aa.util.mask_2d.buffed_mask_2d_from(
            mask_2d=mask
        ).astype("bool")

        assert (mask.edge_buffed_mask == edge_buffed_mask_manual).all()


class TestRegions:
    def test__sub_native_index_for_sub_slim_index__compare_to_array_util(self):

        mask = aa.Mask2D.manual(
            mask=[[True, True, True], [True, False, False], [True, True, False]],
            pixel_scales=1.0,
        )

        sub_native_index_for_sub_slim_index_2d = aa.util.mask_2d.native_index_for_slim_index_2d_from(
            mask_2d=mask, sub_size=1
        )

        assert mask._sub_native_index_for_sub_slim_index == pytest.approx(
            sub_native_index_for_sub_slim_index_2d, 1e-4
        )

    def test__unmasked_mask(self):

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

        assert (mask.unmasked_mask == np.full(fill_value=False, shape=(9, 9))).all()

    def test__blurring_mask_for_psf_shape__compare_to_array_util(self):

        mask = aa.Mask2D.manual(
            mask=[
                [True, True, True, True, True, True, True, True],
                [True, False, True, True, True, False, True, True],
                [True, True, True, True, True, True, True, True],
                [True, True, True, True, True, True, True, True],
                [True, True, True, True, True, True, True, True],
                [True, False, True, True, True, False, True, True],
                [True, True, True, True, True, True, True, True],
                [True, True, True, True, True, True, True, True],
                [True, True, True, True, True, True, True, True],
            ],
            pixel_scales=1.0,
        )

        blurring_mask_via_util = aa.util.mask_2d.blurring_mask_2d_from(
            mask_2d=mask, kernel_shape_native=(3, 3)
        )

        blurring_mask = mask.blurring_mask_from_kernel_shape(kernel_shape_native=(3, 3))

        assert (blurring_mask == blurring_mask_via_util).all()

    def test__edge_image_pixels__compare_to_array_util(self):
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

        edge_pixels_util = aa.util.mask_2d.edge_1d_indexes_from(mask_2d=mask)

        assert mask._edge_1d_indexes == pytest.approx(edge_pixels_util, 1e-4)
        assert mask._edge_2d_indexes[0] == pytest.approx(np.array([1, 1]), 1e-4)
        assert mask._edge_2d_indexes[10] == pytest.approx(np.array([3, 3]), 1e-4)
        assert mask._edge_1d_indexes.shape[0] == mask._edge_2d_indexes.shape[0]

    def test__edge_mask(self):
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

        assert (
            mask.edge_mask
            == np.array(
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
        ).all()

    def test__border_image_pixels__compare_to_array_util(self):
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

        border_pixels_util = aa.util.mask_2d.border_slim_indexes_from(mask_2d=mask)

        assert mask._border_1d_indexes == pytest.approx(border_pixels_util, 1e-4)
        assert mask._border_2d_indexes[0] == pytest.approx(np.array([1, 1]), 1e-4)
        assert mask._border_2d_indexes[10] == pytest.approx(np.array([3, 7]), 1e-4)
        assert mask._border_1d_indexes.shape[0] == mask._border_2d_indexes.shape[0]

    def test__border_mask(self):
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

        assert (
            mask.border_mask
            == np.array(
                [
                    [True, True, True, True, True, True, True, True, True],
                    [True, False, False, False, False, False, False, False, True],
                    [True, False, True, True, True, True, True, False, True],
                    [True, False, True, True, True, True, True, False, True],
                    [True, False, True, True, True, True, True, False, True],
                    [True, False, True, True, True, True, True, False, True],
                    [True, False, True, True, True, True, True, False, True],
                    [True, False, False, False, False, False, False, False, True],
                    [True, True, True, True, True, True, True, True, True],
                ]
            )
        ).all()

    def test__sub_border_flat_indexes__compare_to_array_util_and_numerics(self):

        mask = aa.Mask2D.manual(
            mask=[
                [False, False, False, False, False, False, False, True],
                [False, True, True, True, True, True, False, True],
                [False, True, False, False, False, True, False, True],
                [False, True, False, True, False, True, False, True],
                [False, True, False, False, False, True, False, True],
                [False, True, True, True, True, True, False, True],
                [False, False, False, False, False, False, False, True],
            ],
            pixel_scales=1.0,
            sub_size=2,
        )

        sub_border_pixels_util = aa.util.mask_2d.sub_border_pixel_slim_indexes_from(
            mask_2d=mask, sub_size=2
        )

        assert mask._sub_border_flat_indexes == pytest.approx(
            sub_border_pixels_util, 1e-4
        )

        mask = aa.Mask2D.manual(
            mask=[
                [True, True, True, True, True, True, True],
                [True, True, True, True, True, True, True],
                [True, True, False, False, False, True, True],
                [True, True, False, False, False, True, True],
                [True, True, False, False, False, True, True],
                [True, True, True, True, True, True, True],
                [True, True, True, True, True, True, True],
            ],
            pixel_scales=1.0,
            sub_size=2,
        )

        assert (
            mask._sub_border_flat_indexes == np.array([0, 5, 9, 14, 23, 26, 31, 35])
        ).all()

    def test__slim_index_for_sub_slim_index__compare_to_util(self):
        mask = aa.Mask2D.manual(
            mask=[[True, False, True], [False, False, False], [True, False, False]],
            pixel_scales=1.0,
            sub_size=2,
        )

        slim_index_for_sub_slim_index_util = aa.util.mask_2d.slim_index_for_sub_slim_index_via_mask_2d_from(
            mask_2d=mask, sub_size=2
        )

        assert (
            mask._slim_index_for_sub_slim_index == slim_index_for_sub_slim_index_util
        ).all()

    def test__sub_mask_index_for_sub_mask_1d_index__compare_to_array_util(self):
        mask = aa.Mask2D.manual(
            mask=[[True, True, True], [True, False, False], [True, True, False]],
            pixel_scales=1.0,
            sub_size=2,
        )

        sub_mask_index_for_sub_mask_1d_index = aa.util.mask_2d.native_index_for_slim_index_2d_from(
            mask_2d=mask, sub_size=2
        )

        assert mask._sub_mask_index_for_sub_mask_1d_index == pytest.approx(
            sub_mask_index_for_sub_mask_1d_index, 1e-4
        )

    def test__shape_masked_pixels(self):

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


class TestZoom:
    def test__odd_sized_false_mask__centre_is_0_0__pixels_from_centre_are_0_0(self):

        mask = aa.Mask2D.unmasked(shape_native=(3, 3), pixel_scales=(1.0, 1.0))

        assert mask.zoom_centre == (1.0, 1.0)
        assert mask.zoom_offset_pixels == (0, 0)
        assert mask.zoom_shape_native == (3, 3)

        mask = aa.Mask2D.unmasked(shape_native=(5, 5), pixel_scales=(1.0, 1.0))

        assert mask.zoom_centre == (2.0, 2.0)
        assert mask.zoom_offset_pixels == (0, 0)
        assert mask.zoom_shape_native == (5, 5)

        mask = aa.Mask2D.unmasked(shape_native=(3, 5), pixel_scales=(1.0, 1.0))
        assert mask.zoom_centre == (1.0, 2.0)
        assert mask.zoom_offset_pixels == (0, 0)
        assert mask.zoom_shape_native == (5, 5)

        mask = aa.Mask2D.unmasked(shape_native=(5, 3), pixel_scales=(1.0, 1.0))
        assert mask.zoom_centre == (2.0, 1.0)
        assert mask.zoom_offset_pixels == (0, 0)
        assert mask.zoom_shape_native == (5, 5)

    def test__even_sized_false_mask__centre_is_0_0__pixels_from_centre_are_0_0(self):

        mask = aa.Mask2D.unmasked(shape_native=(4, 4), pixel_scales=(1.0, 1.0))
        assert mask.zoom_centre == (1.5, 1.5)
        assert mask.zoom_offset_pixels == (0, 0)
        assert mask.zoom_shape_native == (4, 4)

        mask = aa.Mask2D.unmasked(shape_native=(6, 6), pixel_scales=(1.0, 1.0))
        assert mask.zoom_centre == (2.5, 2.5)
        assert mask.zoom_offset_pixels == (0, 0)
        assert mask.zoom_shape_native == (6, 6)

        mask = aa.Mask2D.unmasked(shape_native=(4, 6), pixel_scales=(1.0, 1.0))
        assert mask.zoom_centre == (1.5, 2.5)
        assert mask.zoom_offset_pixels == (0, 0)
        assert mask.zoom_shape_native == (6, 6)

        mask = aa.Mask2D.unmasked(shape_native=(6, 4), pixel_scales=(1.0, 1.0))
        assert mask.zoom_centre == (2.5, 1.5)
        assert mask.zoom_offset_pixels == (0, 0)
        assert mask.zoom_shape_native == (6, 6)

    def test__mask_is_single_false__extraction_centre_is_central_pixel(self):

        mask = aa.Mask2D.manual(
            mask=np.array(
                [[False, True, True], [True, True, True], [True, True, True]]
            ),
            pixel_scales=(1.0, 1.0),
        )
        assert mask.zoom_centre == (0, 0)
        assert mask.zoom_offset_pixels == (-1, -1)
        assert mask.zoom_shape_native == (1, 1)

        mask = aa.Mask2D.manual(
            mask=np.array(
                [[True, True, False], [True, True, True], [True, True, True]]
            ),
            pixel_scales=(1.0, 1.0),
        )
        assert mask.zoom_centre == (0, 2)
        assert mask.zoom_offset_pixels == (-1, 1)
        assert mask.zoom_shape_native == (1, 1)

        mask = aa.Mask2D.manual(
            mask=np.array(
                [[True, True, True], [True, True, True], [False, True, True]]
            ),
            pixel_scales=(1.0, 1.0),
        )
        assert mask.zoom_centre == (2, 0)
        assert mask.zoom_offset_pixels == (1, -1)
        assert mask.zoom_shape_native == (1, 1)

        mask = aa.Mask2D.manual(
            mask=np.array(
                [[True, True, True], [True, True, True], [True, True, False]]
            ),
            pixel_scales=(1.0, 1.0),
        )
        assert mask.zoom_centre == (2, 2)
        assert mask.zoom_offset_pixels == (1, 1)
        assert mask.zoom_shape_native == (1, 1)

        mask = aa.Mask2D.manual(
            mask=np.array(
                [[True, False, True], [True, True, True], [True, True, True]]
            ),
            pixel_scales=(1.0, 1.0),
        )
        assert mask.zoom_centre == (0, 1)
        assert mask.zoom_offset_pixels == (-1, 0)
        assert mask.zoom_shape_native == (1, 1)

        mask = aa.Mask2D.manual(
            mask=np.array(
                [[True, True, True], [False, True, True], [True, True, True]]
            ),
            pixel_scales=(1.0, 1.0),
        )
        assert mask.zoom_centre == (1, 0)
        assert mask.zoom_offset_pixels == (0, -1)
        assert mask.zoom_shape_native == (1, 1)

        mask = aa.Mask2D.manual(
            mask=np.array(
                [[True, True, True], [True, True, False], [True, True, True]]
            ),
            pixel_scales=(1.0, 1.0),
        )
        assert mask.zoom_centre == (1, 2)
        assert mask.zoom_offset_pixels == (0, 1)
        assert mask.zoom_shape_native == (1, 1)

        mask = aa.Mask2D.manual(
            mask=np.array(
                [[True, True, True], [True, True, True], [True, False, True]]
            ),
            pixel_scales=(1.0, 1.0),
        )
        assert mask.zoom_centre == (2, 1)
        assert mask.zoom_offset_pixels == (1, 0)
        assert mask.zoom_shape_native == (1, 1)

    def test__mask_is_x2_false__extraction_centre_is_central_pixel(self):
        mask = aa.Mask2D.manual(
            mask=np.array(
                [[False, True, True], [True, True, True], [True, True, False]]
            ),
            pixel_scales=(1.0, 1.0),
        )
        assert mask.zoom_centre == (1, 1)
        assert mask.zoom_offset_pixels == (0, 0)
        assert mask.zoom_shape_native == (3, 3)

        mask = aa.Mask2D.manual(
            mask=np.array(
                [[False, True, True], [True, True, True], [False, True, True]]
            ),
            pixel_scales=(1.0, 1.0),
        )
        assert mask.zoom_centre == (1, 0)
        assert mask.zoom_offset_pixels == (0, -1)
        assert mask.zoom_shape_native == (3, 3)

        mask = aa.Mask2D.manual(
            mask=np.array(
                [[False, True, False], [True, True, True], [True, True, True]]
            ),
            pixel_scales=(1.0, 1.0),
        )
        assert mask.zoom_centre == (0, 1)
        assert mask.zoom_offset_pixels == (-1, 0)
        assert mask.zoom_shape_native == (3, 3)

        mask = aa.Mask2D.manual(
            mask=np.array(
                [[False, False, True], [True, True, True], [True, True, True]]
            ),
            pixel_scales=(1.0, 1.0),
        )
        assert mask.zoom_centre == (0, 0.5)
        assert mask.zoom_offset_pixels == (-1, -0.5)
        assert mask.zoom_shape_native == (1, 2)

    def test__rectangular_mask(self):
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

    def test__zoom_mask_unmasked__is_mask_over_zoomed_region(self):

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


class TestCoordinates:
    def test__central_pixel__gives_same_result_as_geometry_util(self):

        mask = aa.Mask2D.unmasked(shape_native=(3, 3), pixel_scales=(0.1, 0.1))

        central_pixel_coordinates_util = aa.util.geometry.central_pixel_coordinates_2d_from(
            shape_native=(3, 3)
        )

        assert mask.central_pixel_coordinates == central_pixel_coordinates_util

        mask = aa.Mask2D.unmasked(
            shape_native=(5, 3), pixel_scales=(2.0, 1.0), origin=(1.0, 2.0)
        )

        central_pixel_coordinates_util = aa.util.geometry.central_pixel_coordinates_2d_from(
            shape_native=(5, 3)
        )

        assert mask.central_pixel_coordinates == central_pixel_coordinates_util

    def test__centring__adapts_to_max_and_min_of_mask(self):
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


class TestGrids:
    def test__unmasked_grid__compare_to_grid_util(self):

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

    def test__masked_grids_1d(self):

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
            mask.masked_grid_sub_1.slim
            == np.array([[4.0, -2.5], [3.0, -1.5], [2.0, -1.5]])
        ).all()

    def test__edge_grid(self):
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

    def test__border_grid(self):
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

    def test__masked_sub_grid(self):
        mask = aa.Mask2D.unmasked(
            shape_native=(3, 3), pixel_scales=(1.0, 1.0), sub_size=1
        )

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

        mask = aa.Mask2D.unmasked(
            shape_native=(2, 2), pixel_scales=(1.0, 1.0), sub_size=2
        )

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

        mask = aa.Mask2D.unmasked(
            shape_native=(3, 3), pixel_scales=(1.0, 1.0), sub_size=1
        )
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

    def test__sub_border_1d_grid__compare_numerical_values(self):

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
            mask.border_grid_1d
            == np.array([[1.25, -2.25], [1.25, -1.25], [-0.25, 1.25]])
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


class TestScaledToPixel:
    def test__pixel_coordinates_2d_from__gives_same_result_as_geometry_util(self):

        mask = aa.Mask2D.unmasked(
            shape_native=(6, 7), pixel_scales=(2.4, 1.8), origin=(1.0, 1.5)
        )

        pixel_coordinates_util = aa.util.geometry.pixel_coordinates_2d_from(
            scaled_coordinates_2d=(2.3, 1.2),
            shape_native=(6, 7),
            pixel_scales=(2.4, 1.8),
            origins=(1.0, 1.5),
        )

        assert (
            mask.pixel_coordinates_2d_from(scaled_coordinates_2d=(2.3, 1.2))
            == pixel_coordinates_util
        )

    def test__scaled_coordinates_2d_from___gives_same_result_as_geometry_util(self,):

        mask = aa.Mask2D.unmasked(
            shape_native=(6, 7), pixel_scales=(2.4, 1.8), origin=(1.0, 1.5)
        )

        pixel_coordinates_util = aa.util.geometry.scaled_coordinates_2d_from(
            pixel_coordinates_2d=(5, 4),
            shape_native=(6, 7),
            pixel_scales=(2.4, 1.8),
            origins=(1.0, 1.5),
        )

        assert (
            mask.scaled_coordinates_2d_from(pixel_coordinates_2d=(5, 4))
            == pixel_coordinates_util
        )


class TestGridConversions:
    def test__grid_pixels_from_grid_scaled(self):
        mask = aa.Mask2D.unmasked(shape_native=(2, 2), pixel_scales=(2.0, 4.0))

        grid_scaled_1d = np.array([[1.0, -2.0], [1.0, 2.0], [-1.0, -2.0], [-1.0, 2.0]])

        grid_pixels_util = aa.util.grid_2d.grid_pixels_2d_slim_from(
            grid_scaled_2d_slim=grid_scaled_1d,
            shape_native=(2, 2),
            pixel_scales=(2.0, 4.0),
        )
        grid_pixels = mask.grid_pixels_from_grid_scaled_1d(
            grid_scaled_1d=grid_scaled_1d
        )

        assert (grid_pixels == grid_pixels_util).all()
        assert (grid_pixels.slim == grid_pixels_util).all()

    def test__grid_pixel_centres_1d_from_grid_scaled_1d__same_as_grid_util(self):

        mask = aa.Mask2D.unmasked(shape_native=(2, 2), pixel_scales=(2.0, 2.0))

        grid_scaled_1d = np.array([[0.5, -0.5], [0.5, 0.5], [-0.5, -0.5], [-0.5, 0.5]])

        grid_pixels_util = aa.util.grid_2d.grid_pixel_centres_2d_slim_from(
            grid_scaled_2d_slim=grid_scaled_1d,
            shape_native=(2, 2),
            pixel_scales=(2.0, 2.0),
        )

        grid_pixels = mask.grid_pixel_centres_from_grid_scaled_1d(
            grid_scaled_1d=grid_scaled_1d
        )

        assert (grid_pixels == grid_pixels_util).all()

        mask = aa.Mask2D.unmasked(shape_native=(2, 2), pixel_scales=(7.0, 2.0))

        grid_scaled_1d = np.array([[1.0, -2.0], [1.0, 2.0], [-1.0, -2.0], [-1.0, 2.0]])

        grid_pixels_util = aa.util.grid_2d.grid_pixel_centres_2d_slim_from(
            grid_scaled_2d_slim=grid_scaled_1d,
            shape_native=(2, 2),
            pixel_scales=(7.0, 2.0),
        )

        grid_pixels = mask.grid_pixel_centres_from_grid_scaled_1d(
            grid_scaled_1d=grid_scaled_1d
        )

        assert (grid_pixels == grid_pixels_util).all()

    def test__grid_pixel_indexes_1d_from_grid_scaled_1d__same_as_grid_util(self):
        mask = aa.Mask2D.unmasked(shape_native=(2, 2), pixel_scales=(2.0, 2.0))

        grid_scaled = np.array([[1.0, -1.0], [1.0, 1.0], [-1.0, -1.0], [-1.0, 1.0]])

        grid_pixel_indexes_util = aa.util.grid_2d.grid_pixel_indexes_2d_slim_from(
            grid_scaled_2d_slim=grid_scaled,
            shape_native=(2, 2),
            pixel_scales=(2.0, 2.0),
        )

        grid_pixel_indexes = mask.grid_pixel_indexes_from_grid_scaled_1d(
            grid_scaled_1d=grid_scaled
        )

        assert (grid_pixel_indexes == grid_pixel_indexes_util).all()

        mask = aa.Mask2D.unmasked(shape_native=(2, 2), pixel_scales=(2.0, 4.0))

        grid_scaled = np.array([[1.0, -2.0], [1.0, 2.0], [-1.0, -2.0], [-1.0, 2.0]])

        grid_pixels_util = aa.util.grid_2d.grid_pixel_indexes_2d_slim_from(
            grid_scaled_2d_slim=grid_scaled,
            shape_native=(2, 2),
            pixel_scales=(2.0, 4.0),
        )

        grid_pixels = mask.grid_pixel_indexes_from_grid_scaled_1d(
            grid_scaled_1d=grid_scaled
        )

        assert (grid_pixels == grid_pixels_util).all()

    def test__grid_scaled_1d_from_grid_pixels_1d__same_as_grid_util(self):
        mask = aa.Mask2D.unmasked(shape_native=(2, 2), pixel_scales=(2.0, 2.0))

        grid_pixels = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])

        grid_pixels_util = aa.util.grid_2d.grid_scaled_2d_slim_from(
            grid_pixels_2d_slim=grid_pixels,
            shape_native=(2, 2),
            pixel_scales=(2.0, 2.0),
        )

        grid_pixels = mask.grid_scaled_from_grid_pixels_1d(grid_pixels_1d=grid_pixels)

        assert (grid_pixels == grid_pixels_util).all()

        mask = aa.Mask2D.unmasked(shape_native=(2, 2), pixel_scales=(2.0, 2.0))

        grid_pixels = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])

        grid_pixels_util = aa.util.grid_2d.grid_scaled_2d_slim_from(
            grid_pixels_2d_slim=grid_pixels,
            shape_native=(2, 2),
            pixel_scales=(2.0, 2.0),
        )
        grid_pixels = mask.grid_scaled_from_grid_pixels_1d(grid_pixels_1d=grid_pixels)

        assert (grid_pixels == grid_pixels_util).all()

    def test__pixel_grid__grids_with_nonzero_centres__same_as_grid_util(self):
        mask = aa.Mask2D.unmasked(
            shape_native=(2, 2), pixel_scales=(2.0, 2.0), origin=(1.0, 2.0)
        )

        grid_scaled = np.array([[1.0, -2.0], [1.0, 2.0], [-1.0, -2.0], [-1.0, 2.0]])

        grid_pixels_util = aa.util.grid_2d.grid_pixels_2d_slim_from(
            grid_scaled_2d_slim=grid_scaled,
            shape_native=(2, 2),
            pixel_scales=(2.0, 2.0),
            origin=(1.0, 2.0),
        )
        grid_pixels = mask.grid_pixels_from_grid_scaled_1d(grid_scaled_1d=grid_scaled)
        assert (grid_pixels == grid_pixels_util).all()

        grid_pixels_util = aa.util.grid_2d.grid_pixel_indexes_2d_slim_from(
            grid_scaled_2d_slim=grid_scaled,
            shape_native=(2, 2),
            pixel_scales=(2.0, 2.0),
            origin=(1.0, 2.0),
        )
        grid_pixels = mask.grid_pixel_indexes_from_grid_scaled_1d(
            grid_scaled_1d=grid_scaled
        )
        assert grid_pixels == pytest.approx(grid_pixels_util, 1e-4)

        grid_pixels_util = aa.util.grid_2d.grid_pixel_centres_2d_slim_from(
            grid_scaled_2d_slim=grid_scaled,
            shape_native=(2, 2),
            pixel_scales=(2.0, 2.0),
            origin=(1.0, 2.0),
        )
        grid_pixels = mask.grid_pixel_centres_from_grid_scaled_1d(
            grid_scaled_1d=grid_scaled
        )
        assert grid_pixels == pytest.approx(grid_pixels_util, 1e-4)

        grid_pixels = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])

        grid_scaled_util = aa.util.grid_2d.grid_scaled_2d_slim_from(
            grid_pixels_2d_slim=grid_pixels,
            shape_native=(2, 2),
            pixel_scales=(2.0, 2.0),
            origin=(1.0, 2.0),
        )

        grid_scaled = mask.grid_scaled_from_grid_pixels_1d(grid_pixels_1d=grid_pixels)

        assert (grid_scaled == grid_scaled_util).all()

        grid_scaled = np.array([[1.0, -2.0], [1.0, 2.0], [-1.0, -2.0], [-1.0, 2.0]])

        mask = aa.Mask2D.unmasked(
            shape_native=(2, 2), pixel_scales=(2.0, 1.0), origin=(1.0, 2.0)
        )

        grid_pixels_util = aa.util.grid_2d.grid_pixels_2d_slim_from(
            grid_scaled_2d_slim=grid_scaled,
            shape_native=(2, 2),
            pixel_scales=(2.0, 1.0),
            origin=(1.0, 2.0),
        )
        grid_pixels = mask.grid_pixels_from_grid_scaled_1d(grid_scaled_1d=grid_scaled)
        assert (grid_pixels == grid_pixels_util).all()

        grid_pixels_util = aa.util.grid_2d.grid_pixel_indexes_2d_slim_from(
            grid_scaled_2d_slim=grid_scaled,
            shape_native=(2, 2),
            pixel_scales=(2.0, 1.0),
            origin=(1.0, 2.0),
        )
        grid_pixels = mask.grid_pixel_indexes_from_grid_scaled_1d(
            grid_scaled_1d=grid_scaled
        )
        assert (grid_pixels == grid_pixels_util).all()

        grid_pixels_util = aa.util.grid_2d.grid_pixel_centres_2d_slim_from(
            grid_scaled_2d_slim=grid_scaled,
            shape_native=(2, 2),
            pixel_scales=(2.0, 1.0),
            origin=(1.0, 2.0),
        )
        grid_pixels = mask.grid_pixel_centres_from_grid_scaled_1d(
            grid_scaled_1d=grid_scaled
        )
        assert grid_pixels == pytest.approx(grid_pixels_util, 1e-4)

        grid_pixels = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])

        grid_scaled_util = aa.util.grid_2d.grid_scaled_2d_slim_from(
            grid_pixels_2d_slim=grid_pixels,
            shape_native=(2, 2),
            pixel_scales=(2.0, 1.0),
            origin=(1.0, 2.0),
        )

        grid_scaled = mask.grid_scaled_from_grid_pixels_1d(grid_pixels_1d=grid_pixels)

        assert (grid_scaled == grid_scaled_util).all()
