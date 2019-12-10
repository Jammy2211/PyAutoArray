from autoarray.structures import kernel
from autoarray.masked import masked_structures
from autoarray.operators import convolver, transformer

import numpy as np


class AbstractMaskedDataset(object):
    def __init__(
        self,
        mask,
        pixel_scale_interpolation_grid=None,
        inversion_pixel_limit=None,
        inversion_uses_border=True,
    ):

        self.mask = mask

        ### GRIDS ###

        if mask.pixel_scales is not None:

            self.grid = masked_structures.MaskedGrid.from_mask(mask=mask)

            if pixel_scale_interpolation_grid is not None:

                self.pixel_scale_interpolation_grid = pixel_scale_interpolation_grid

                self.grid = self.grid.new_grid_with_interpolator(
                    pixel_scale_interpolation_grid=pixel_scale_interpolation_grid
                )

        else:

            self.grid = None

        self.inversion_pixel_limit = inversion_pixel_limit
        self.inversion_uses_border = inversion_uses_border


class MaskedImaging(AbstractMaskedDataset):
    def __init__(
        self,
        imaging,
        mask,
        psf_shape_2d=None,
        pixel_scale_interpolation_grid=None,
        inversion_pixel_limit=None,
        inversion_uses_border=True,
    ):
        """
        The lens dataset is the collection of data_type (image, noise-map, PSF), a mask, grid, convolver \
        and other utilities that are used for modeling and fitting an image of a strong lens.

        Whilst the image, noise-map, etc. are loaded in 2D, the lens dataset creates reduced 1D arrays of each \
        for lens calculations.

        Parameters
        ----------
        imaging: im.Imaging
            The imaging data_type all in 2D (the image, noise-map, PSF, etc.)
        mask: msk.Mask
            The 2D mask that is applied to the image.
        sub_size : int
            The size of the sub-grid used for each lens SubGrid. E.g. a value of 2 grid each image-pixel on a 2x2 \
            sub-grid.
        psf_shape_2d : (int, int)
            The shape of the PSF used for convolving model image generated using analytic light profiles. A smaller \
            shape will trim the PSF relative to the input image PSF, giving a faster analysis run-time.
        positions : [[]]
            Lists of image-pixel coordinates (arc-seconds) that mappers close to one another in the source-plane(s), \
            used to speed up the non-linear sampling.
        pixel_scale_interpolation_grid : float
            If *True*, expensive to compute mass profile deflection angles will be computed on a sparse grid and \
            interpolated to the grid, sub and blurring grids.
        inversion_pixel_limit : int or None
            The maximum number of pixels that can be used by an inversion, with the limit placed primarily to speed \
            up run.
        """

        self.imaging = imaging

        super(MaskedImaging, self).__init__(
            mask=mask,
            pixel_scale_interpolation_grid=pixel_scale_interpolation_grid,
            inversion_pixel_limit=inversion_pixel_limit,
            inversion_uses_border=inversion_uses_border,
        )

        self.image = mask.mapping.array_stored_1d_from_array_2d(
            array_2d=imaging.image.in_2d
        )
        self.noise_map = mask.mapping.array_stored_1d_from_array_2d(
            array_2d=imaging.noise_map.in_2d
        )

        self.pixel_scale_interpolation_grid = pixel_scale_interpolation_grid

        ### PSF TRIMMING + CONVOLVER ###

        if imaging.psf is not None:

            if psf_shape_2d is None:
                self.psf_shape_2d = imaging.psf.shape_2d
            else:
                self.psf_shape_2d = psf_shape_2d

            self.psf = kernel.Kernel.manual_2d(
                array=imaging.psf.resized_from_new_shape(
                    new_shape=self.psf_shape_2d
                ).in_2d
            )

            self.convolver = convolver.Convolver(mask=mask, kernel=self.psf)

            if mask.pixel_scales is not None:

                self.blurring_grid = self.grid.blurring_grid_from_kernel_shape(
                    kernel_shape_2d=self.psf_shape_2d
                )

                if pixel_scale_interpolation_grid is not None:

                    self.blurring_grid = self.blurring_grid.new_grid_with_interpolator(
                        pixel_scale_interpolation_grid=self.pixel_scale_interpolation_grid
                    )

    @property
    def data(self):
        return self.image

    @classmethod
    def manual(
        cls,
        imaging,
        mask,
        psf_shape_2d=None,
        pixel_scale_interpolation_grid=None,
        inversion_pixel_limit=None,
        inversion_uses_border=True,
    ):
        return cls(
            imaging=imaging,
            mask=mask,
            psf_shape_2d=psf_shape_2d,
            pixel_scale_interpolation_grid=pixel_scale_interpolation_grid,
            inversion_pixel_limit=inversion_pixel_limit,
            inversion_uses_border=inversion_uses_border,
        )

    def signal_to_noise_map(self):
        return self.image / self.noise_map

    def binned_from_bin_up_factor(self, bin_up_factor):

        binned_imaging = self.imaging.binned_from_bin_up_factor(
            bin_up_factor=bin_up_factor
        )
        binned_mask = self.mask.mapping.binned_mask_from_bin_up_factor(
            bin_up_factor=bin_up_factor
        )

        return self.__class__(
            imaging=binned_imaging,
            mask=binned_mask,
            psf_shape_2d=self.psf_shape_2d,
            pixel_scale_interpolation_grid=self.pixel_scale_interpolation_grid,
            inversion_pixel_limit=self.inversion_pixel_limit,
            inversion_uses_border=self.inversion_uses_border,
        )

    def signal_to_noise_limited_from_signal_to_noise_limit(self, signal_to_noise_limit):

        imaging_with_signal_to_noise_limit = self.imaging.signal_to_noise_limited_from_signal_to_noise_limit(
            signal_to_noise_limit=signal_to_noise_limit
        )

        return self.__class__(
            imaging=imaging_with_signal_to_noise_limit,
            mask=self.mask,
            psf_shape_2d=self.psf_shape_2d,
            pixel_scale_interpolation_grid=self.pixel_scale_interpolation_grid,
            inversion_pixel_limit=self.inversion_pixel_limit,
            inversion_uses_border=self.inversion_uses_border,
        )


class MaskedInterferometer(AbstractMaskedDataset):
    def __init__(
        self,
        interferometer,
        real_space_mask,
        primary_beam_shape_2d=None,
        pixel_scale_interpolation_grid=None,
        inversion_pixel_limit=None,
        inversion_uses_border=True,
    ):
        """
        The lens dataset is the collection of data_type (image, noise-map, primary_beam), a mask, grid, convolver \
        and other utilities that are used for modeling and fitting an image of a strong lens.

        Whilst the image, noise-map, etc. are loaded in 2D, the lens dataset creates reduced 1D arrays of each \
        for lens calculations.

        Parameters
        ----------
        imaging: im.Imaging
            The imaging data_type all in 2D (the image, noise-map, primary_beam, etc.)
        real_space_mask: msk.Mask
            The 2D mask that is applied to the image.
        sub_size : int
            The size of the sub-grid used for each lens SubGrid. E.g. a value of 2 grid each image-pixel on a 2x2 \
            sub-grid.
        primary_beam_shape_2d : (int, int)
            The shape of the primary_beam used for convolving model image generated using analytic light profiles. A smaller \
            shape will trim the primary_beam relative to the input image primary_beam, giving a faster analysis run-time.
        positions : [[]]
            Lists of image-pixel coordinates (arc-seconds) that mappers close to one another in the source-plane(s), \
            used to speed up the non-linear sampling.
        pixel_scale_interpolation_grid : float
            If *True*, expensive to compute mass profile deflection angles will be computed on a sparse grid and \
            interpolated to the grid, sub and blurring grids.
        inversion_pixel_limit : int or None
            The maximum number of pixels that can be used by an inversion, with the limit placed primarily to speed \
            up run.
        """

        self.interferometer = interferometer

        super(MaskedInterferometer, self).__init__(
            mask=real_space_mask,
            pixel_scale_interpolation_grid=pixel_scale_interpolation_grid,
            inversion_pixel_limit=inversion_pixel_limit,
            inversion_uses_border=inversion_uses_border,
        )

        if self.interferometer.primary_beam is None:
            self.primary_beam_shape_2d = None
        elif (
            primary_beam_shape_2d is None
            and self.interferometer.primary_beam is not None
        ):
            self.primary_beam_shape_2d = self.interferometer.primary_beam.shape_2d
        else:
            self.primary_beam_shape_2d = primary_beam_shape_2d

        if self.primary_beam_shape_2d is not None:
            self.primary_beam = kernel.Kernel.manual_2d(
                array=interferometer.primary_beam.resized_from_new_shape(
                    new_shape=self.primary_beam_shape_2d
                ).in_2d
            )

        self.transformer = transformer.Transformer(
            uv_wavelengths=interferometer.uv_wavelengths,
            grid_radians=self.grid.in_1d_binned.in_radians,
        )

        self.visibilities = interferometer.visibilities
        self.noise_map = interferometer.noise_map
        self.visibilities_mask = np.full(
            fill_value=False, shape=self.interferometer.uv_wavelengths.shape
        )

    @property
    def uv_distances(self):
        return self.interferometer.uv_distances

    @property
    def real_space_mask(self):
        return self.mask

    @classmethod
    def manual(
        cls,
        interferometer,
        real_space_mask,
        primary_beam_shape_2d=None,
        pixel_scale_interpolation_grid=None,
        inversion_pixel_limit=None,
        inversion_uses_border=True,
    ):
        return cls(
            interferometer,
            real_space_mask=real_space_mask,
            primary_beam_shape_2d=primary_beam_shape_2d,
            pixel_scale_interpolation_grid=pixel_scale_interpolation_grid,
            inversion_pixel_limit=inversion_pixel_limit,
            inversion_uses_border=inversion_uses_border,
        )

    def signal_to_noise_map(self):
        return self.visibilities / self.noise_map
