from autoarray.structures import arrays, grids
from autoarray.operators import convolution

class AbstractMaskedData(object):

    def __init__(
            self,
            mask,
            pixel_scale_interpolation_grid=None,
            inversion_pixel_limit=None,
            inversion_uses_border=True,
            hyper_noise_map_max=None,
    ):

        self.mask = mask

        ### GRIDS ###

        if mask.pixel_scales is not None:

            self.grid = grids.MaskedGrid.from_mask(mask=mask)

            if pixel_scale_interpolation_grid is not None:

                self.pixel_scale_interpolation_grid = pixel_scale_interpolation_grid

                self.grid = self.grid.new_grid_with_interpolator(
                    pixel_scale_interpolation_grid=pixel_scale_interpolation_grid
                )

        else:

            self.grid = None

        self.hyper_noise_map_max = hyper_noise_map_max

        self.inversion_pixel_limit = inversion_pixel_limit
        self.inversion_uses_border = inversion_uses_border


class MaskedImaging(AbstractMaskedData):

    def __init__(
            self,
            imaging,
            mask,
            trimmed_psf_shape_2d=None,
            pixel_scale_interpolation_grid=None,
            inversion_pixel_limit=None,
            inversion_uses_border=True,
            hyper_noise_map_max=None,
    ):
        """
        The lens data is the collection of data_type (image, noise-map, PSF), a mask, grid, convolver \
        and other utilities that are used for modeling and fitting an image of a strong lens.

        Whilst the image, noise-map, etc. are loaded in 2D, the lens data creates reduced 1D arrays of each \
        for lensing calculations.

        Parameters
        ----------
        imaging: im.Imaging
            The imaging data_type all in 2D (the image, noise-map, PSF, etc.)
        mask: msk.Mask
            The 2D mask that is applied to the image.
        sub_size : int
            The size of the sub-grid used for each lens SubGrid. E.g. a value of 2 grid each image-pixel on a 2x2 \
            sub-grid.
        trimmed_psf_shape_2d : (int, int)
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
            hyper_noise_map_max=hyper_noise_map_max,
        )

        self.image = mask.mapping.array_from_array_2d(array_2d=imaging.image.in_2d)
        self.noise_map = mask.mapping.array_from_array_2d(array_2d=imaging.noise_map.in_2d)

        self.pixel_scale_interpolation_grid = pixel_scale_interpolation_grid

        ### PSF TRIMMING + CONVOLVER ###

        if hasattr(imaging, 'psf'):

            if trimmed_psf_shape_2d is None:
                self.trimmed_psf_shape_2d = imaging.psf.shape_2d
            else:
                self.trimmed_psf_shape_2d = trimmed_psf_shape_2d

            self.psf = imaging.psf.resized_from_new_shape(new_shape=self.trimmed_psf_shape_2d)

            self.convolver = convolution.Convolver(
                mask_2d=mask,
                kernel_2d=self.psf.in_2d,
            )

            if mask.pixel_scales is not None:

                self.blurring_grid = self.grid.blurring_grid_from_kernel_shape(
                    kernel_shape=self.trimmed_psf_shape_2d
                )

                if pixel_scale_interpolation_grid is not None:

                    self.blurring_grid = self.blurring_grid.new_grid_with_interpolator(
                        pixel_scale_interpolation_grid=self.pixel_scale_interpolation_grid
                    )

    @classmethod
    def manual(cls, imaging,
               mask,
               trimmed_psf_shape_2d=None,
               pixel_scale_interpolation_grid=None,
               inversion_pixel_limit=None,
               inversion_uses_border=True,
               hyper_noise_map_max=None):
        return cls(imaging=imaging, mask=mask,
                   trimmed_psf_shape_2d=trimmed_psf_shape_2d,
                   pixel_scale_interpolation_grid=pixel_scale_interpolation_grid,
                   inversion_pixel_limit=inversion_pixel_limit, inversion_uses_border=inversion_uses_border,
                   hyper_noise_map_max=hyper_noise_map_max)

    def binned_from_bin_up_factor(self, bin_up_factor):

        binned_imaging = self.imaging.binned_from_bin_up_factor(
            bin_up_factor=bin_up_factor
        )
        binned_mask = self.mask.mapping.binned_mask_from_bin_up_factor(bin_up_factor=bin_up_factor)

        return MaskedImaging(
            imaging=binned_imaging,
            mask=binned_mask,
            trimmed_psf_shape_2d=self.trimmed_psf_shape_2d,
            pixel_scale_interpolation_grid=self.pixel_scale_interpolation_grid,
            inversion_pixel_limit=self.inversion_pixel_limit,
            inversion_uses_border=self.inversion_uses_border,
            hyper_noise_map_max=self.hyper_noise_map_max,
        )

    def signal_to_noise_limited_from_signal_to_noise_limit(self, signal_to_noise_limit):

        imaging_with_signal_to_noise_limit = self.imaging.signal_to_noise_limited_from_signal_to_noise_limit(
            signal_to_noise_limit=signal_to_noise_limit
        )

        return MaskedImaging(
            imaging=imaging_with_signal_to_noise_limit,
            mask=self.mask,
            trimmed_psf_shape_2d=self.trimmed_psf_shape_2d,
            pixel_scale_interpolation_grid=self.pixel_scale_interpolation_grid,
            inversion_pixel_limit=self.inversion_pixel_limit,
            inversion_uses_border=self.inversion_uses_border,
            hyper_noise_map_max=self.hyper_noise_map_max,
        )