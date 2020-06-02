from autoarray.util import transformer_util
from autoarray.structures import arrays, visibilities as vis
from autoarray.util import array_util
from astropy import units
from scipy import interpolate
from pynufft import NUFFT_cpu

import numpy as np

from autoarray import exc


class TransformerDFT:
    def __init__(self, uv_wavelengths, grid, preload_transform=True):

        self.uv_wavelengths = uv_wavelengths.astype("float")
        self.grid = grid.in_1d_binned

        self.total_visibilities = uv_wavelengths.shape[0]
        self.total_image_pixels = grid.shape_1d

        self.preload_transform = preload_transform

        if preload_transform:

            self.preload_real_transforms = transformer_util.preload_real_transforms(
                grid_radians=self.grid, uv_wavelengths=self.uv_wavelengths
            )

            self.preload_imag_transforms = transformer_util.preload_imag_transforms(
                grid_radians=self.grid, uv_wavelengths=self.uv_wavelengths
            )

    def real_visibilities_from_image(self, image):

        if self.preload_transform:

            return transformer_util.real_visibilities_via_preload_jit_from(
                image_1d=image.in_1d_binned,
                preloaded_reals=self.preload_real_transforms,
            )

        else:

            return transformer_util.real_visibilities_jit(
                image_1d=image.in_1d_binned,
                grid_radians=self.grid,
                uv_wavelengths=self.uv_wavelengths,
            )

    def imag_visibilities_from_image(self, image):

        if self.preload_transform:

            return transformer_util.imag_visibilities_from_via_preload_jit_from(
                image_1d=image.in_1d_binned,
                preloaded_imags=self.preload_imag_transforms,
            )

        else:

            return transformer_util.imag_visibilities_jit(
                image_1d=image.in_1d_binned,
                grid_radians=self.grid,
                uv_wavelengths=self.uv_wavelengths,
            )

    def visibilities_from_image(self, image):

        real_visibilities = self.real_visibilities_from_image(image=image)
        imag_visibilities = self.imag_visibilities_from_image(image=image)

        return vis.Visibilities(
            visibilities_1d=np.stack((real_visibilities, imag_visibilities), axis=-1)
        )

    def real_transformed_mapping_matrix_from_mapping_matrix(self, mapping_matrix):

        if self.preload_transform:

            return transformer_util.real_transformed_mapping_matrix_via_preload_jit_from(
                mapping_matrix=mapping_matrix,
                preloaded_reals=self.preload_real_transforms,
            )

        else:

            return transformer_util.real_transformed_mapping_matrix_jit(
                mapping_matrix=mapping_matrix,
                grid_radians=self.grid,
                uv_wavelengths=self.uv_wavelengths,
            )

    def imag_transformed_mapping_matrix_from_mapping_matrix(self, mapping_matrix):

        if self.preload_transform:

            return transformer_util.imag_transformed_mapping_matrix_via_preload_jit_from(
                mapping_matrix=mapping_matrix,
                preloaded_imags=self.preload_imag_transforms,
            )

        else:

            return transformer_util.imag_transformed_mapping_matrix_jit(
                mapping_matrix=mapping_matrix,
                grid_radians=self.grid,
                uv_wavelengths=self.uv_wavelengths,
            )

    def transformed_mapping_matrices_from_mapping_matrix(self, mapping_matrix):

        real_transformed_mapping_matrix = self.real_transformed_mapping_matrix_from_mapping_matrix(
            mapping_matrix=mapping_matrix
        )
        imag_transformed_mapping_matrix = self.imag_transformed_mapping_matrix_from_mapping_matrix(
            mapping_matrix=mapping_matrix
        )

        return [real_transformed_mapping_matrix, imag_transformed_mapping_matrix]


class TransformerFFT(object):
    def __init__(self, uv_wavelengths, grid):

        super(TransformerFFT, self).__init__()

        self.uv_wavelengths = uv_wavelengths.astype("float")
        self.grid = grid.in_1d_binned

        self.u_fft = np.fft.fftshift(
            np.fft.fftfreq(
                grid.shape_2d[0], grid.pixel_scales[0] * units.arcsec.to(units.rad)
            )
        )
        self.v_fft = np.fft.fftshift(
            np.fft.fftfreq(
                grid.shape_2d[1], grid.pixel_scales[1] * units.arcsec.to(units.rad)
            )
        )
        u_fft_meshgrid, v_fft_meshgrid = np.meshgrid(self.u_fft, self.v_fft)

        # ... This is nessesary due to the way the grid in autolens is set up.
        self.shift = np.exp(
            -2.0
            * np.pi
            * 1j
            * (
                self.grid.pixel_scales[0]
                / 2.0
                * units.arcsec.to(units.rad)
                * u_fft_meshgrid
                + self.grid.pixel_scales[0]
                / 2.0
                * units.arcsec.to(units.rad)
                * v_fft_meshgrid
            )
        )

        self.uv = np.array(
            list(zip(self.uv_wavelengths[:, 0], self.uv_wavelengths[:, 1]))
        )

    def visibilities_from_image(self, image):
        """
        Generate visibilities from an image (in this case the image was created using autolens).
        """

        # NOTE: The input image is flipped to account for the way autolens is generating images
        z_fft = np.fft.fftshift(
            np.fft.fft2(np.fft.fftshift(image.in_2d_binned[::-1, :]))
        )

        # ...
        z_fft_shifted = z_fft * self.shift

        # ...

        real_interp = interpolate.RegularGridInterpolator(
            points=(self.u_fft, self.v_fft),
            values=z_fft_shifted.real.T,
            method="linear",
            bounds_error=False,
            fill_value=0.0,
        )
        imag_interp = interpolate.RegularGridInterpolator(
            points=(self.u_fft, self.v_fft),
            values=z_fft_shifted.imag.T,
            method="linear",
            bounds_error=False,
            fill_value=0.0,
        )

        real_visibilities = real_interp(self.uv)
        imag_visibilities = imag_interp(self.uv)

        return vis.Visibilities(
            visibilities_1d=np.stack((real_visibilities, imag_visibilities), axis=-1)
        )

    def transformed_mapping_matrices_from_mapping_matrix(self, mapping_matrix):
        """
        ...
        """

        real_transfomed_mapping_matrix = np.zeros(
            (self.uv_wavelengths.shape[0], mapping_matrix.shape[1])
        )
        imag_transfomed_mapping_matrix = np.zeros(
            (self.uv_wavelengths.shape[0], mapping_matrix.shape[1])
        )

        for source_pixel_1d_index in range(mapping_matrix.shape[1]):

            image = arrays.Array.manual_1d(
                array=mapping_matrix[:, source_pixel_1d_index],
                shape_2d=self.grid.shape_2d,
                pixel_scales=self.grid.pixel_scales,
                store_in_1d=False,
            )

            visibilities = self.visibilities_from_image(image=image)

            real_transfomed_mapping_matrix[:, source_pixel_1d_index] = visibilities.real
            imag_transfomed_mapping_matrix[:, source_pixel_1d_index] = visibilities.imag

        return [real_transfomed_mapping_matrix, imag_transfomed_mapping_matrix]


class TransformerNUFFT(NUFFT_cpu):
    def __init__(self, uv_wavelengths, grid):

        super(TransformerNUFFT, self).__init__()

        self.uv_wavelengths = uv_wavelengths
        self.grid = grid.in_1d_binned

        # NOTE: The plan need only be initialized once
        self.initialize_plan()

        # ...
        self.shift = np.exp(
            -2.0
            * np.pi
            * 1j
            * (
                self.grid.pixel_scales[0]
                / 2.0
                * units.arcsec.to(units.rad)
                * self.uv_wavelengths[:, 1]
                + self.grid.pixel_scales[0]
                / 2.0
                * units.arcsec.to(units.rad)
                * self.uv_wavelengths[:, 0]
            )
        )

    def initialize_plan(self, ratio=2, interp_kernel=(6, 6)):

        if not isinstance(ratio, int):
            ratio = int(ratio)

        # ... NOTE : The u,v coordinated should be given in the order ...
        visibilities_normalized = np.array(
            [
                self.uv_wavelengths[:, 1]
                / (1.0 / (2.0 * self.grid.pixel_scales[0] * units.arcsec.to(units.rad)))
                * np.pi,
                self.uv_wavelengths[:, 0]
                / (1.0 / (2.0 * self.grid.pixel_scales[0] * units.arcsec.to(units.rad)))
                * np.pi,
            ]
        ).T

        # NOTE:
        self.plan(
            visibilities_normalized,
            self.grid.shape_2d,
            (ratio * self.grid.shape_2d[0], ratio * self.grid.shape_2d[1]),
            interp_kernel,
        )

    def visibilities_from_image(self, image):
        """
        ...
        """

        # NOTE: Flip the image the autolens produces.
        visibilities = self.forward(image.in_2d_binned[::-1, :])

        # ... NOTE:
        visibilities *= self.shift

        return vis.Visibilities(
            visibilities_1d=np.stack((visibilities.real, visibilities.imag), axis=-1)
        )

    def transformed_mapping_matrices_from_mapping_matrix(self, mapping_matrix):

        real_transfomed_mapping_matrix = np.zeros(
            (self.uv_wavelengths.shape[0], mapping_matrix.shape[1])
        )
        imag_transfomed_mapping_matrix = np.zeros(
            (self.uv_wavelengths.shape[0], mapping_matrix.shape[1])
        )

        for source_pixel_1d_index in range(mapping_matrix.shape[1]):

            image_2d = array_util.sub_array_2d_from(
                sub_array_1d=mapping_matrix[:, source_pixel_1d_index],
                mask=self.grid.mask,
                sub_size=1,
            )

            image = arrays.Array(array=image_2d, mask=self.grid.mask, store_in_1d=False)

            visibilities = self.visibilities_from_image(image=image)

            real_transfomed_mapping_matrix[:, source_pixel_1d_index] = visibilities.real
            imag_transfomed_mapping_matrix[:, source_pixel_1d_index] = visibilities.imag

        return [real_transfomed_mapping_matrix, imag_transfomed_mapping_matrix]
