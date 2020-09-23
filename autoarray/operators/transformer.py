from autoarray.util import transformer_util
from autoarray.structures import arrays, visibilities as vis, grids
from autoarray.util import array_util
from astropy import units
from pynufft import NUFFT_cpu
import pylops

import copy
import numpy as np


class TransformerDFT:
    def __init__(self, uv_wavelengths, real_space_mask, preload_transform=True):

        self.uv_wavelengths = uv_wavelengths.astype("float")
        self.real_space_mask = real_space_mask.mask_sub_1
        self.grid = (
            self.real_space_mask.geometry.masked_grid_sub_1.in_1d_binned.in_radians
        )

        self.total_visibilities = uv_wavelengths.shape[0]
        self.total_image_pixels = self.real_space_mask.pixels_in_mask

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


class TransformerNUFFT(NUFFT_cpu, pylops.LinearOperator):
    def __init__(self, uv_wavelengths, real_space_mask):

        super(TransformerNUFFT, self).__init__()

        self.uv_wavelengths = uv_wavelengths
        self.real_space_mask = real_space_mask.mask_sub_1
        #        self.grid = self.real_space_mask.geometry.unmasked_grid.in_radians
        self.grid = grids.Grid.from_mask(mask=self.real_space_mask).in_radians
        self._mask_index_for_mask_1d_index = copy.copy(
            real_space_mask.regions._mask_index_for_mask_1d_index.astype("int")
        )

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

        self.real_space_pixels = self.real_space_mask.pixels_in_mask
        self.total_visibilities = uv_wavelengths.shape[0]

        self.shape = (
            int(np.prod(self.total_visibilities)),
            int(np.prod(self.real_space_pixels)),
        )
        self.dtype = "complex128"
        self.explicit = False

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
            om=visibilities_normalized,
            Nd=self.grid.shape_2d,
            Kd=(ratio * self.grid.shape_2d[0], ratio * self.grid.shape_2d[1]),
            Jd=interp_kernel,
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

    def forward_lop(self, x):
        """
        Forward NUFFT on CPU
        :param x: The input numpy array, with the size of Nd or Nd + (batch,)
        :type: numpy array with the dtype of numpy.complex64
        :return: y: The output numpy array, with the size of (M,) or (M, batch)
        :rtype: numpy array with the dtype of numpy.complex64
        """
        x2d = array_util.sub_array_complex_2d_via_sub_indexes_from(
            sub_array_1d=x,
            sub_shape=self.real_space_mask.shape_2d,
            sub_mask_index_for_sub_mask_1d_index=self._mask_index_for_mask_1d_index,
        )

        return self.k2y(self.xx2k(self.x2xx(x2d)))

    def adjoint_lop(self, y):
        """
        Adjoint NUFFT on CPU
        :param y: The input numpy array, with the size of (M,) or (M, batch)
        :type: numpy array with the dtype of numpy.complex64
        :return: x: The output numpy array,
                    with the size of Nd or Nd + (batch, )
        :rtype: numpy array with the dtype of numpy.complex64
        """
        x = self.xx2x(self.k2xx(self.y2k(y)))
        return array_util.sub_array_complex_1d_from(
            sub_array_2d=x, sub_size=1, mask=self.real_space_mask
        )

    def _matvec(self, x):
        return self.forward_lop(x)

    def _rmatvec(self, x):
        return self.adjoint_lop(x)
