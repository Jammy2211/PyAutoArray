from autoarray.operators import transformer_util
from autoarray.structures.arrays.two_d import array_2d
from autoarray.structures.grids.two_d import grid_2d
from autoarray.structures import visibilities as vis
from autoarray.structures.arrays.two_d import array_2d_util
from astropy import units
from pynufft.linalg.nufft_cpu import NUFFT_cpu
import pylops
import warnings

import copy
import numpy as np


class TransformerDFT(pylops.LinearOperator):
    def __init__(self, uv_wavelengths, real_space_mask, preload_transform=True):

        super(TransformerDFT, self).__init__()

        self.uv_wavelengths = uv_wavelengths.astype("float")
        self.real_space_mask = real_space_mask.mask_sub_1
        self.grid = self.real_space_mask.masked_grid_sub_1.binned.in_radians

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

        self.real_space_pixels = self.real_space_mask.pixels_in_mask

        self.shape = (
            int(np.prod(self.total_visibilities)),
            int(np.prod(self.real_space_pixels)),
        )
        self.dtype = "complex128"
        self.explicit = False

    def visibilities_from_image(self, image):

        if self.preload_transform:

            visibilities = transformer_util.visibilities_via_preload_jit_from(
                image_1d=image.binned,
                preloaded_reals=self.preload_real_transforms,
                preloaded_imags=self.preload_imag_transforms,
            )

        else:

            visibilities = transformer_util.visibilities_jit(
                image_1d=image.binned,
                grid_radians=self.grid,
                uv_wavelengths=self.uv_wavelengths,
            )

        return vis.Visibilities(visibilities=visibilities)

    def transformed_mapping_matrix_from_mapping_matrix(self, mapping_matrix):

        if self.preload_transform:

            return transformer_util.transformed_mapping_matrix_via_preload_jit_from(
                mapping_matrix=mapping_matrix,
                preloaded_reals=self.preload_real_transforms,
                preloaded_imags=self.preload_imag_transforms,
            )

        else:

            return transformer_util.transformed_mapping_matrix_jit(
                mapping_matrix=mapping_matrix,
                grid_radians=self.grid,
                uv_wavelengths=self.uv_wavelengths,
            )


class TransformerNUFFT(NUFFT_cpu, pylops.LinearOperator):
    def __init__(self, uv_wavelengths, real_space_mask):

        super(TransformerNUFFT, self).__init__()

        self.uv_wavelengths = uv_wavelengths
        self.real_space_mask = real_space_mask.mask_sub_1
        #        self.grid = self.real_space_mask.unmasked_grid.in_radians
        self.grid = grid_2d.Grid2D.from_mask(mask=self.real_space_mask).in_radians
        self._sub_native_index_for_sub_slim_index = copy.copy(
            real_space_mask._sub_native_index_for_sub_slim_index.astype("int")
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

        # NOTE: If reshaped the shape of the operator is (2 x Nvis, Np) else it is (Nvis, Np)
        self.total_visibilities = int(uv_wavelengths.shape[0] * uv_wavelengths.shape[1])

        self.shape = (
            int(np.prod(self.total_visibilities)),
            int(np.prod(self.real_space_pixels)),
        )

        # NOTE: If the operator is reshaped then the output is real.
        self.dtype = "float64"

        self.explicit = False

        # NOTE: This is the scaling factor that needs to be applied to the adjoint operator
        self.adjoint_scaling = (2.0 * self.grid.shape_native[0]) * (
            2.0 * self.grid.shape_native[1]
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
            om=visibilities_normalized,
            Nd=self.grid.shape_native,
            Kd=(ratio * self.grid.shape_native[0], ratio * self.grid.shape_native[1]),
            Jd=interp_kernel,
        )

    def visibilities_from_image(self, image):
        """
        ...
        """

        warnings.filterwarnings("ignore")

        return vis.Visibilities(
            visibilities=self.forward(
                image.binned.native[::-1, :]
            )  # flip due to PyNUFFT internal flip
        )

    def image_from_visibilities(self, visibilities):
        image = np.real(self.adjoint(visibilities))
        return array_2d.Array2D.manual_native(array=image, pixel_scales=self.real_space_mask.pixel_scales)

    def transformed_mapping_matrix_from_mapping_matrix(self, mapping_matrix):

        transformed_mapping_matrix = 0 + 0j * np.zeros(
            (self.uv_wavelengths.shape[0], mapping_matrix.shape[1])
        )

        for source_pixel_1d_index in range(mapping_matrix.shape[1]):

            image_2d = array_2d_util.array_2d_native_from(
                array_2d_slim=mapping_matrix[:, source_pixel_1d_index],
                mask_2d=self.grid.mask,
                sub_size=1,
            )

            image = array_2d.Array2D(array=image_2d, mask=self.grid.mask)

            visibilities = self.visibilities_from_image(image=image)

            transformed_mapping_matrix[:, source_pixel_1d_index] = visibilities

        return transformed_mapping_matrix

    def forward_lop(self, x):
        """
        Forward NUFFT on CPU
        :param x: The input numpy array, with the size of Nd or Nd + (batch,)
        :type: numpy array with the dtype of numpy.complex64
        :return: y: The output numpy array, with the size of (M,) or (M, batch)
        :rtype: numpy array with the dtype of numpy.complex64
        """

        warnings.filterwarnings("ignore")

        x2d = array_2d_util.array_2d_native_complex_via_indexes_from(
            array_2d_slim=x,
            sub_shape_native=self.real_space_mask.shape_native,
            native_index_for_slim_index_2d=self._sub_native_index_for_sub_slim_index,
        )[::-1, :]

        y = self.k2y(self.xx2k(self.x2xx(x2d)))
        return np.concatenate((y.real, y.imag), axis=0)

    def adjoint_lop(self, y):
        """
        Adjoint NUFFT on CPU
        :param y: The input numpy array, with the size of (M,) or (M, batch)
        :type: numpy array with the dtype of numpy.complex64
        :return: x: The output numpy array,
                    with the size of Nd or Nd + (batch, )
        :rtype: numpy array with the dtype of numpy.complex64
        """

        warnings.filterwarnings("ignore")

        def a_complex_from_a_real_and_a_imag(a_real, a_imag):

            return a_real + 1j * a_imag

        y = a_complex_from_a_real_and_a_imag(
            a_real=y[: int(self.shape[0] / 2.0)], a_imag=y[int(self.shape[0] / 2.0) :]
        )

        x2d = np.real(self.xx2x(self.k2xx(self.y2k(y))))

        x = array_2d_util.array_2d_slim_complex_from(
            array_2d_native=x2d[::-1, :], sub_size=1, mask=self.real_space_mask
        )
        x = x.real  # NOTE:

        # NOTE:
        x *= self.adjoint_scaling

        return x

    def _matvec(self, x):
        return self.forward_lop(x)

    def _rmatvec(self, x):
        return self.adjoint_lop(x)
