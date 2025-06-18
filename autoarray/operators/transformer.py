from astropy import units
import copy
import numpy as np
import warnings


class NUFFTPlaceholder:
    pass


try:
    from pynufft.linalg.nufft_cpu import NUFFT_cpu
except ModuleNotFoundError:
    NUFFT_cpu = NUFFTPlaceholder


from autoarray.structures.arrays.uniform_2d import Array2D
from autoarray.structures.grids.uniform_2d import Grid2D
from autoarray.structures.visibilities import Visibilities

from autoarray.structures.arrays import array_2d_util
from autoarray.operators import transformer_util


def pynufft_exception():
    raise ModuleNotFoundError(
        "\n--------------------\n"
        "You are attempting to perform interferometer analysis.\n\n"
        "However, the optional library PyNUFFT (https://github.com/jyhmiinlin/pynufft) is not installed.\n\n"
        "Install it via the command `pip install pynufft==2022.2.2`.\n\n"
        "----------------------"
    )


class TransformerDFT:
    def __init__(self, uv_wavelengths, real_space_mask, preload_transform=True):

        super().__init__()

        self.uv_wavelengths = uv_wavelengths.astype("float")
        self.real_space_mask = real_space_mask
        self.grid = self.real_space_mask.derive_grid.unmasked.in_radians

        self.total_visibilities = uv_wavelengths.shape[0]
        self.total_image_pixels = self.real_space_mask.pixels_in_mask

        self.preload_transform = preload_transform

        if preload_transform:
            self.preload_real_transforms = transformer_util.preload_real_transforms_from(
                grid_radians=np.array(self.grid.array),
                uv_wavelengths=self.uv_wavelengths,
            )

            self.preload_imag_transforms = transformer_util.preload_imag_transforms_from(
                grid_radians=np.array(self.grid.array),
                uv_wavelengths=self.uv_wavelengths,
            )

        self.real_space_pixels = self.real_space_mask.pixels_in_mask

        self.shape = (
            int(np.prod(self.total_visibilities)),
            int(np.prod(self.real_space_pixels)),
        )
        self.dtype = "complex128"
        self.explicit = False

        # NOTE: This is the scaling factor that needs to be applied to the adjoint operator
        self.adjoint_scaling = (2.0 * self.grid.shape_native[0]) * (
            2.0 * self.grid.shape_native[1]
        )

        self.matvec_count = 0
        self.rmatvec_count = 0
        self.matmat_count = 0
        self.rmatmat_count = 0

    def visibilities_from(self, image):
        if self.preload_transform:
            visibilities = transformer_util.visibilities_via_preload_from(
                image_1d=np.array(image.array),
                preloaded_reals=self.preload_real_transforms_from,
                preloaded_imags=self.preload_imag_transforms_from,
            )

        else:
            visibilities = transformer_util.visibilities_from(
                image_1d=np.array(image.slim.array),
                grid_radians=np.array(self.grid),
                uv_wavelengths=self.uv_wavelengths,
            )

        return Visibilities(visibilities=visibilities)

    def image_from(self, visibilities, use_adjoint_scaling: bool = False):
        image_slim = transformer_util.image_direct_from(
            visibilities=visibilities.in_array,
            grid_radians=np.array(self.grid.array),
            uv_wavelengths=self.uv_wavelengths,
        )

        image_native = array_2d_util.array_2d_native_from(
            array_2d_slim=image_slim,
            mask_2d=self.real_space_mask,
        )

        return Array2D(values=image_native, mask=self.real_space_mask)

    def transform_mapping_matrix(self, mapping_matrix):
        if self.preload_transform:
            return transformer_util.transformed_mapping_matrix_via_preload_jit_from(
                mapping_matrix=mapping_matrix,
                preloaded_reals=self.preload_real_transforms_from,
                preloaded_imags=self.preload_imag_transforms_from,
            )

        else:
            return transformer_util.transformed_mapping_matrix_jit(
                mapping_matrix=mapping_matrix,
                grid_radians=np.array(self.grid),
                uv_wavelengths=self.uv_wavelengths,
            )


class TransformerNUFFT(NUFFT_cpu):
    def __init__(self, uv_wavelengths, real_space_mask):
        if isinstance(self, NUFFTPlaceholder):
            pynufft_exception()

        super(TransformerNUFFT, self).__init__()

        self.uv_wavelengths = uv_wavelengths
        self.real_space_mask = real_space_mask
        #        self.grid = self.real_space_mask.unmasked_grid.in_radians
        self.grid = Grid2D.from_mask(mask=self.real_space_mask).in_radians
        self.native_index_for_slim_index = copy.copy(
            real_space_mask.derive_indexes.native_for_slim.astype("int")
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

        self.matvec_count = 0
        self.rmatvec_count = 0
        self.matmat_count = 0
        self.rmatmat_count = 0

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

    def visibilities_from(self, image):
        """
        ...
        """

        warnings.filterwarnings("ignore")

        return Visibilities(
            visibilities=self.forward(
                image.native[::-1, :]
            )  # flip due to PyNUFFT internal flip
        )

    def image_from(self, visibilities, use_adjoint_scaling: bool = False):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            image = np.real(self.adjoint(visibilities))[::-1, :]

        if use_adjoint_scaling:
            image *= self.adjoint_scaling

        return Array2D(values=image, mask=self.real_space_mask)

    def transform_mapping_matrix(self, mapping_matrix):
        transformed_mapping_matrix = 0 + 0j * np.zeros(
            (self.uv_wavelengths.shape[0], mapping_matrix.shape[1])
        )

        for source_pixel_1d_index in range(mapping_matrix.shape[1]):
            image_2d = array_2d_util.array_2d_native_from(
                array_2d_slim=mapping_matrix[:, source_pixel_1d_index],
                mask_2d=self.grid.mask,
            )

            image = Array2D(values=image_2d, mask=self.grid.mask)

            visibilities = self.visibilities_from(image=image)

            transformed_mapping_matrix[:, source_pixel_1d_index] = visibilities

        return transformed_mapping_matrix
