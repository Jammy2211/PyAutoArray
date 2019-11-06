from autoarray.util import transformer_util
from autoarray.structures import visibilities as vis

import numpy as np


class Transformer(object):
    def __init__(self, uv_wavelengths, grid_radians, preload_transform=True):

        self.uv_wavelengths = uv_wavelengths.astype("float")
        self.grid_radians = grid_radians.in_1d_binned

        self.total_visibilities = uv_wavelengths.shape[0]
        self.total_image_pixels = grid_radians.shape_1d

        self.preload_transform = preload_transform

        if preload_transform:

            self.preload_real_transforms = transformer_util.preload_real_transforms(
                grid_radians=self.grid_radians, uv_wavelengths=self.uv_wavelengths
            )

            self.preload_imag_transforms = transformer_util.preload_imag_transforms(
                grid_radians=self.grid_radians, uv_wavelengths=self.uv_wavelengths
            )

    def real_visibilities_from_image(self, image):

        if self.preload_transform:

            return transformer_util.real_visibilities_from_image_via_preload(
                image_1d=image.in_1d_binned,
                preloaded_reals=self.preload_real_transforms,
            )

        else:

            return transformer_util.real_visibilities_jit(
                image_1d=image.in_1d_binned,
                grid_radians=self.grid_radians,
                uv_wavelengths=self.uv_wavelengths,
            )

    def imag_visibilities_from_image(self, image):

        if self.preload_transform:

            return transformer_util.imag_visibilities_via_preload_jit(
                image_1d=image.in_1d_binned,
                preloaded_imags=self.preload_imag_transforms,
            )

        else:

            return transformer_util.imag_visibilities_jit(
                image_1d=image.in_1d_binned,
                grid_radians=self.grid_radians,
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

            return transformer_util.real_transformed_mapping_matrix_via_preload_jit(
                mapping_matrix=mapping_matrix,
                preloaded_reals=self.preload_real_transforms,
            )

        else:

            return transformer_util.real_transformed_mapping_matrix_jit(
                mapping_matrix=mapping_matrix,
                grid_radians=self.grid_radians,
                uv_wavelengths=self.uv_wavelengths,
            )

    def imag_transformed_mapping_matrix_from_mapping_matrix(self, mapping_matrix):

        if self.preload_transform:

            return transformer_util.imag_transformed_mapping_matrix_via_preload_jit(
                mapping_matrix=mapping_matrix,
                preloaded_imags=self.preload_imag_transforms,
            )

        else:

            return transformer_util.imag_transformed_mapping_matrix_jit(
                mapping_matrix=mapping_matrix,
                grid_radians=self.grid_radians,
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
