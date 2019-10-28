from autoarray import decorator_util
from autoarray.structures import visibilities as vis

import numpy as np


class Transformer(object):
    def __init__(self, uv_wavelengths, grid_radians, preload_transform=True):

        self.uv_wavelengths = uv_wavelengths
        self.grid_radians = grid_radians

        self.total_visibilities = uv_wavelengths.shape[0]
        self.total_image_pixels = grid_radians.shape[0]

        self.preload_transform = preload_transform

        if preload_transform:

            self.preload_real_transforms = self.preload_real_transforms(
                grid_radians=grid_radians,
                uv_wavelengths=uv_wavelengths,
                total_image_pixels=self.total_image_pixels,
            )

            self.preload_imag_transforms = self.preload_imag_transforms(
                grid_radians=grid_radians,
                uv_wavelengths=uv_wavelengths,
                total_image_pixels=self.total_image_pixels,
            )

    def real_visibilities_from_image(self, image):

        if self.preload_transform:

            return self.real_visibilities_via_preload_jit(
                image_1d=image.in_1d_binned,
                preloaded_reals=self.preload_real_transforms,
                total_visibilities=self.total_visibilities,
                total_image_pixels=self.total_image_pixels,
            )

        else:

            return self.real_visibilities_jit(
                image_1d=image.in_1d_binned,
                grid_radians=self.grid_radians,
                uv_wavelengths=self.uv_wavelengths,
                total_visibilities=self.total_visibilities,
                total_image_pixels=self.total_image_pixels,
            )

    @staticmethod
    @decorator_util.jit()
    def preload_real_transforms(grid_radians, uv_wavelengths, total_image_pixels):

        preloaded_real_transforms = np.zeros(
            shape=(total_image_pixels, uv_wavelengths.shape[0])
        )

        for i in range(total_image_pixels):
            for j in range(uv_wavelengths.shape[0]):
                preloaded_real_transforms[i, j] += np.cos(
                    -2.0
                    * np.pi
                    * (
                        grid_radians[i, 1] * uv_wavelengths[j, 0]
                        - grid_radians[i, 0] * uv_wavelengths[j, 1]
                    )
                )

        return preloaded_real_transforms

    @staticmethod
    @decorator_util.jit()
    def real_visibilities_via_preload_jit(
        image_1d, preloaded_reals, total_visibilities, total_image_pixels
    ):

        real_visibilities = np.zeros(shape=(total_visibilities))

        for i in range(total_image_pixels):
            for j in range(total_visibilities):
                real_visibilities[j] += image_1d[i] * preloaded_reals[i, j]

        return real_visibilities

    @staticmethod
    @decorator_util.jit()
    def real_visibilities_jit(
        image_1d, grid_radians, uv_wavelengths, total_visibilities, total_image_pixels
    ):

        real_visibilities = np.zeros(shape=(total_visibilities))

        for i in range(total_image_pixels):
            for j in range(total_visibilities):
                real_visibilities[j] += image_1d[i] * np.cos(
                    -2.0
                    * np.pi
                    * (
                        grid_radians[i, 1] * uv_wavelengths[j, 0]
                        - grid_radians[i, 0] * uv_wavelengths[j, 1]
                    )
                )

        return real_visibilities

    def imag_visibilities_from_image(self, image):

        if self.preload_transform:

            return self.imag_visibilities_via_preload_jit(
                image_1d=image.in_1d_binned,
                preloaded_imags=self.preload_imag_transforms,
                total_visibilities=self.total_visibilities,
                total_image_pixels=self.total_image_pixels,
            )

        else:

            return self.imag_visibilities_jit(
                image_1d=image.in_1d_binned,
                grid_radians=self.grid_radians,
                uv_wavelengths=self.uv_wavelengths,
                total_visibilities=self.total_visibilities,
                total_image_pixels=self.total_image_pixels,
            )

    @staticmethod
    @decorator_util.jit()
    def preload_imag_transforms(grid_radians, uv_wavelengths, total_image_pixels):

        preloaded_imag_transforms = np.zeros(
            shape=(total_image_pixels, uv_wavelengths.shape[0])
        )

        for i in range(total_image_pixels):
            for j in range(uv_wavelengths.shape[0]):
                preloaded_imag_transforms[i, j] += np.sin(
                    -2.0
                    * np.pi
                    * (
                        grid_radians[i, 1] * uv_wavelengths[j, 0]
                        - grid_radians[i, 0] * uv_wavelengths[j, 1]
                    )
                )

        return preloaded_imag_transforms

    @staticmethod
    @decorator_util.jit()
    def imag_visibilities_via_preload_jit(
        image_1d, preloaded_imags, total_visibilities, total_image_pixels
    ):

        imag_visibilities = np.zeros(shape=(total_visibilities))

        for i in range(total_image_pixels):
            for j in range(total_visibilities):
                imag_visibilities[j] += image_1d[i] * preloaded_imags[i, j]

        return imag_visibilities

    @staticmethod
    @decorator_util.jit()
    def imag_visibilities_jit(
        image_1d, grid_radians, uv_wavelengths, total_visibilities, total_image_pixels
    ):

        imag_visibilities = np.zeros(shape=(total_visibilities))

        for i in range(total_image_pixels):
            for j in range(total_visibilities):
                imag_visibilities[j] += image_1d[i] * np.sin(
                    -2.0
                    * np.pi
                    * (
                        grid_radians[i, 1] * uv_wavelengths[j, 0]
                        - grid_radians[i, 0] * uv_wavelengths[j, 1]
                    )
                )

        return imag_visibilities

    def visibilities_from_image(self, image):

        real_visibilities = self.real_visibilities_from_image(image=image)
        imag_visibilities = self.imag_visibilities_from_image(image=image)

        return vis.Visibilities(
            visibilities_1d=np.stack((real_visibilities, imag_visibilities), axis=-1)
        )
