import numpy as np
from typing import Dict, Optional, Union

from autoconf import cached_property
from autoarray.numba_util import profile_func

from autoarray.structures.arrays.two_d.array_2d import Array2D
from autoarray.operators.convolver import Convolver
from autoarray.inversion.linear_eqn.imaging import LinearEqnImagingMapping
from autoarray.inversion.linear_eqn.imaging import LinearEqnImagingWTilde
from autoarray.inversion.inversion.abstract import AbstractInversion
from autoarray.inversion.regularizations.abstract import AbstractRegularization
from autoarray.inversion.mappers.rectangular import MapperRectangular
from autoarray.inversion.mappers.voronoi import MapperVoronoi
from autoarray.preloads import Preloads
from autoarray.inversion.inversion.settings import SettingsInversion

from autoarray.inversion.inversion import inversion_util


def inversion_imaging_from(
    dataset,
    mapper: Union[MapperRectangular, MapperVoronoi],
    regularization: AbstractRegularization,
    settings: SettingsInversion = SettingsInversion(),
    preloads: Preloads = Preloads(),
    profiling_dict: Optional[Dict] = None,
):

    return inversion_imaging_unpacked_from(
        image=dataset.image,
        noise_map=dataset.noise_map,
        convolver=dataset.convolver,
        w_tilde=dataset.w_tilde,
        mapper=mapper,
        regularization=regularization,
        settings=settings,
        preloads=preloads,
        profiling_dict=profiling_dict,
    )


def inversion_imaging_unpacked_from(
    image: Array2D,
    noise_map: Array2D,
    convolver: Convolver,
    w_tilde,
    mapper: Union[MapperRectangular, MapperVoronoi],
    regularization: AbstractRegularization,
    settings: SettingsInversion = SettingsInversion(),
    preloads: Preloads = Preloads(),
    profiling_dict: Optional[Dict] = None,
):

    if preloads.use_w_tilde is not None:
        use_w_tilde = preloads.use_w_tilde
    else:
        use_w_tilde = settings.use_w_tilde

    if use_w_tilde:

        linear_eqn = LinearEqnImagingWTilde(
            noise_map=noise_map,
            convolver=convolver,
            w_tilde=w_tilde,
            mapper=mapper,
            regularization=regularization,
            preloads=preloads,
            profiling_dict=profiling_dict,
        )

    else:

        linear_eqn = LinearEqnImagingMapping(
            noise_map=noise_map,
            convolver=convolver,
            mapper=mapper,
            regularization=regularization,
            preloads=preloads,
            profiling_dict=profiling_dict,
        )

    return InversionImaging(
        data=image,
        linear_eqn=linear_eqn,
        settings=settings,
        profiling_dict=profiling_dict,
    )


class InversionImaging(AbstractInversion):
    def __init__(
        self,
        data: Union[Array2D],
        linear_eqn: Union[LinearEqnImagingWTilde, LinearEqnImagingMapping],
        settings: SettingsInversion = SettingsInversion(),
        profiling_dict: Optional[Dict] = None,
    ):

        super().__init__(
            data=data,
            linear_eqn=linear_eqn,
            settings=settings,
            profiling_dict=profiling_dict,
        )

    @property
    def image(self):
        return self.data

    @cached_property
    @profile_func
    def blurred_mapping_matrix(self) -> np.ndarray:
        """
        For a given pixelization pixel on the mapping matrix, we can use it to map it to a set of image-pixels in the
        image  plane. This therefore creates a 'image' of the source pixel (which corresponds to a set of values that
        mostly zeros, but with 1's where mappings occur).

        Before reconstructing the source, we blur every one of these source pixel images with the Point Spread Function
        of our  dataset via 2D convolution. This uses the methods
        in `Convolver.__init__` and `Convolver.convolve_mapping_matrix`:
        """
        return self.linear_eqn.blurred_mapping_matrix

    @property
    def residual_map(self):
        return inversion_util.inversion_residual_map_from(
            pixelization_values=self.reconstruction,
            data=self.image,
            slim_index_for_sub_slim_index=self.mapper_list[
                0
            ].source_grid_slim.mask.slim_index_for_sub_slim_index,
            all_sub_slim_indexes_for_pixelization_index=self.mapper_list[
                0
            ].all_sub_slim_indexes_for_pixelization_index,
        )

    @property
    def normalized_residual_map(self):
        return inversion_util.inversion_normalized_residual_map_from(
            pixelization_values=self.reconstruction,
            data=self.image,
            noise_map_1d=self.noise_map,
            slim_index_for_sub_slim_index=self.mapper_list[
                0
            ].source_grid_slim.mask.slim_index_for_sub_slim_index,
            all_sub_slim_indexes_for_pixelization_index=self.mapper_list[
                0
            ].all_sub_slim_indexes_for_pixelization_index,
        )

    @property
    def chi_squared_map(self):
        return inversion_util.inversion_chi_squared_map_from(
            pixelization_values=self.reconstruction,
            data=self.image,
            noise_map_1d=self.noise_map,
            slim_index_for_sub_slim_index=self.mapper_list[
                0
            ].source_grid_slim.mask.slim_index_for_sub_slim_index,
            all_sub_slim_indexes_for_pixelization_index=self.mapper_list[
                0
            ].all_sub_slim_indexes_for_pixelization_index,
        )
