import numpy as np
from typing import Dict, Optional, Union

from autoconf import cached_property
from autoarray.numba_util import profile_func

from autoarray.structures.visibilities import VisibilitiesNoiseMap
from autoarray.preloads import Preloads
from autoarray.structures.arrays.two_d.array_2d import Array2D
from autoarray.structures.grids.two_d.grid_2d_irregular import Grid2DIrregular
from autoarray.inversion.mappers.rectangular import MapperRectangular
from autoarray.inversion.mappers.voronoi import MapperVoronoi

from autoarray.inversion.inversion import inversion_util


class AbstractLinearEqn:
    def __init__(
        self,
        noise_map: Union[Array2D, VisibilitiesNoiseMap],
        mapper: Union[MapperRectangular, MapperVoronoi],
        preloads: Preloads = Preloads(),
        profiling_dict: Optional[Dict] = None,
    ):

        self.noise_map = noise_map
        self.mapper = mapper

        self.preloads = preloads

        self.profiling_dict = profiling_dict

    @property
    def operated_mapping_matrix(self) -> np.ndarray:
        raise NotImplementedError

    @profile_func
    def data_vector_from(self, data):
        raise NotImplementedError

    @property
    def curvature_matrix(self) -> np.ndarray:
        raise NotImplementedError

    @profile_func
    def mapped_reconstructed_image_from(self, reconstruction) -> Array2D:
        raise NotImplementedError

    def residual_map_from(
        self, data: np.ndarray, reconstruction: np.ndarray
    ) -> np.ndarray:
        return inversion_util.residual_map_from(
            reconstruction=reconstruction,
            data=data,
            slim_index_for_sub_slim_index=self.mapper.source_grid_slim.mask.slim_index_for_sub_slim_index,
            all_sub_slim_indexes_for_pixelization_index=self.mapper.all_sub_slim_indexes_for_pixelization_index,
        )

    def normalized_residual_map_from(
        self, data: np.ndarray, reconstruction: np.ndarray
    ) -> np.ndarray:
        return inversion_util.inversion_normalized_residual_map_from(
            reconstruction=reconstruction,
            data=data,
            noise_map_1d=self.noise_map,
            slim_index_for_sub_slim_index=self.mapper.source_grid_slim.mask.slim_index_for_sub_slim_index,
            all_sub_slim_indexes_for_pixelization_index=self.mapper.all_sub_slim_indexes_for_pixelization_index,
        )

    def chi_squared_map_from(
        self, data: np.ndarray, reconstruction: np.ndarray
    ) -> np.ndarray:
        return inversion_util.inversion_chi_squared_map_from(
            data=data,
            reconstruction=reconstruction,
            noise_map_1d=self.noise_map,
            slim_index_for_sub_slim_index=self.mapper.source_grid_slim.mask.slim_index_for_sub_slim_index,
            all_sub_slim_indexes_for_pixelization_index=self.mapper.all_sub_slim_indexes_for_pixelization_index,
        )

    def brightest_reconstruction_pixel_from(self, reconstruction: np.ndarray):
        return np.argmax(reconstruction)

    def brightest_reconstruction_pixel_centre_from(self, reconstruction: np.ndarray):

        brightest_reconstruction_pixel = self.brightest_reconstruction_pixel_from(
            reconstruction=reconstruction
        )

        return Grid2DIrregular(
            grid=[self.mapper.source_pixelization_grid[brightest_reconstruction_pixel]]
        )
