import numpy as np
from typing import List, Union

from autoarray.inversion.linear_eqn.mapper.imaging import AbstractLEqMapper
from autoarray.structures.arrays.two_d.array_2d import Array2D


class LEqCombine:
    def __init__(self, leq_mapper: AbstractLEqMapper):

        self.leq_mapper = leq_mapper

    @property
    def noise_map(self):
        return self.leq_mapper.noise_map

    @property
    def mapper_list(self):
        return self.leq_mapper.mapper_list

    @property
    def total_mappers(self):
        return len(self.mapper_list)

    @property
    def mapping_matrix(self) -> np.ndarray:
        return self.leq_mapper.mapping_matrix

    @property
    def operated_mapping_matrix(self) -> np.ndarray:
        return self.leq_mapper.operated_mapping_matrix

    def data_vector_from(self, data: Array2D, preloads) -> np.ndarray:
        return self.leq_mapper.data_vector_from(data=data, preloads=preloads)

    @property
    def curvature_matrix(self):
        return self.leq_mapper.curvature_matrix

    def mapped_reconstructed_data_of_mappers_from(
        self, reconstruction: np.ndarray
    ) -> List[Array2D]:
        return self.leq_mapper.mapped_reconstructed_data_of_mappers_from(
            reconstruction=reconstruction
        )

    def mapped_reconstructed_image_of_mappers_from(
        self, reconstruction: np.ndarray
    ) -> List[Array2D]:
        return self.leq_mapper.mapped_reconstructed_image_of_mappers_from(
            reconstruction=reconstruction
        )
