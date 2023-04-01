import numpy as np
from typing import Dict

from autoarray.inversion.inversion.imaging.mapping import InversionImagingMapping
from autoarray.inversion.inversion.imaging.w_tilde import InversionImagingWTilde
from autoarray.inversion.inversion.settings import SettingsInversion
from autoarray.preloads import Preloads


class MockInversionImaging(InversionImagingMapping):
    def __init__(
        self,
        data=None,
        noise_map=None,
        convolver=None,
        linear_obj_list=None,
        operated_mapping_matrix=None,
        linear_func_operated_mapping_matrix_dict=None,
        data_linear_func_matrix_dict=None,
        curvature_matrix_preload=None,
        curvature_matrix_counts=None,
        settings: SettingsInversion = SettingsInversion(),
        preloads: Preloads = Preloads(),
    ):

        super().__init__(
            data=data,
            noise_map=noise_map,
            convolver=convolver,
            linear_obj_list=linear_obj_list,
            settings=settings,
            preloads=preloads,
        )

        self._operated_mapping_matrix = operated_mapping_matrix

        self._linear_func_operated_mapping_matrix_dict = (
            linear_func_operated_mapping_matrix_dict
        )
        self._data_linear_func_matrix_dict = data_linear_func_matrix_dict
        self._curvature_matrix_preload = curvature_matrix_preload
        self._curvature_matrix_counts = curvature_matrix_counts

    @property
    def operated_mapping_matrix(self) -> np.ndarray:
        if self._operated_mapping_matrix is None:
            return super().operated_mapping_matrix

        return self._operated_mapping_matrix

    @property
    def linear_func_operated_mapping_matrix_dict(self) -> Dict:
        if self._linear_func_operated_mapping_matrix_dict is None:
            return super().linear_func_operated_mapping_matrix_dict

        return self._linear_func_operated_mapping_matrix_dict

    @property
    def data_linear_func_matrix_dict(self) -> Dict:
        if self._data_linear_func_matrix_dict is None:
            return super().data_linear_func_matrix_dict

        return self._data_linear_func_matrix_dict

    @property
    def curvature_matrix_preload(self):
        if self._curvature_matrix_preload is None:
            return super().curvature_matrix_preload

        return self._curvature_matrix_preload

    @property
    def curvature_matrix_counts(self):
        if self._curvature_matrix_counts is None:
            return super().curvature_matrix_counts

        return self._curvature_matrix_counts


class MockWTildeImaging:
    def check_noise_map(self, noise_map):

        pass


class MockInversionImagingWTilde(InversionImagingWTilde):
    def __init__(
        self,
        data=None,
        noise_map=None,
        convolver=None,
        w_tilde=None,
        linear_obj_list=None,
        curvature_matrix_mapper_diag=None,
        settings: SettingsInversion = SettingsInversion(),
        preloads: Preloads = Preloads(),
    ):

        super().__init__(
            data=data,
            noise_map=noise_map,
            convolver=convolver,
            w_tilde=w_tilde or MockWTildeImaging(),
            linear_obj_list=linear_obj_list,
            settings=settings,
            preloads=preloads,
        )

        self.__curvature_matrix_mapper_diag = curvature_matrix_mapper_diag

    @property
    def curvature_matrix_mapper_diag(self):
        if self.__curvature_matrix_mapper_diag is None:
            return super()._curvature_matrix_mapper_diag

        return self.__curvature_matrix_mapper_diag
