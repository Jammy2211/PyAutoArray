import numpy as np

from autoarray.inversion.inversion.dataset_interface import DatasetInterface
from autoarray.inversion.inversion.interferometer.mapping import (
    InversionInterferometerMapping,
)
from autoarray.inversion.inversion.settings import SettingsInversion
from autoarray.preloads import Preloads


class MockInversionInterferometer(InversionInterferometerMapping):
    def __init__(
        self,
        data=None,
        noise_map=None,
        transformer=None,
        linear_obj_list=None,
        operated_mapping_matrix=None,
        settings: SettingsInversion = SettingsInversion(),
        preloads: Preloads = Preloads(),
    ):
        dataset = DatasetInterface(
            data=data,
            noise_map=noise_map,
            transformer=transformer,
        )

        super().__init__(
            dataset=dataset,
            linear_obj_list=linear_obj_list,
            settings=settings,
            preloads=preloads,
        )

        self._operated_mapping_matrix = operated_mapping_matrix

    @property
    def operated_mapping_matrix(self) -> np.ndarray:
        if self._operated_mapping_matrix is None:
            return super().operated_mapping_matrix

        return self._operated_mapping_matrix
