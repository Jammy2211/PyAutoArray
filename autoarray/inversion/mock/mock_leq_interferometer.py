from autoarray.inversion.linear_eqn.interferometer.mapping import (
    LEqInterferometerMapping,
)
from autoarray.inversion.inversion.settings import SettingsInversion


class MockLEqInterferometerMapping(LEqInterferometerMapping):
    def __init__(
        self,
        noise_map=None,
        transformer=None,
        linear_obj_list=None,
        transformed_mapping_matrix=None,
        settings: SettingsInversion = SettingsInversion(),
    ):

        super().__init__(
            noise_map=noise_map,
            transformer=transformer,
            linear_obj_list=linear_obj_list,
            settings=settings,
        )

        self._transformed_mapping_matrix = transformed_mapping_matrix

    @property
    def transformed_mapping_matrix(self):
        if self._transformed_mapping_matrix is None:
            return super().transformed_mapping_matrix

        return self._transformed_mapping_matrix
