from autoarray.inversion.linear_eqn.imaging.abstract import AbstractLEqImaging


class MockLEqImaging(AbstractLEqImaging):
    def __init__(
        self,
        noise_map=None,
        convolver=None,
        linear_obj_list=None,
        operated_mapping_matrix=None,
    ):

        super().__init__(
            noise_map=noise_map, convolver=convolver, linear_obj_list=linear_obj_list
        )

        self._operated_mapping_matrix = operated_mapping_matrix

    @property
    def operated_mapping_matrix(self):
        if self._operated_mapping_matrix is None:
            return super().operated_mapping_matrix

        return self._operated_mapping_matrix
