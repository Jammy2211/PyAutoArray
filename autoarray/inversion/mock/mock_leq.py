import numpy as np
from typing import List, Union

from autoarray.inversion.inversion.settings import SettingsInversion
from autoarray.inversion.linear_eqn.abstract import AbstractLEq

from autoarray.inversion.mock.mock_linear_obj_func import MockLinearObjFunc
from autoarray.inversion.mock.mock_mapper import MockMapper


class MockLEq(AbstractLEq):
    def __init__(
        self,
        noise_map=None,
        linear_obj_list: List[Union[MockLinearObjFunc, MockMapper]] = None,
        operated_mapping_matrix=None,
        data_vector=None,
        curvature_matrix=None,
        mapped_reconstructed_data_dict=None,
        mapped_reconstructed_image_dict=None,
        settings: SettingsInversion = SettingsInversion(),
    ):

        super().__init__(
            noise_map=noise_map, linear_obj_list=linear_obj_list, settings=settings
        )

        self._operated_mapping_matrix = operated_mapping_matrix
        self._data_vector = data_vector
        self._curvature_matrix = curvature_matrix
        self._mapped_reconstructed_data_dict = mapped_reconstructed_data_dict
        self._mapped_reconstructed_image_dict = mapped_reconstructed_image_dict

    @property
    def operated_mapping_matrix(self) -> np.ndarray:
        if self._operated_mapping_matrix is None:
            return super().operated_mapping_matrix

        return self._operated_mapping_matrix

    def data_vector_from(self, data, preloads=None) -> np.ndarray:
        if self._data_vector is None:
            return super().data_vector_from(data=data, preloads=preloads)

        return self._data_vector

    @property
    def curvature_matrix_diag(self):
        return self._curvature_matrix

    def mapped_reconstructed_data_dict_from(self, reconstruction: np.ndarray):
        """
        Using the reconstructed source pixel fluxes we map each source pixel flux back to the image plane and
        reconstruct the image data.

        This uses the unique mappings of every source pixel to image pixels, which is a quantity that is already
        computed when using the w-tilde formalism.

        Returns
        -------
        Array2D
            The reconstructed image data which the inversion fits.
        """

        if self._mapped_reconstructed_data_dict is None:
            return super().mapped_reconstructed_data_dict_from(
                reconstruction=reconstruction
            )

        return self._mapped_reconstructed_data_dict

    def mapped_reconstructed_image_dict_from(self, reconstruction: np.ndarray):
        """
        Using the reconstructed source pixel fluxes we map each source pixel flux back to the image plane and
        reconstruct the image image.

        This uses the unique mappings of every source pixel to image pixels, which is a quantity that is already
        computed when using the w-tilde formalism.

        Returns
        -------
        Array2D
            The reconstructed image image which the inversion fits.
        """

        if self._mapped_reconstructed_image_dict is None:
            return super().mapped_reconstructed_image_dict_from(
                reconstruction=reconstruction
            )

        return self._mapped_reconstructed_image_dict
