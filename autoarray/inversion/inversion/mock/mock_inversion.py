import numpy as np
from typing import List

from autoarray.inversion.inversion.matrices import InversionMatrices
from autoarray.inversion.inversion.settings import SettingsInversion
from autoarray.preloads import Preloads

from autoarray.inversion.linear_eqn.mock.mock_leq import MockLEq
from autoarray.inversion.regularization.mock.mock_regularization import (
    MockRegularization,
)


class MockInversion(InversionMatrices):
    def __init__(
        self,
        data=None,
        leq: MockLEq = None,
        regularization_list: List[MockRegularization] = None,
        data_vector=None,
        regularization_matrix=None,
        curvature_reg_matrix=None,
        reconstruction: np.ndarray = None,
        reconstruction_dict: List[np.ndarray] = None,
        errors: np.ndarray = None,
        errors_dict: List[np.ndarray] = None,
        regularization_term=None,
        log_det_curvature_reg_matrix_term=None,
        log_det_regularization_matrix_term=None,
        curvature_matrix_preload=None,
        curvature_matrix_counts=None,
        settings: SettingsInversion = SettingsInversion(),
        preloads: Preloads = Preloads(),
    ):

        # self.__dict__["curvature_matrix"] = curvature_matrix
        # self.__dict__["curvature_reg_matrix_cholesky"] = curvature_reg_matrix_cholesky
        # self.__dict__["regularization_matrix"] = regularization_matrix
        # self.__dict__["curvature_reg_matrix"] = curvature_reg_matrix
        # self.__dict__["reconstruction"] = reconstruction
        # self.__dict__["mapped_reconstructed_image"] = mapped_reconstructed_image

        super().__init__(
            data=data,
            leq=leq,
            regularization_list=regularization_list,
            settings=settings,
            preloads=preloads,
        )

        self._data_vector = data_vector
        self._regularization_matrix = regularization_matrix
        self._curvature_reg_matrix = curvature_reg_matrix
        self._reconstruction = reconstruction
        self._reconstruction_dict = reconstruction_dict

        self._errors = errors
        self._errors_dict = errors_dict

        self._regularization_term = regularization_term
        self._log_det_curvature_reg_matrix_term = log_det_curvature_reg_matrix_term
        self._log_det_regularization_matrix_term = log_det_regularization_matrix_term

        self._curvature_matrix_preload = curvature_matrix_preload
        self._curvature_matrix_counts = curvature_matrix_counts

    @property
    def data_vector(self) -> np.ndarray:
        if self._data_vector is None:
            return super().data_vector
        return self._data_vector

    @property
    def regularization_matrix(self):

        if self._regularization_matrix is None:
            return super().regularization_matrix

        return self._regularization_matrix

    @property
    def curvature_reg_matrix(self):
        return self._curvature_reg_matrix

    @property
    def reconstruction(self):

        if self._reconstruction is None:
            return super().reconstruction
        return self._reconstruction

    @property
    def reconstruction_dict(self):

        if self._reconstruction_dict is None:
            return super().reconstruction_dict
        return self._reconstruction_dict

    @property
    def errors(self):

        if self._errors is None:
            return super().errors
        return self._errors

    @property
    def errors_dict(self):

        if self._errors_dict is None:
            return super().errors_dict
        return self._errors_dict

    @property
    def regularization_term(self):

        if self._regularization_term is None:
            return super().regularization_term

        return self._regularization_term

    @property
    def log_det_curvature_reg_matrix_term(self):

        if self._log_det_curvature_reg_matrix_term is None:
            return super().log_det_curvature_reg_matrix_term

        return self._log_det_curvature_reg_matrix_term

    @property
    def log_det_regularization_matrix_term(self):

        if self._log_det_regularization_matrix_term is None:
            return super().log_det_regularization_matrix_term

        return self._log_det_regularization_matrix_term

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
