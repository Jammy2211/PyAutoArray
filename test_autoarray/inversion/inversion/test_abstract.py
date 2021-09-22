import numpy as np
from os import path
import pytest

import autoarray as aa

from autoarray.mock.mock import MockLinearEqn, MockInversion

from autoarray import exc


directory = path.dirname(path.realpath(__file__))


class TestAbstractLinearEqn:
    def test__errors_and_errors_with_covariance(self,):

        curvature_reg_matrix = np.array(
            [[1.0, 1.0, 1.0], [1.0, 2.0, 1.0], [1.0, 1.0, 3.0]]
        )

        inversion = MockLinearEqn(curvature_reg_matrix=curvature_reg_matrix)

        assert inversion.errors_with_covariance == pytest.approx(
            np.array([[2.5, -1.0, -0.5], [-1.0, 1.0, 0.0], [-0.5, 0.0, 0.5]]), 1.0e-2
        )
        assert inversion.errors == pytest.approx(np.array([2.5, 1.0, 0.5]), 1.0e-3)

    def test__preload_of_regularization_matrix__overwrites_calculation(self):

        inversion = MockLinearEqn(
            preloads=aa.Preloads(regularization_matrix=np.ones((2, 2)))
        )

        assert (inversion.regularization_matrix == np.ones((2, 2))).all()

    def test__reconstruction_raises_exception_for_linalg_error(self):

        linear_eqn = MockLinearEqn(curvature_reg_matrix=np.ones((3, 3)))

        with pytest.raises(exc.InversionException):

            # noinspection PyTypeChecker
            inversion = MockInversion(data_vector=np.ones(3), linear_eqn=linear_eqn)

            # noinspection PyStatementEffect
            inversion.reconstruction
