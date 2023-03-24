from __future__ import annotations
import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from autoarray.inversion.linear_obj.linear_obj import LinearObj

from autoarray.inversion.regularization.abstract import AbstractRegularization

from autoarray.inversion.regularization import regularization_util


class BrightnessZeroth(AbstractRegularization):
    def __init__(
        self,
        coefficient: float = 1.0,
        signal_scale: float = 1.0,
    ):
        """
        An adaptive regularization scheme which applies zeroth order regularization to pixels with low expected
        signal values.

        For the weighted regularization scheme, each pixel is given an 'effective regularization weight', which is
        controls the degree of zeroth order regularization applied to each pixel. The motivation of this is that
        the exterior regions different regions of a pixelization's mesh ought to have a signal consistent with zero,
        but may have a low level of non-zero signal when fitting the data.

        To implement this regularization, values on the diagonal of the regularization matrix are increased
        according to the regularization weight_list of each pixel.

        Parameters
        ----------
        coefficient
            The regularization coefficient which controls the degree of zeroth order regularizaiton applied to
            the inversion reconstruction, in regions of low signal.
        signal_scale
            A factor which controls how rapidly the smoothness of regularization varies from high signal regions to
            low signal regions.
        """

        super().__init__()

        self.coefficient = coefficient
        self.signal_scale = signal_scale

    def regularization_weights_from(self, linear_obj: LinearObj) -> np.ndarray:
        """
        Returns the regularization weights of the ``BrightnessZeroth`` regularization scheme.

        The weights define the level of zeroth order regularization applied to every mesh parameter (typically pixels
        of a ``Mapper``).

        They are computed using an estimate of the expected signal in each pixel.

        Parameters
        ----------
        linear_obj
            The linear object (e.g. a ``Mapper``) which uses these weights when performing regularization.

        Returns
        -------
        The regularization weights.
        """
        pixel_signals = linear_obj.pixel_signals_from(signal_scale=self.signal_scale)

        return regularization_util.brightness_zeroth_regularization_weights_from(
            coefficient=self.coefficient, pixel_signals=pixel_signals
        )

    def regularization_matrix_from(self, linear_obj: LinearObj) -> np.ndarray:
        """
        Returns the regularization matrix of this regularization scheme.

        Parameters
        ----------
        linear_obj
            The linear object (e.g. a ``Mapper``) which uses this matrix to perform regularization.

        Returns
        -------
        The regularization matrix.
        """
        regularization_weights = self.regularization_weights_from(linear_obj=linear_obj)

        return regularization_util.brightness_zeroth_regularization_matrix_from(
            regularization_weights=regularization_weights
        )
