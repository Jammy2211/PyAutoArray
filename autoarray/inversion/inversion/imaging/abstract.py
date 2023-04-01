import numpy as np
from typing import Dict, List, Optional

from autoconf import cached_property

from autoarray.numba_util import profile_func

from autoarray.inversion.linear_obj.func_list import AbstractLinearObjFuncList
from autoarray.inversion.inversion.abstract import AbstractInversion
from autoarray.inversion.linear_obj.linear_obj import LinearObj
from autoarray.inversion.inversion.settings import SettingsInversion
from autoarray.structures.arrays.uniform_2d import Array2D
from autoarray.operators.convolver import Convolver


class AbstractInversionImaging(AbstractInversion):
    def __init__(
        self,
        data: Array2D,
        noise_map: Array2D,
        convolver: Convolver,
        linear_obj_list: List[LinearObj],
        settings: SettingsInversion = SettingsInversion(),
        preloads=None,
        profiling_dict: Optional[Dict] = None,
    ):
        """
        An `Inversion` reconstructs an input dataset using a list of linear objects (e.g. a list of analytic functions
        or a pixelized grid).

        The inversion constructs simultaneous linear equations (via vectors and matrices) which allow for the values
        of the linear object parameters that best reconstruct the dataset to be solved, via linear matrix algebra.

        This object contains matrices and vectors which perform an inversion for fits to an `Imaging` dataset. This
        includes operations which use a PSF / `Convolver` in order to incorporate blurring into the solved for
        linear object pixels.

        The inversion may be regularized, whereby the parameters of the linear objects used to reconstruct the data
        are smoothed with one another such that their solved for values conform to certain properties (e.g. smoothness
        based regularization requires that parameters in the linear objects which neighbor one another have similar
        values).

        This object contains properties which compute all of the different matrices necessary to perform the inversion.

        The linear algebra required to perform an `Inversion` depends on the type of dataset being fitted (e.g.
        `Imaging`, `Interferometer) and the formalism chosen (e.g. a using a `mapping_matrix` or the
        w_tilde formalism). The children of this class overwrite certain methods in order to be appropriate for
        certain datasets or use a specific formalism.

        Inversions use the formalism's outlined in the following Astronomy papers:

        https://arxiv.org/pdf/astro-ph/0302587.pdf
        https://arxiv.org/abs/1708.07377
        https://arxiv.org/abs/astro-ph/0601493

        Parameters
        ----------
        data
            The data of the dataset (e.g. the `image` of `Imaging` data) which may have been changed.
        noise_map
            The noise_map of the noise_mapset (e.g. the `noise_map` of `Imaging` noise_map) which may have been changed.
        linear_obj_list
            The list of linear objects (e.g. analytic functions, a mapper with a pixelized grid) which reconstruct the
            input dataset's data and whose values are solved for via the inversion.
        settings
            Settings controlling how an inversion is fitted for example which linear algebra formalism is used.
        preloads
            Preloads in memory certain arrays which may be known beforehand in order to speed up the calculation,
            for example certain matrices used by the linear algebra could be preloaded.
        profiling_dict
            A dictionary which contains timing of certain functions calls which is used for profiling.
        """

        from autoarray.preloads import Preloads

        preloads = preloads or Preloads()

        self.convolver = convolver

        super().__init__(
            data=data,
            noise_map=noise_map,
            linear_obj_list=linear_obj_list,
            settings=settings,
            preloads=preloads,
            profiling_dict=profiling_dict,
        )

    @property
    def operated_mapping_matrix_list(self) -> List[np.ndarray]:
        """
        The `operated_mapping_matrix` of a linear object describes the mappings between the observed data's values and
        the linear objects model, including a 2D convolution operation.

        This is used to construct the simultaneous linear equations which reconstruct the data.

        This property returns the a list of each linear object's blurred mapping matrix, which is computed by
        blurring each linear object's `mapping_matrix` property with the `Convolver` operator.

        A linear object may have a `operated_mapping_matrix_override` property, which bypasses  the `mapping_matrix`
        computation and convolution operator and is directly placed in the `operated_mapping_matrix_list`.
        """

        return [
            self.convolver.convolve_mapping_matrix(
                mapping_matrix=linear_obj.mapping_matrix
            )
            if linear_obj.operated_mapping_matrix_override is None
            else linear_obj.operated_mapping_matrix_override
            for linear_obj in self.linear_obj_list
        ]

    def _linear_func_preload_dict_map(self, linear_func_preload_dict: Dict) -> Dict:

        linear_func_dict = {}

        for linear_func, values in zip(
            self.cls_list_from(cls=AbstractLinearObjFuncList),
            linear_func_preload_dict.values(),
        ):
            linear_func_dict[linear_func] = values

        return linear_func_dict

    @cached_property
    @profile_func
    def linear_func_operated_mapping_matrix_dict(self) -> Dict:
        """
        The `operated_mapping_matrix` of a linear object describes the mappings between the observed data's values and
        the linear objects model, including a 2D convolution operation. It is described fully in the method
        `operated_mapping_matrix`.

        This property returns a dictionary mapping every linear func object to its corresponded operated mapping
        matrix, which is used for constructing the matrices that perform the linear inversion in an efficent way
        for the w_tilde calculation.

        Returns
        -------
        A dictionary mapping every linear function object to its operated mapping matrix.
        """

        if self.preloads.linear_func_operated_mapping_matrix_dict is not None:
            return self._linear_func_preload_dict_map(
                linear_func_preload_dict=self.preloads.linear_func_operated_mapping_matrix_dict
            )

        linear_func_operated_mapping_matrix_dict = {}

        for linear_func in self.cls_list_from(cls=AbstractLinearObjFuncList):

            if linear_func.operated_mapping_matrix_override is not None:
                operated_mapping_matrix = linear_func.operated_mapping_matrix_override
            else:
                operated_mapping_matrix = self.convolver.convolve_mapping_matrix(
                    mapping_matrix=linear_func.mapping_matrix
                )

            linear_func_operated_mapping_matrix_dict[
                linear_func
            ] = operated_mapping_matrix

        return linear_func_operated_mapping_matrix_dict
