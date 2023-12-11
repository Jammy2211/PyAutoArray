import numpy as np
from typing import Dict, List, Optional, Type

from autoconf import cached_property

from autoarray.numba_util import profile_func

from autoarray.inversion.linear_obj.func_list import AbstractLinearObjFuncList
from autoarray.inversion.pixelization.mappers.abstract import AbstractMapper
from autoarray.inversion.inversion.abstract import AbstractInversion
from autoarray.inversion.linear_obj.linear_obj import LinearObj
from autoarray.inversion.inversion.settings import SettingsInversion
from autoarray.structures.arrays.uniform_2d import Array2D
from autoarray.operators.convolver import Convolver

from autoarray.inversion.inversion.imaging import inversion_imaging_util


class AbstractInversionImaging(AbstractInversion):
    def __init__(
        self,
        data: Array2D,
        noise_map: Array2D,
        convolver: Convolver,
        linear_obj_list: List[LinearObj],
        settings: SettingsInversion = SettingsInversion(),
        preloads=None,
        run_time_dict: Optional[Dict] = None,
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
        run_time_dict
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
            run_time_dict=run_time_dict,
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
            else self.linear_func_operated_mapping_matrix_dict[linear_obj]
            for linear_obj in self.linear_obj_list
        ]

    def _updated_cls_key_dict_from(self, cls: Type, preload_dict: Dict) -> Dict:
        cls_dict = {}

        for linear_func, values in zip(
            self.cls_list_from(cls=cls),
            preload_dict.values(),
        ):
            cls_dict[linear_func] = values

        return cls_dict

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
            return self._updated_cls_key_dict_from(
                cls=AbstractLinearObjFuncList,
                preload_dict=self.preloads.linear_func_operated_mapping_matrix_dict,
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

    @property
    def data_linear_func_matrix_dict(self):
        """
        Returns a matrix that for each data pixel, maps it to the sum of the values of a linear object function
        convolved with the PSF kernel at the data pixel.

        If a linear function in an inversion is fixed, its values can be evaluated and preloaded beforehand. For every
        data pixel, the PSF convolution with this preloaded linear function can also be preloaded, in a matrix of
        shape [data_pixels, 1].

        Given that multiple linear functions can be used and fixed in an inversion, this matrix is extended to have
        dimensions [data_pixels, total_fixed_linear_functions].

        When mapper objects and linear functions are used simultaneously in an inversion, this preloaded matrix
        significantly speed up the computation of their off-diagonal terms in the curvature matrix.

        This is similar to the preloading performed via the w-tilde formalism, except that there it is the PSF convolved
        values of each noise-map value pair that are preloaded.

        In **PyAutoGalaxy** and **PyAutoLens**, this preload is used when linear light profiles are fixed in the model.
        For example, when using a multi Gaussian expansion, the values defining how those Gaussians are evaluated
        (e.g. `centre`, `ell_comps` and `sigma`) are often fixed in a model, meaning this matrix can be preloaded and
        used for speed up.

        Returns
        -------
        ndarray
            A matrix of shape [data_pixels, total_fixed_linear_functions] that for each data pixel, maps it to the sum
            of the values of a linear object function convolved with the PSF kernel at the data pixel.
        """
        if self.preloads.data_linear_func_matrix_dict is not None:
            return self._updated_cls_key_dict_from(
                cls=AbstractLinearObjFuncList,
                preload_dict=self.preloads.data_linear_func_matrix_dict,
            )

        linear_func_list = self.cls_list_from(cls=AbstractLinearObjFuncList)

        data_linear_func_matrix_dict = {}

        for func_index, linear_func in enumerate(linear_func_list):
            curvature_weights = (
                self.linear_func_operated_mapping_matrix_dict[linear_func]
                / self.noise_map[:, None] ** 2
            )

            data_linear_func_matrix = (
                inversion_imaging_util.data_linear_func_matrix_from(
                    curvature_weights_matrix=curvature_weights,
                    image_frame_1d_lengths=self.convolver.image_frame_1d_lengths,
                    image_frame_1d_indexes=self.convolver.image_frame_1d_indexes,
                    image_frame_1d_kernels=self.convolver.image_frame_1d_kernels,
                )
            )

            data_linear_func_matrix_dict[linear_func] = data_linear_func_matrix

        return data_linear_func_matrix_dict

    @cached_property
    @profile_func
    def mapper_operated_mapping_matrix_dict(self) -> Dict:
        """
        The `operated_mapping_matrix` of a `Mapper` object describes the mappings between the observed data's values
        and the mapper's mesh pixels after a 2D convolution operation. It is described fully in the method
        `operated_mapping_matrix`.

        This property returns a dictionary mapping every mapper object to its corresponded operated mapping
        matrix, which is used for constructing the matrices that perform the linear inversion in an efficent way
        for the w_tilde calculation.

        Returns
        -------
        A dictionary mapping every mapper object to its operated mapping matrix.
        """

        if self.preloads.mapper_operated_mapping_matrix_dict is not None:
            return self._updated_cls_key_dict_from(
                cls=AbstractMapper,
                preload_dict=self.preloads.mapper_operated_mapping_matrix_dict,
            )

        mapper_operated_mapping_matrix_dict = {}

        for mapper in self.cls_list_from(cls=AbstractMapper):
            operated_mapping_matrix = self.convolver.convolve_mapping_matrix(
                mapping_matrix=mapper.mapping_matrix
            )

            mapper_operated_mapping_matrix_dict[mapper] = operated_mapping_matrix

        return mapper_operated_mapping_matrix_dict
