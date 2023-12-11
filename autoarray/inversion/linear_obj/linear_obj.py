import numpy as np
from typing import Dict, Optional

from autoconf import cached_property

from autoarray.inversion.linear_obj.neighbors import Neighbors
from autoarray.inversion.regularization.abstract import AbstractRegularization

from autoarray.numba_util import profile_func


class LinearObj:
    def __init__(
        self,
        regularization: Optional[AbstractRegularization],
        run_time_dict: Optional[Dict] = None,
    ):
        """
        A linear object which reconstructs a dataset based on mapping between the data points of that dataset and
        the parameters of the linear object. For example, the linear obj could map to the data via analytic functions
        or discrete pixels on a mesh.

        The values of linear object are computed via a regularized linear matrix inversion, which infers a solution
        which best fits the data given its noise-map (by minimizing a chi-squared term) whilst accounting for smoothing
        due to regularizaiton.

        For example, in `PyAutoGalaxy` and `PyAutoLens` the light of galaxies is represented using `LightProfile`
        objects, which describe the surface brightness of a galaxy as a function. This function can either be assigned
        an overall intensity (e.g. the normalization) which describes how bright it is. Using a `LinearObj` the
        intensity can be solved for linearly instead.

        Parameters
        ----------
        regularization
            The regularization scheme which may be applied to this linear object in order to smooth its solution.
        run_time_dict
            A dictionary which contains timing of certain functions calls which is used for profiling.
        """
        self.regularization = regularization
        self.run_time_dict = run_time_dict

    @property
    def params(self) -> int:
        """
        The total number of parameters used to reconstruct the data.

        For example for the following linear objects:

        - `AbstractLinearObjFuncList` the number of analytic functions.
        - `Mapper` the number of piels in the mesh used to reconstruct the data.

        Returns
        -------
        The number of parameters used to reconstruct the data.
        """
        raise NotImplementedError

    @property
    def neighbors(self) -> Neighbors:
        """
        An object describing how the different parameters in the linear object neighbor one another, which is used
        to apply smoothing to neighboring parameters via regularization.

        For example for the following linear objects:

        - `AbstractLinearObjFuncList` whether certain analytic functions reconstruct nearby components next to
          one another.

        - `Mapper` how the pixels on the mesh used to reconstruct the data neighbor one another.

        Returns
        -------
        An object describing how the parameters of the linear object neighbor one another.
        """
        raise NotImplementedError

    @cached_property
    @profile_func
    def unique_mappings(self):
        """
        An object describing the unique mappings between data points / pixels in the data and the parameters of the
        linear object.

        For example for the following linear objects:

        - `AbstractLinearObjFuncList` All pixels in the data map to every analytic function, therefore the unique
          mappings are one-to-one with each function.

        - `Mapper` Every group of sub-pixels map to a unique mesh pixel and the unique mappings describe each of these
          unique group mappings.

        This object is used to speed up the computation of certain matrices for inversions using the w-tilde formalism.

        Returns
        -------
        An object describing the unique mappings between data points / pixels in the data and the parameters of the
        linear object.
        """
        raise NotImplementedError

    @property
    def mapping_matrix(self) -> np.ndarray:
        """
        The `mapping_matrix` of a linear object describes the mappings between the observed data's data-points / pixels
        and the linear object parameters. It is used to construct the simultaneous linear equations which reconstruct
        the data.

        The matrix has shape [total_data_points, data_linear_object_parameters], whereby all non-zero entries
        indicate that a data point maps to a linear object parameter.

        For `Mapper` linear objects it is described in the following paper as
        matrix `f` https://arxiv.org/pdf/astro-ph/0302587.pdf and in more detail in the
        function `mapper_util.mapping_matrix_from()`.

        If there are multiple linear objects, the mapping matrices are stacked such that their simultaneous linear
        equations are solved simultaneously. This property returns the stacked mapping matrix.
        """
        raise NotImplementedError

    def pixel_signals_from(self, signal_scale) -> np.ndarray:
        raise NotImplementedError

    @property
    def operated_mapping_matrix_override(self) -> Optional[np.ndarray]:
        """
        An `Inversion` takes the `mapping_matrix` of each linear object and combines it with the data's operators
        (e.g. a `Convolver` for `Imaging` data) to compute the `operated_mapping_matrix`.

        If this property is overwritten this operation is not performed, with the `operated_mapping_matrix` output
        by this property automatically used instead.

        This is used for linear objects where properly performing the operator (e.g. 2D convolution) within only
        the scope of the `Inversion` object is not possible. For example, images may have flux outside the masked
        region which is blurred into the masked region which is linear solved for. This flux is outside the region
        that defines the `mapping_matrix` and thus this override is required to properly incorporate it.

        Returns
        -------
        An operated mapping matrix of dimensions (total_mask_pixels, total_parameters) which overrides the mapping
        matrix calculations performed in the linear equation solvers.
        """
        return None

    @property
    def regularization_matrix(self) -> np.ndarray:
        """
        The regularization matrix H is used to impose smoothness on our inversion's reconstruction. This enters the
        linear algebra system we solve for using D and F above and is given by
        equation (12) in https://arxiv.org/pdf/astro-ph/0302587.pdf.

        A complete description of regularization is given in the `regularization.py` and `regularization_util.py`
        modules.

        For multiple mappers, the regularization matrix is computed as the block diagonal of each individual mapper.
        The scipy function `block_diag` has an overhead associated with it and if there is only one mapper and
        regularization it is bypassed.
        """
        if self.regularization is None:
            return np.zeros((self.params, self.params))

        return self.regularization.regularization_matrix_from(linear_obj=self)
