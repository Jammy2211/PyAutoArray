import numpy as np
from scipy.linalg import block_diag
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import splu
from typing import Dict, List, Optional, Tuple, Type, Union

from autoconf import cached_property
from autoarray.numba_util import profile_func

from autoarray.inversion.linear_obj.func_list import LinearObj
from autoarray.inversion.pixelization.mappers.abstract import AbstractMapper
from autoarray.inversion.regularization.abstract import AbstractRegularization
from autoarray.inversion.inversion.settings import SettingsInversion
from autoarray.structures.arrays.uniform_2d import Array2D
from autoarray.structures.grids.irregular_2d import Grid2DIrregular
from autoarray.structures.visibilities import Visibilities

from autoarray import exc
from autoarray.util import misc_util
from autoarray.inversion.inversion import inversion_util


class AbstractInversion:
    def __init__(
        self,
        data: Union[Visibilities, Array2D],
        noise_map: Union[Visibilities, Array2D],
        linear_obj_list: List[LinearObj],
        settings: SettingsInversion = SettingsInversion(),
        preloads: Optional["Preloads"] = None,
        profiling_dict: Optional[Dict] = None,
    ):
        """
        An inversion uses linear matrix
        Parameters
        ----------
        data
        inversion
        settings
        preloads
        profiling_dict
        """

        from autoarray.preloads import Preloads

        preloads = preloads or Preloads()

        try:
            import numba
        except ModuleNotFoundError:
            raise exc.InversionException(
                "Inversion functionality (linear light profiles, pixelized reconstructions) is "
                "disabled if numba is not installed.\n\n"
                "This is because the run-times without numba are too slow.\n\n"
                "Please install numba, which is described at the following web page:\n\n"
                "https://pyautolens.readthedocs.io/en/latest/installation/overview.html"
            )

        self.data = data
        self.noise_map = noise_map

        self.linear_obj_list = linear_obj_list

        self.settings = settings

        self.preloads = preloads
        self.profiling_dict = profiling_dict

    def has(self, cls: Type) -> bool:
        """
        Does this `Inversion` have an attribute which is of type `cls`?

        Parameters
        ----------
        dict_values
            A class dictionary of values which instances of the input `cls` are checked to see if it has an instance
            of that class.
        cls
            The type of class that is checked if the object has an instance of.
        """
        return misc_util.has(
            values=self.linear_obj_list + self.regularization_list, cls=cls
        )

    def total(self, cls: Type) -> int:
        """
        Returns the total number of attribute in the `Inversion` which are of type `cls`?

        Parameters
        ----------
        dict_values
            A class dictionary of values which instances of the input `cls` are checked to see how many instances
            of that class it contains.
        cls
            The type of class that is checked if the object has an instance of.
        """
        return misc_util.total(
            values=self.linear_obj_list + self.regularization_list, cls=cls
        )

    def cls_list_from(self, cls: Type, cls_filtered: Optional[Type] = None) -> List:
        """
        Returns a list of objects in the `Inversion` which are an instance of the input `cls`.

        The optional `cls_filtered` input removes classes of an input instance type.

        For example:

        - If the input is `cls=aa.mesh.Mesh`, a list containing all pixelizations in the class are returned.

        - If `cls=aa.mesh.Mesh` and `cls_filtered=aa.mesh.Rectangular`, a list of all pixelizations
        excluding those which are `Rectangular` pixelizations will be returned.

        Parameters
        ----------
        cls
            The type of class that a list of instances of this class in the galaxy are returned for.
        cls_filtered
            A class type which is filtered and removed from the class list.
        """
        return misc_util.cls_list_from(
            values=self.linear_obj_list + self.regularization_list,
            cls=cls,
            cls_filtered=cls_filtered,
        )

    @property
    def regularization_list(self) -> List[AbstractRegularization]:
        return [linear_obj.regularization for linear_obj in self.linear_obj_list]

    @property
    def all_linear_obj_have_regularization(self) -> bool:
        return len(self.linear_obj_list) == len(
            list(filter(None, self.regularization_list))
        )

    @property
    def total_regularizations(self) -> int:
        return sum(
            regularization is not None for regularization in self.regularization_list
        )

    @property
    def no_regularization_index_list(self) -> List[int]:

        # TODO : Needs to be range based on pixels.

        no_regularization_index_list = []

        pixel_count = 0

        for linear_obj, regularization in zip(
            self.linear_obj_list, self.regularization_list
        ):

            if regularization is None:

                for pixel in range(pixel_count, pixel_count + linear_obj.parameters):

                    no_regularization_index_list.append(pixel)

            pixel_count += linear_obj.parameters

        return no_regularization_index_list

    @property
    def mask(self) -> Array2D:
        return self.data.mask

    @cached_property
    @profile_func
    def mapping_matrix(self) -> np.ndarray:
        """
        The `mapping_matrix` of a linear object describes the mappings between the observed data's values and the
        linear objects model. These are used to construct the simultaneous linear equations which reconstruct the data.

        If there are multiple linear objects, the mapping matrices are stacked such that their simultaneous linear
        equations are solved simultaneously. This property returns the stacked mapping matrix.
        """
        return np.hstack(
            [linear_obj.mapping_matrix for linear_obj in self.linear_obj_list]
        )

    @property
    def operated_mapping_matrix_list(self) -> np.ndarray:
        raise NotImplementedError

    @cached_property
    @profile_func
    def operated_mapping_matrix(self) -> np.ndarray:
        """
        The `operated_mapping_matrix` of a linear object describes the mappings between the observed data's values and
        the linear objects model, including a 2D convolution operation.

        This is used to construct the simultaneous linear equations which reconstruct the data.

        If there are multiple linear objects, the blurred mapping matrices are stacked such that their simultaneous
        linear equations are solved simultaneously.
        """

        if self.preloads.operated_mapping_matrix is not None:
            return self.preloads.operated_mapping_matrix

        return np.hstack(self.operated_mapping_matrix_list)

    @cached_property
    @profile_func
    def data_vector(self) -> np.ndarray:
        raise NotImplementedError

    @cached_property
    @profile_func
    def curvature_matrix(self) -> np.ndarray:
        raise NotImplementedError

    @cached_property
    @profile_func
    def regularization_matrix(self) -> Optional[np.ndarray]:
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
        if self.preloads.regularization_matrix is not None:
            return self.preloads.regularization_matrix

        return block_diag(
            *[linear_obj.regularization_matrix for linear_obj in self.linear_obj_list]
        )

    @cached_property
    @profile_func
    def regularization_matrix_reduced(self) -> Optional[np.ndarray]:
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

        regularization_matrix = self.regularization_matrix

        if self.all_linear_obj_have_regularization:
            return regularization_matrix

        regularization_matrix = np.delete(
            regularization_matrix, self.no_regularization_index_list, 0
        )
        regularization_matrix = np.delete(
            regularization_matrix, self.no_regularization_index_list, 1
        )

        return regularization_matrix

    @cached_property
    @profile_func
    def curvature_reg_matrix(self) -> np.ndarray:
        """
        The linear system of equations solves for F + regularization_coefficient*H, which is computed below.

        For a single mapper, this function overwrites the cached `curvature_matrix`, because for large matrices this
        avoids overheads in memory allocation. The `curvature_matrix` is removed as a cached property as a result,
        to ensure if we access it after computing the `curvature_reg_matrix` it is correctly recalculated in a new
        array of memory.
        """
        if not self.has(cls=AbstractRegularization):
            return self.curvature_matrix

        # TODO : Done for speed, instead check if there is one regularization

        if len(self.regularization_list) == 1:

            curvature_matrix = self.curvature_matrix
            curvature_matrix += self.regularization_matrix

            del self.__dict__["curvature_matrix"]

            return curvature_matrix

        return np.add(self.curvature_matrix, self.regularization_matrix)

    @cached_property
    @profile_func
    def curvature_reg_matrix_reduced(self) -> np.ndarray:
        """
        The linear system of equations solves for F + regularization_coefficient*H, which is computed below.

        This is the curvature reg matrix for only the mappers, which is necessary for computing the log det
        term without the linear light profiles included.
        """
        if self.all_linear_obj_have_regularization:
            return self.curvature_reg_matrix

        curvature_reg_matrix = self.curvature_reg_matrix

        curvature_reg_matrix = np.delete(
            curvature_reg_matrix, self.no_regularization_index_list, 0
        )
        curvature_reg_matrix = np.delete(
            curvature_reg_matrix, self.no_regularization_index_list, 1
        )

        return curvature_reg_matrix

    @cached_property
    @profile_func
    def curvature_reg_matrix_cholesky(self) -> np.ndarray:
        """
        Performs a Cholesky decomposition of the `curvature_reg_matrix`, the result of which is used to solve the
        linear system of equations of the `Inversion`.

        The method `np.linalg.solve` is faster to do this, but the Cholesky decomposition is used later in the code
        to speed up the calculation of `log_det_curvature_reg_matrix_term`.
        """
        try:
            return np.linalg.cholesky(self.curvature_reg_matrix)
        except np.linalg.LinAlgError:
            raise exc.InversionException()

    @cached_property
    @profile_func
    def curvature_reg_matrix_reduced_cholesky(self) -> np.ndarray:
        """
        Performs a Cholesky decomposition of the `curvature_reg_matrix`, the result of which is used to solve the
        linear system of equations of the `Inversion`.

        The method `np.linalg.solve` is faster to do this, but the Cholesky decomposition is used later in the code
        to speed up the calculation of `log_det_curvature_reg_matrix_term`.
        """
        try:
            return np.linalg.cholesky(self.curvature_reg_matrix_reduced)
        except np.linalg.LinAlgError:
            raise exc.InversionException()

    @cached_property
    @profile_func
    def reconstruction(self) -> np.ndarray:
        """
        Solve the linear system [F + reg_coeff*H] S = D -> S = [F + reg_coeff*H]^-1 D given by equation (12)
        of https://arxiv.org/pdf/astro-ph/0302587.pdf

        S is the vector of reconstructed inversion values.
        """

        return inversion_util.reconstruction_from(
            data_vector=self.data_vector,
            curvature_reg_matrix_cholesky=self.curvature_reg_matrix_cholesky,
            settings=self.settings,
        )

    @cached_property
    @profile_func
    def reconstruction_reduced(self) -> np.ndarray:
        """
        Solve the linear system [F + reg_coeff*H] S = D -> S = [F + reg_coeff*H]^-1 D given by equation (12)
        of https://arxiv.org/pdf/astro-ph/0302587.pdf

        S is the vector of reconstructed inversion values.
        """

        if self.all_linear_obj_have_regularization:
            return self.reconstruction

        return np.delete(self.reconstruction, self.no_regularization_index_list, axis=0)

    @property
    def reconstruction_dict(self) -> Dict[LinearObj, np.ndarray]:
        return self.source_quantity_dict_from(source_quantity=self.reconstruction)

    def source_quantity_dict_from(
        self, source_quantity: np.ndarray
    ) -> Dict[LinearObj, np.ndarray]:
        """
        When constructing the simultaneous linear equations (via vectors and matrices) the quantities of each individual
        linear object (e.g. their `mapping_matrix`) are combined into single ndarrays via stacking. This does not track
        which quantities belong to which linear objects, therefore the linear equation's solutions (which are returned
        as ndarrays) do not contain information on which linear object(s) they correspond to.

        For example, consider if two `Mapper` objects with 50 and 100 source pixels are used in an `Inversion`.
        The `reconstruction` (which contains the solved for source pixels values) is an ndarray of shape [150], but
        the ndarray itself does not track which values belong to which `Mapper`.

        This function converts an ndarray of a `source_quantity` (like a `reconstruction`) to a dictionary of ndarrays,
        where the keys are the instances of each mapper in the inversion.

        Parameters
        ----------
        source_quantity
            The quantity whose values are mapped to a dictionary of values for each individual mapper.

        Returns
        -------
        The dictionary of ndarrays of values for each individual mapper.
        """
        source_quantity_dict = {}

        index = 0

        for linear_obj in self.linear_obj_list:

            source_quantity_dict[linear_obj] = source_quantity[
                index : index + linear_obj.parameters
            ]

            index += linear_obj.parameters

        return source_quantity_dict

    @property
    @profile_func
    def mapped_reconstructed_data_dict(self) -> Dict[LinearObj, Array2D]:
        raise NotImplementedError

    @property
    def mapped_reconstructed_image_dict(self) -> Dict[LinearObj, Array2D]:
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
        return self.mapped_reconstructed_data_dict

    @cached_property
    @profile_func
    def mapped_reconstructed_data(self) -> Union[Array2D, Visibilities]:
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
        return sum(self.mapped_reconstructed_data_dict.values())

    @cached_property
    def mapped_reconstructed_image(self) -> Array2D:
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
        return sum(self.mapped_reconstructed_image_dict.values())

    @cached_property
    @profile_func
    def regularization_term(self) -> float:
        """
        Returns the regularization term of an inversion. This term represents the sum of the difference in flux
        between every pair of neighboring pixels.

        This is computed as:

        s_T * H * s = solution_vector.T * regularization_matrix * solution_vector

        The term is referred to as *G_l* in Warren & Dye 2003, Nightingale & Dye 2015.

        The above works include the regularization_matrix coefficient (lambda) in this calculation. In PyAutoLens,
        this is already in the regularization matrix and thus implicitly included in the matrix multiplication.
        """

        if not self.has(cls=AbstractRegularization):
            return 0.0

        return np.matmul(
            self.reconstruction_reduced.T,
            np.matmul(self.regularization_matrix_reduced, self.reconstruction_reduced),
        )

    @cached_property
    @profile_func
    def log_det_curvature_reg_matrix_term(self) -> float:
        """
        The log determinant of [F + reg_coeff*H] is used to determine the Bayesian evidence of the solution.

        This uses the Cholesky decomposition which is already computed before solving the reconstruction.
        """
        if not self.has(cls=AbstractRegularization):
            return 0.0

        return 2.0 * np.sum(np.log(np.diag(self.curvature_reg_matrix_reduced_cholesky)))

    @cached_property
    @profile_func
    def log_det_regularization_matrix_term(self) -> float:
        """
        The Bayesian evidence of an inversion which quantifies its overall goodness-of-fit uses the log determinant
        of regularization matrix, Log[Det[Lambda*H]].

        Unlike the determinant of the curvature reg matrix, which uses an existing preloading Cholesky decomposition
        used for the source reconstruction, this uses scipy sparse linear algebra to solve the determinant efficiently.

        Returns
        -------
        float
            The log determinant of the regularization matrix.
        """

        if not self.has(cls=AbstractRegularization):
            return 0.0

        if self.preloads.log_det_regularization_matrix_term is not None:
            return self.preloads.log_det_regularization_matrix_term

        try:

            lu = splu(csc_matrix(self.regularization_matrix_reduced))
            diagL = lu.L.diagonal()
            diagU = lu.U.diagonal()
            diagL = diagL.astype(np.complex128)
            diagU = diagU.astype(np.complex128)

            return np.real(np.log(diagL).sum() + np.log(diagU).sum())

        except RuntimeError:

            try:
                return 2.0 * np.sum(
                    np.log(
                        np.diag(np.linalg.cholesky(self.regularization_matrix_reduced))
                    )
                )
            except np.linalg.LinAlgError:
                raise exc.InversionException()

    @property
    def errors_with_covariance(self) -> np.ndarray:
        return np.linalg.inv(self.curvature_reg_matrix)

    def interpolated_reconstruction_list_from(
        self,
        shape_native: Tuple[int, int] = (401, 401),
        extent: Optional[Tuple[float, float, float, float]] = None,
    ) -> List[Array2D]:
        """
        The reconstruction of an `Inversion` can be on an irregular pixelization (e.g. a Delaunay triangulation,
        Voronoi mesh).

        Analysing the reconstruction can therefore be difficult and require specific functionality tailored to using
        this irregular grid.

        This function therefore interpolates the irregular reconstruction on to a regular grid of square pixels.
        The routine that performs the interpolation is specific to each pixelization and contained in its
        corresponding `Mapper` and `Grid2DMesh` objects, which are called by this function.

        The output interpolated reconstruction cis by default returned on a grid of 401 x 401 square pixels. This
        can be customized by changing the `shape_native` input, and a rectangular grid with rectangular pixels can
        be returned by instead inputting the optional `shape_scaled` tuple.

        Parameters
        ----------
        shape_native
            The 2D shape in pixels of the interpolated reconstruction, which is always returned using square pixels.
        extent
            The (x0, x1, y0, y1) extent of the grid in scaled coordinates over which the grid is created if it
            is input.
        """
        return [
            mapper.interpolated_array_from(
                values=self.reconstruction_dict[mapper],
                shape_native=shape_native,
                extent=extent,
            )
            for mapper in self.cls_list_from(cls=AbstractMapper)
        ]

    @property
    def errors(self):
        return np.diagonal(self.errors_with_covariance)

    @property
    def errors_dict(self) -> Dict[LinearObj, np.ndarray]:
        return self.source_quantity_dict_from(source_quantity=self.errors)

    def interpolated_errors_list_from(
        self,
        shape_native: Tuple[int, int] = (401, 401),
        extent: Optional[Tuple[float, float, float, float]] = None,
    ) -> List[Array2D]:
        """
        Analogous to the function `interpolated_reconstruction_list_from()` but for the error in every reconstructed
        pixel.

        See this function for a full description.

        Parameters
        ----------
        shape_native
            The 2D shape in pixels of the interpolated errors, which is always returned using square pixels.
        extent
            The (x0, x1, y0, y1) extent of the grid in scaled coordinates over which the grid is created if it
            is input.
        """
        return [
            mapper.interpolated_array_from(
                values=self.errors_dict[mapper],
                shape_native=shape_native,
                extent=extent,
            )
            for mapper in self.cls_list_from(cls=AbstractMapper)
        ]

    @property
    def magnification_list(self) -> List[float]:

        magnification_list = []

        interpolated_reconstruction_list = self.interpolated_reconstruction_list_from(
            shape_native=(401, 401)
        )

        for i, linear_obj in enumerate(self.linear_obj_list):

            mapped_reconstructed_image = self.mapped_reconstructed_image_dict[
                linear_obj
            ]
            interpolated_reconstruction = interpolated_reconstruction_list[i]

            magnification_list.append(
                np.sum(mapped_reconstructed_image) / np.sum(interpolated_reconstruction)
            )

        return magnification_list

    @property
    def brightest_reconstruction_pixel_list(self):

        brightest_reconstruction_pixel_list = []

        for mapper in self.cls_list_from(cls=AbstractMapper):

            brightest_reconstruction_pixel_list.append(
                np.argmax(self.reconstruction_dict[mapper])
            )

        return brightest_reconstruction_pixel_list

    @property
    def brightest_reconstruction_pixel_centre_list(self):

        brightest_reconstruction_pixel_centre_list = []

        for mapper in self.cls_list_from(cls=AbstractMapper):

            brightest_reconstruction_pixel = np.argmax(self.reconstruction_dict[mapper])

            centre = Grid2DIrregular(
                grid=[mapper.source_mesh_grid[brightest_reconstruction_pixel]]
            )

            brightest_reconstruction_pixel_centre_list.append(centre)

        return brightest_reconstruction_pixel_centre_list

    def regularization_weights_from(self, index: int) -> np.ndarray:

        linear_obj = self.linear_obj_list[index]
        regularization = self.regularization_list[index]

        if regularization is None:

            pixels = linear_obj.parameters

            return np.zero((pixels,))

        return regularization.regularization_weights_from(linear_obj=linear_obj)

    @property
    def regularization_weights_mapper_dict(self) -> Dict[LinearObj, np.ndarray]:

        regularization_weights_dict = {}

        for index, mapper in enumerate(self.cls_list_from(cls=AbstractMapper)):

            regularization_weights_dict[mapper] = self.regularization_weights_from(
                index=index
            )

        return regularization_weights_dict

    @property
    def residual_map_mapper_dict(self) -> Dict[LinearObj, np.ndarray]:

        return {
            mapper: inversion_util.inversion_residual_map_from(
                reconstruction=self.reconstruction_dict[mapper],
                data=self.data,
                slim_index_for_sub_slim_index=mapper.source_grid_slim.mask.slim_index_for_sub_slim_index,
                sub_slim_indexes_for_pix_index=mapper.sub_slim_indexes_for_pix_index,
            )
            for mapper in self.cls_list_from(cls=AbstractMapper)
        }

    @property
    def normalized_residual_map_mapper_dict(self) -> Dict[LinearObj, np.ndarray]:

        return {
            mapper: inversion_util.inversion_normalized_residual_map_from(
                reconstruction=self.reconstruction_dict[mapper],
                data=self.data,
                noise_map_1d=self.noise_map.slim,
                slim_index_for_sub_slim_index=mapper.source_grid_slim.mask.slim_index_for_sub_slim_index,
                sub_slim_indexes_for_pix_index=mapper.sub_slim_indexes_for_pix_index,
            )
            for mapper in self.cls_list_from(cls=AbstractMapper)
        }

    @property
    def chi_squared_map_mapper_dict(self) -> Dict[LinearObj, np.ndarray]:

        return {
            mapper: inversion_util.inversion_chi_squared_map_from(
                reconstruction=self.reconstruction_dict[mapper],
                data=self.data,
                noise_map_1d=self.noise_map.slim,
                slim_index_for_sub_slim_index=mapper.source_grid_slim.mask.slim_index_for_sub_slim_index,
                sub_slim_indexes_for_pix_index=mapper.sub_slim_indexes_for_pix_index,
            )
            for mapper in self.cls_list_from(cls=AbstractMapper)
        }

    @property
    def curvature_matrix_preload(self) -> np.ndarray:
        (
            curvature_matrix_preload,
            curvature_matrix_counts,
        ) = inversion_util.curvature_matrix_preload_from(
            mapping_matrix=self.operated_mapping_matrix
        )

        return curvature_matrix_preload

    @property
    def curvature_matrix_counts(self) -> np.ndarray:
        (
            curvature_matrix_preload,
            curvature_matrix_counts,
        ) = inversion_util.curvature_matrix_preload_from(
            mapping_matrix=self.operated_mapping_matrix
        )

        return curvature_matrix_counts
