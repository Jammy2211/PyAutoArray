import numpy as np
from scipy.linalg import block_diag
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import splu
from typing import Dict, List, Optional, Tuple, Union

from autoconf import cached_property
from autoarray.numba_util import profile_func

from autoarray.structures.arrays.uniform_2d import Array2D
from autoarray.structures.grids.irregular_2d import Grid2DIrregular
from autoarray.structures.visibilities import Visibilities
from autoarray.inversion.mappers.abstract import AbstractMapper
from autoarray.inversion.linear_obj.linear_obj import LinearObj
from autoarray.inversion.regularization.abstract import AbstractRegularization
from autoarray.inversion.linear_eqn.imaging.abstract import AbstractLEqImaging
from autoarray.inversion.linear_eqn.interferometer.abstract import (
    AbstractLEqInterferometer,
)
from autoarray.inversion.inversion.settings import SettingsInversion
from autoarray.preloads import Preloads

from autoarray import exc
from autoarray.inversion.inversion import inversion_util


class AbstractInversion:
    def __init__(
        self,
        linear_obj_list : List[LinearObj],
        profiling_dict: Optional[Dict] = None,
    ):

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

        self.linear_obj_list = linear_obj_list

    @property
    def linear_obj_all_with_regularization(self):

        regularization_list = [linear_obj.regularization for linear_obj in self.linear_obj_list]

        if any(regularization_list):
            return True


    @property
    def linear_obj_with_regularization_index_list(self) -> List[int]:

        linear_obj_with_regularization_index_list = []

        pixel_count = 0

        for linear_obj in self.linear_obj_list:

            if linear_obj.regularization is not None:

                linear_obj_with_regularization_index_list.append(pixel_count)

            pixel_count += linear_obj.pixels

        return linear_obj_with_regularization_index_list

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

        if len(self.linear_obj_list) == 1:
            return self.linear_obj_list[0].regularization_matrix

        return block_diag(
            *[
                linear_obj.regularization_matrix
                for linear_obj
                in self.linear_obj_list
            ]
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
        if not self.has_linear_obj_func:
            return self.regularization_matrix

        regularization_matrix = self.regularization_matrix

        regularization_matrix = np.delete(
            regularization_matrix, self.linear_obj_with_regularization_index_list, 0
        )
        regularization_matrix = np.delete(
            regularization_matrix, self.linear_obj_with_regularization_index_list, 1
        )

        return regularization_matrix

    @cached_property
    @profile_func
    def reconstruction(self):
        raise NotImplementedError

    @cached_property
    @profile_func
    def reconstruction_mapper(self):
        """
        Solve the linear system [F + reg_coeff*H] S = D -> S = [F + reg_coeff*H]^-1 D given by equation (12)
        of https://arxiv.org/pdf/astro-ph/0302587.pdf

        S is the vector of reconstructed inversion values.
        """
        if not self.has_linear_obj_func:
            return self.reconstruction

        return np.delete(
            self.reconstruction, self.linear_obj_with_regularization_index_list, axis=0
        )

    @property
    def reconstruction_dict(self) -> Dict[LinearObj, np.ndarray]:
        return self.leq.source_quantity_dict_from(source_quantity=self.reconstruction)

    @property
    def mapped_reconstructed_data_dict(
        self
    ) -> Dict[LinearObj, Union[Array2D, Visibilities]]:
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
        return self.leq.mapped_reconstructed_data_dict_from(
            reconstruction=self.reconstruction
        )

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
        return self.leq.mapped_reconstructed_image_dict_from(
            reconstruction=self.reconstruction
        )

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

    @property
    def errors(self):
        raise NotImplementedError

    @property
    def errors_dict(self) -> Dict[LinearObj, np.ndarray]:
        return self.leq.source_quantity_dict_from(source_quantity=self.errors)

    @property
    def errors_with_covariance(self):
        raise NotImplementedError

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

    @cached_property
    @profile_func
    def regularization_term(self):
        """
        Returns the regularization term of an inversion. This term represents the sum of the difference in flux
        between every pair of neighboring pixels.

        This is computed as:

        s_T * H * s = solution_vector.T * regularization_matrix * solution_vector

        The term is referred to as *G_l* in Warren & Dye 2003, Nightingale & Dye 2015.

        The above works include the regularization_matrix coefficient (lambda) in this calculation. In PyAutoLens,
        this is already in the regularization matrix and thus implicitly included in the matrix multiplication.
        """

        if self.has_mapper:
            return np.matmul(
                self.reconstruction_mapper.T,
                np.matmul(
                    self.regularization_matrix_reduced, self.reconstruction_mapper
                ),
            )
        return 0.0

    @cached_property
    @profile_func
    def log_det_curvature_reg_matrix_term(self):
        """
        The log determinant of [F + reg_coeff*H] is used to determine the Bayesian evidence of the solution.

        This uses the Cholesky decomposition which is already computed before solving the reconstruction.
        """
        raise NotImplementedError

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
    def brightest_reconstruction_pixel_list(self):

        brightest_reconstruction_pixel_list = []

        for mapper in self.mapper_list:

            brightest_reconstruction_pixel_list.append(
                np.argmax(self.reconstruction_dict[mapper])
            )

        return brightest_reconstruction_pixel_list

    @property
    def brightest_reconstruction_pixel_centre_list(self):

        brightest_reconstruction_pixel_centre_list = []

        for mapper in self.mapper_list:

            brightest_reconstruction_pixel = np.argmax(self.reconstruction_dict[mapper])

            centre = Grid2DIrregular(
                grid=[mapper.source_pixelization_grid[brightest_reconstruction_pixel]]
            )

            brightest_reconstruction_pixel_centre_list.append(centre)

        return brightest_reconstruction_pixel_centre_list

    @property
    def error_dict(self) -> Dict[LinearObj, np.ndarray]:
        return self.leq.source_quantity_dict_from(source_quantity=self.errors)

    @property
    def regularization_weights_mapper_dict(self) -> Dict[LinearObj, np.ndarray]:

        regularization_weights_dict = {}

        for mapper, reg in zip(self.mapper_list, self.regularization_list):

            regularization_weights = reg.regularization_weights_from(mapper=mapper)

            regularization_weights_dict[mapper] = regularization_weights

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
            for mapper in self.mapper_list
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
            for mapper in self.mapper_list
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
            for mapper in self.mapper_list
        }

    @property
    def total_mappers(self):
        return len(self.mapper_list)

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
        corresponding `Mapper` and `Grid2DPixelization` objects, which are called by this function.

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
            for mapper in self.mapper_list
        ]

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
            for mapper in self.mapper_list
        ]
