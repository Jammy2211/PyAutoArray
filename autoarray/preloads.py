import logging
import numpy as np
from typing import List

from autoarray.inversion.inversion.imaging.abstract import AbstractInversionImaging
from autoarray.inversion.pixelization.mappers.abstract import AbstractMapper

from autoarray import exc
from autoarray.inversion.inversion.imaging import inversion_imaging_util

logger = logging.getLogger(__name__)

logger.setLevel(level="INFO")


class Preloads:
    def __init__(
        self,
        w_tilde=None,
        use_w_tilde=None,
        sparse_image_plane_grid_pg_list=None,
        relocated_grid=None,
        mapper_list=None,
        operated_mapping_matrix=None,
        curvature_matrix_preload=None,
        curvature_matrix_counts=None,
        curvature_matrix=None,
        regularization_matrix=None,
        log_det_regularization_matrix_term=None,
        traced_sparse_grids_list_of_planes=None,
        sparse_image_plane_grid_list=None,
    ):

        self.w_tilde = w_tilde
        self.use_w_tilde = use_w_tilde

        self.sparse_image_plane_grid_pg_list = sparse_image_plane_grid_pg_list
        self.relocated_grid = relocated_grid
        self.mapper_list = mapper_list
        self.operated_mapping_matrix = operated_mapping_matrix
        self.curvature_matrix_preload = curvature_matrix_preload
        self.curvature_matrix_counts = curvature_matrix_counts
        self.curvature_matrix = curvature_matrix
        self.regularization_matrix = regularization_matrix
        self.log_det_regularization_matrix_term = log_det_regularization_matrix_term

        self.traced_sparse_grids_list_of_planes = traced_sparse_grids_list_of_planes
        self.sparse_image_plane_grid_list = sparse_image_plane_grid_list

    def set_w_tilde_imaging(self, fit_0, fit_1):
        """
        The w-tilde linear algebra formalism speeds up inversions by computing beforehand quantities that enable
        efficiently construction of the curvature matrix. These quantities can only be used if the noise-map is
        fixed, therefore this function preloads these w-tilde quantities if the noise-map does not change.

        This function compares the noise map of two fit's corresponding to two model instances, and preloads wtilde
        if the noise maps of both fits are the same.

        The preload is typically used through search chaining pipelines, as it is uncommon for the noise map to be
        scaled during the model-fit (although it is common for a fixed but scaled noise map to be used).

        Parameters
        ----------
        fit_0
            The first fit corresponding to a model with a specific set of unit-values.
        fit_1
            The second fit corresponding to a model with a different set of unit-values.
        """

        self.w_tilde = None
        self.use_w_tilde = False

        if fit_0.inversion is None:
            return

        if not fit_0.inversion.has(cls=AbstractMapper):
            return

        if np.max(abs(fit_0.noise_map - fit_1.noise_map)) < 1e-8:

            logger.info("PRELOADS - Computing W-Tilde... May take a moment.")

            from autoarray.dataset.imaging.w_tilde import WTildeImaging

            (
                preload,
                indexes,
                lengths,
            ) = inversion_imaging_util.w_tilde_curvature_preload_imaging_from(
                noise_map_native=fit_0.noise_map.native,
                kernel_native=fit_0.dataset.psf.native,
                native_index_for_slim_index=fit_0.dataset.mask.indexes.native_for_slim,
            )

            self.w_tilde = WTildeImaging(
                curvature_preload=preload,
                indexes=indexes.astype("int"),
                lengths=lengths.astype("int"),
                noise_map_value=fit_0.noise_map[0],
            )

            self.use_w_tilde = True

            logger.info("PRELOADS - W-Tilde preloaded for this model-fit.")

    def set_relocated_grid(self, fit_0, fit_1):
        """
        If the `MassProfile`'s in a model are fixed their traced grids (which may have had coordinates relocated at
        the border) does not change during the model=fit and can therefore be preloaded.

        This function compares the relocated grids of the mappers of two fit corresponding to two model instances, and
        preloads the grid if the grids of both fits are the same.

        The preload is typically used in hyper searches, where the mass model is fixed and the hyper-parameters are
        varied.

        Parameters
        ----------
        fit_0
            The first fit corresponding to a model with a specific set of unit-values.
        fit_1
            The second fit corresponding to a model with a different set of unit-values.
        """

        self.relocated_grid = None

        if fit_0.inversion is None:
            return

        if (
            fit_0.inversion.total(cls=AbstractMapper) > 1
            or fit_0.inversion.total(cls=AbstractMapper) == 0
        ):
            return

        mapper_0 = fit_0.inversion.cls_list_from(cls=AbstractMapper)[0]
        mapper_1 = fit_1.inversion.cls_list_from(cls=AbstractMapper)[0]

        if (
            mapper_0.source_plane_data_grid.shape[0]
            == mapper_1.source_plane_data_grid.shape[0]
        ):

            if (
                np.max(
                    abs(
                        mapper_0.source_plane_data_grid
                        - mapper_1.source_plane_data_grid
                    )
                )
                < 1.0e-8
            ):

                self.relocated_grid = mapper_0.source_plane_data_grid

                logger.info(
                    "PRELOADS - Relocated grid of pxielization preloaded for this model-fit."
                )

    def set_mapper_list(self, fit_0, fit_1):
        """
        If the `MassProfile`'s and `Mesh`'s in a model are fixed, the mapping of image-pixels to the
        source-pixels does not change during the model-fit and the list of `Mapper`'s containing this information can
        be preloaded. This includes preloading the `mapping_matrix`.

        This function compares the mapping matrix of two fit's corresponding to two model instances, and preloads the
        list of mappers if the mapping matrix of both fits are the same.

        The preload is typically used in searches where only light profiles vary (e.g. when only the lens's light is
        being fitted for).

        Parameters
        ----------
        fit_0
            The first fit corresponding to a model with a specific set of unit-values.
        fit_1
            The second fit corresponding to a model with a different set of unit-values.
        """

        self.mapper_list = None

        if fit_0.inversion is None:
            return

        if fit_0.inversion.total(cls=AbstractMapper) == 0:
            return

        from autoarray.inversion.inversion.interferometer.lop import (
            InversionInterferometerMappingPyLops,
        )

        if isinstance(fit_0.inversion, InversionInterferometerMappingPyLops):
            return

        inversion_0 = fit_0.inversion
        inversion_1 = fit_1.inversion

        if inversion_0.mapping_matrix.shape[1] == inversion_1.mapping_matrix.shape[1]:

            if np.allclose(inversion_0.mapping_matrix, inversion_1.mapping_matrix):

                self.mapper_list = inversion_0.cls_list_from(cls=AbstractMapper)

                logger.info(
                    "PRELOADS - Mappers of planes preloaded for this model-fit."
                )

    def set_operated_mapping_matrix_with_preloads(self, fit_0, fit_1):
        """
        If the `MassProfile`'s and `Mesh`'s in a model are fixed, the mapping of image-pixels to the
        source-pixels does not change during the model-fit and matrices used to perform the linear algebra in an
        inversion can be preloaded, which help efficiently construct the curvature matrix.

        This function compares the operated mapping matrix of two fit's corresponding to two model instances, and
        preloads the mapper if the mapping matrix of both fits are the same.

        The preload is typically used in searches where only light profiles vary (e.g. when only the lens's light is
        being fitted for).

        Parameters
        ----------
        fit_0
            The first fit corresponding to a model with a specific set of unit-values.
        fit_1
            The second fit corresponding to a model with a different set of unit-values.
        """

        self.operated_mapping_matrix = None
        self.curvature_matrix_preload = None
        self.curvature_matrix_counts = None

        from autoarray.inversion.inversion.interferometer.lop import (
            InversionInterferometerMappingPyLops,
        )

        if isinstance(fit_0.inversion, InversionInterferometerMappingPyLops):
            return

        inversion_0 = fit_0.inversion
        inversion_1 = fit_1.inversion

        if inversion_0 is None:
            return

        if (
            inversion_0.operated_mapping_matrix.shape[1]
            == inversion_1.operated_mapping_matrix.shape[1]
        ):

            if (
                np.max(
                    abs(
                        inversion_0.operated_mapping_matrix
                        - inversion_1.operated_mapping_matrix
                    )
                )
                < 1e-8
            ):

                self.operated_mapping_matrix = inversion_0.operated_mapping_matrix

                if isinstance(inversion_0, AbstractInversionImaging):

                    self.curvature_matrix_preload = (
                        inversion_0.curvature_matrix_preload
                    ).astype("int")
                    self.curvature_matrix_counts = (
                        inversion_0.curvature_matrix_counts
                    ).astype("int")

                logger.info(
                    "PRELOADS - Inversion linear algebra quantities preloaded for this model-fit."
                )

    def set_curvature_matrix(self, fit_0, fit_1):
        """
        If the `MassProfile`'s and `Mesh`'s in a model are fixed, the mapping of image-pixels to the
        source-pixels does not change during the model-fit and therefore its associated cruvature matrix is also
        fixed, meaning the curvature matrix preloaded.

        This function compares the curvature matrix of two fit's corresponding to two model instances, and preloads
        this value if it is the same for both fits.

        The preload is typically used in **PyAutoGalaxy** inversions using a `Rectangular` pixelization.

        Parameters
        ----------
        fit_0
            The first fit corresponding to a model with a specific set of unit-values.
        fit_1
            The second fit corresponding to a model with a different set of unit-values.
        """
        self.curvature_matrix = None

        inversion_0 = fit_0.inversion
        inversion_1 = fit_1.inversion

        if inversion_0 is None:
            return

        if inversion_0.curvature_matrix.shape == inversion_1.curvature_matrix.shape:

            if (
                np.max(abs(inversion_0.curvature_matrix - inversion_1.curvature_matrix))
                < 1e-8
            ):

                self.curvature_matrix = inversion_0.curvature_matrix

                logger.info(
                    "PRELOADS - Inversion Curvature Matrix preloaded for this model-fit."
                )

    def set_regularization_matrix_and_term(self, fit_0, fit_1):
        """
        If the `MassProfile`'s and `Mesh`'s in a model are fixed, the mapping of image-pixels to the
        source-pixels does not change during the model-fit and therefore its associated regularization matrices are
        also fixed, meaning the log determinant of the regularization matrix term of the Bayesian evidence can be
        preloaded.

        This function compares the value of the log determinant of the regularization matrix of two fit's corresponding
        to two model instances, and preloads this value if it is the same for both fits.

        The preload is typically used in searches where only light profiles vary (e.g. when only the lens's light is
        being fitted for).

        Parameters
        ----------
        fit_0
            The first fit corresponding to a model with a specific set of unit-values.
        fit_1
            The second fit corresponding to a model with a different set of unit-values.
        """
        self.regularization_matrix = None
        self.log_det_regularization_matrix_term = None

        inversion_0 = fit_0.inversion
        inversion_1 = fit_1.inversion

        if inversion_0 is None:
            return

        if inversion_0.total(cls=AbstractMapper) == 0:
            return

        if (
            abs(
                inversion_0.log_det_regularization_matrix_term
                - inversion_1.log_det_regularization_matrix_term
            )
            < 1e-8
        ):

            self.regularization_matrix = inversion_0.regularization_matrix
            self.log_det_regularization_matrix_term = (
                inversion_0.log_det_regularization_matrix_term
            )

            logger.info(
                "PRELOADS - Inversion Log Det Regularization Matrix Term preloaded for this model-fit."
            )

    def check_via_fit(self, fit):

        import copy

        settings_inversion = copy.deepcopy(fit.settings_inversion)
        settings_inversion.use_curvature_matrix_preload = False

        fom_with_preloads = fit.refit_with_new_preloads(
            preloads=self, settings_inversion=settings_inversion
        ).figure_of_merit

        fom_without_preloads = fit.refit_with_new_preloads(
            preloads=self.__class__(use_w_tilde=False)
        ).figure_of_merit

        if abs(fom_with_preloads - fom_without_preloads) > 1.0e-4:

            raise exc.PreloadsException(
                f"The log likelihood of fits using and not using preloads are not"
                f"consistent, indicating preloading has gone wrong."
                f"The likelihood values are {fom_with_preloads} (with preloads) and "
                f"{fom_without_preloads} (without preloads)"
            )

    def reset_all(self):
        """
        Reset all preloads, typically done at the end of a model-fit to save memory.
        """
        self.w_tilde = None

        self.blurred_image = None
        self.traced_grids_of_planes_for_inversion = None
        self.sparse_image_plane_grid_pg_list = None
        self.relocated_grid = None
        self.mapper_list = None
        self.operated_mapping_matrix = None
        self.curvature_matrix_preload = None
        self.curvature_matrix_counts = None
        self.curvature_matrix = None
        self.regularization_matrix = None
        self.log_det_regularization_matrix_term = None

    @property
    def info(self) -> List[str]:
        """
        The information on what has or has not been preloaded, which is written to the file `preloads.summary`.

        Returns
        -------
            A list of strings containing True and False values as to whether a quantity has been preloaded.
        """
        line = [f"W Tilde = {self.w_tilde is not None}\n"]
        line += [f"Relocated Grid = {self.relocated_grid is not None}\n"]
        line += [f"Mapper = {self.mapper_list is not None}\n"]
        line += [
            f"Blurred Mapping Matrix = {self.operated_mapping_matrix is not None}\n"
        ]
        line += [
            f"Curvature Matrix Sparse = {self.curvature_matrix_preload is not None}\n"
        ]
        line += [f"Curvature Matrix = {self.curvature_matrix is not None}\n"]
        line += [f"Regularization Matrix = {self.regularization_matrix is not None}\n"]
        line += [
            f"Log Det Regularization Matrix Term = {self.log_det_regularization_matrix_term is not None}\n"
        ]

        return line
