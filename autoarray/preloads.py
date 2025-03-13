import logging
import numpy as np
import os
from typing import List

from autoconf import conf

from autoarray.inversion.linear_obj.func_list import AbstractLinearObjFuncList
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
        image_plane_mesh_grid_pg_list=None,
        relocated_grid=None,
        mapper_list=None,
        operated_mapping_matrix=None,
        linear_func_operated_mapping_matrix_dict=None,
        data_linear_func_matrix_dict=None,
        mapper_operated_mapping_matrix_dict=None,
        curvature_matrix=None,
        data_vector_mapper=None,
        curvature_matrix_mapper_diag=None,
        regularization_matrix=None,
        log_det_regularization_matrix_term=None,
        traced_mesh_grids_list_of_planes=None,
        image_plane_mesh_grid_list=None,
    ):
        self.w_tilde = w_tilde
        self.use_w_tilde = use_w_tilde

        self.image_plane_mesh_grid_pg_list = image_plane_mesh_grid_pg_list
        self.relocated_grid = relocated_grid
        self.mapper_list = mapper_list
        self.operated_mapping_matrix = operated_mapping_matrix
        self.linear_func_operated_mapping_matrix_dict = (
            linear_func_operated_mapping_matrix_dict
        )
        self.data_linear_func_matrix_dict = data_linear_func_matrix_dict
        self.mapper_operated_mapping_matrix_dict = mapper_operated_mapping_matrix_dict
        self.curvature_matrix = curvature_matrix
        self.data_vector_mapper = data_vector_mapper
        self.curvature_matrix_mapper_diag = curvature_matrix_mapper_diag
        self.regularization_matrix = regularization_matrix
        self.log_det_regularization_matrix_term = log_det_regularization_matrix_term

        self.traced_mesh_grids_list_of_planes = traced_mesh_grids_list_of_planes
        self.image_plane_mesh_grid_list = image_plane_mesh_grid_list

    @property
    def check_threshold(self):
        return conf.instance["general"]["test"]["preloads_check_threshold"]

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
                noise_map_native=np.array(fit_0.noise_map.native),
                kernel_native=np.array(fit_0.dataset.psf.native),
                native_index_for_slim_index=np.array(
                    fit_0.dataset.mask.derive_indexes.native_for_slim
                ),
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

        The preload is typically used in adapt searches, where the mass model is fixed and the parameters are
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

                logger.info(
                    "PRELOADS - Inversion linear algebra quantities preloaded for this model-fit."
                )

    def set_linear_func_inversion_dicts(self, fit_0, fit_1):
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

        from autoarray.inversion.pixelization.pixelization import Pixelization

        self.linear_func_operated_mapping_matrix_dict = None

        inversion_0 = fit_0.inversion
        inversion_1 = fit_1.inversion

        if inversion_0 is None:
            return

        if not inversion_0.has(cls=AbstractMapper):
            return

        if not inversion_0.has(cls=AbstractLinearObjFuncList):
            return

        try:
            inversion_0.linear_func_operated_mapping_matrix_dict
        except NotImplementedError:
            return

        if not hasattr(inversion_0, "linear_func_operated_mapping_matrix_dict"):
            return

        should_preload = False

        for operated_mapping_matrix_0, operated_mapping_matrix_1 in zip(
            inversion_0.linear_func_operated_mapping_matrix_dict.values(),
            inversion_1.linear_func_operated_mapping_matrix_dict.values(),
        ):
            if (
                np.max(abs(operated_mapping_matrix_0 - operated_mapping_matrix_1))
                < 1e-8
            ):
                should_preload = True
            else:
                should_preload = False
                break

        if should_preload:
            self.linear_func_operated_mapping_matrix_dict = (
                inversion_0.linear_func_operated_mapping_matrix_dict
            )
            self.data_linear_func_matrix_dict = inversion_0.data_linear_func_matrix_dict

            logger.info(
                "PRELOADS - Inversion linear light profile operated mapping matrix / data linear func matrix preloaded for this model-fit."
            )

    def set_curvature_matrix(self, fit_0, fit_1):
        """
        If the `MassProfile`'s and `Mesh`'s in a model are fixed, the mapping of image-pixels to the
        source-pixels does not change during the model-fit and therefore its associated curvature matrix is also
        fixed, meaning the curvature matrix preloaded.

        If linear ``LightProfiles``'s are included, the regions of the curvature matrix associatd with these
        objects vary but the diagonals of the mapper do not change. In this case, the `curvature_matrix_mapper_diag`
        is preloaded.

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
        self.data_vector_mapper = None
        self.curvature_matrix_mapper_diag = None
        self.mapper_operated_mapping_matrix_dict = None

        inversion_0 = fit_0.inversion
        inversion_1 = fit_1.inversion

        if inversion_0 is None:
            return

        try:
            inversion_0._curvature_matrix_mapper_diag
        except NotImplementedError:
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

                return

            if inversion_0._curvature_matrix_mapper_diag is not None:
                if (
                    np.max(
                        abs(
                            inversion_0._curvature_matrix_mapper_diag
                            - inversion_1._curvature_matrix_mapper_diag
                        )
                    )
                    < 1e-8
                ):
                    self.mapper_operated_mapping_matrix_dict = (
                        inversion_0.mapper_operated_mapping_matrix_dict
                    )
                    self.data_vector_mapper = inversion_0._data_vector_mapper
                    self.curvature_matrix_mapper_diag = (
                        inversion_0._curvature_matrix_mapper_diag
                    )

                    logger.info(
                        "PRELOADS - Inversion Curvature Matrix Mapper Diag preloaded for this model-fit."
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

        fit_with_preloads = fit.refit_with_new_preloads(
            preloads=self, settings_inversion=settings_inversion
        )

        fit_without_preloads = fit.refit_with_new_preloads(
            preloads=self.__class__(use_w_tilde=False),
            settings_inversion=settings_inversion,
        )

        if os.environ.get("PYAUTOFIT_TEST_MODE") == "1":
            return

        try:
            if (
                abs(
                    fit_with_preloads.figure_of_merit
                    - fit_without_preloads.figure_of_merit
                )
                > self.check_threshold
            ):
                raise exc.PreloadsException(
                    f"""
                    The log likelihood of fits using and not using preloads are not consistent by a value larger than
                    the preloads check threshold of {self.check_threshold}, indicating preloading has gone wrong.

                    The likelihood values are: 

                    With Preloads: {fit_with_preloads.figure_of_merit}
                    Without Preloads: {fit_without_preloads.figure_of_merit}
                    
                    Double check that the model-fit is set up correctly and that the preloads are being used correctly.
                    
                    This exception can be turned off by setting the general.yaml -> test -> check_preloads to False
                    in the config files. However, care should be taken when doing this.
                    """
                )

        except exc.InversionException:
            data_vector_difference = np.max(
                np.abs(
                    fit_with_preloads.inversion.data_vector
                    - fit_without_preloads.inversion.data_vector
                )
            )

            if data_vector_difference > 1.0e-4:
                raise exc.PreloadsException(
                    f"""
                    The data vectors of fits using and not using preloads are not consistent, indicating 
                    preloading has gone wrong.

                    The maximum value a data vector absolute value difference is: {data_vector_difference} 
                    """
                )

            curvature_reg_matrix_difference = np.max(
                np.abs(
                    fit_with_preloads.inversion.curvature_reg_matrix
                    - fit_without_preloads.inversion.curvature_reg_matrix
                )
            )

            if curvature_reg_matrix_difference > 1.0e-4:
                raise exc.PreloadsException(
                    f"""
                    The curvature matrices of fits using and not using preloads are not consistent, indicating 
                    preloading has gone wrong.

                    The maximum value of a curvature matrix absolute value difference is: {curvature_reg_matrix_difference} 
                    """
                )

    @property
    def info(self) -> List[str]:
        """
        The information on what has or has not been preloaded, which is written to the file `preloads.summary`.

        Returns
        -------
            A list of strings containing statements on what has or has not been preloaded.
        """
        line = [f"W Tilde = {self.w_tilde is not None}\n"]
        line += [f"Relocated Grid = {self.relocated_grid is not None}\n"]
        line += [f"Mapper = {self.mapper_list is not None}\n"]
        line += [
            f"Blurred Mapping Matrix = {self.operated_mapping_matrix is not None}\n"
        ]
        line += [
            f"Inversion Linear Func (Linear Light Profile) Dicts = {self.linear_func_operated_mapping_matrix_dict is not None}\n"
        ]
        line += [f"Curvature Matrix = {self.curvature_matrix is not None}\n"]
        line += [
            f"Curvature Matrix Mapper Diag = {self.curvature_matrix_mapper_diag is not None}\n"
        ]
        line += [f"Regularization Matrix = {self.regularization_matrix is not None}\n"]
        line += [
            f"Log Det Regularization Matrix Term = {self.log_det_regularization_matrix_term is not None}\n"
        ]

        return line
