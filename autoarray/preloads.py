import logging
import numpy as np
from typing import List

from autoconf import conf

from autoarray import exc
from autoarray.inversion.inversion import inversion_util


logger = logging.getLogger(__name__)

logger.setLevel(level="INFO")


class Preloads:
    def __init__(
        self,
        w_tilde=None,
        use_w_tilde=None,
        sparse_image_plane_grids_of_planes=None,
        relocated_grid=None,
        mapper=None,
        blurred_mapping_matrix=None,
        curvature_matrix_sparse_preload=None,
        curvature_matrix_preload_counts=None,
        regularization_matrix=None,
        log_det_regularization_matrix_term=None,
    ):

        self.w_tilde = w_tilde
        self.use_w_tilde = use_w_tilde

        self.sparse_image_plane_grids_of_planes = sparse_image_plane_grids_of_planes
        self.relocated_grid = relocated_grid
        self.mapper = mapper
        self.blurred_mapping_matrix = blurred_mapping_matrix
        self.curvature_matrix_sparse_preload = curvature_matrix_sparse_preload
        self.curvature_matrix_preload_counts = curvature_matrix_preload_counts
        self.regularization_matrix = regularization_matrix
        self.log_det_regularization_matrix_term = log_det_regularization_matrix_term

    def set_w_tilde_imaging(self, fit_0, fit_1):
        """
        The w-tilde linear algebra formalism speeds up inversions by computing beforehand quantities that enable
        efficiently construction of the curvature matrix. These quantites can only be used if the noise-map is
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

        if (
            fit_0.inversion is not None
            and np.max(abs(fit_0.noise_map - fit_1.noise_map)) < 1e-8
        ):

            logger.info("PRELOADS - Computing W-Tilde... May take a moment.")

            if conf.instance["general"]["w_tilde"]["snr_cut_iteration"]:
                self.w_tilde = self.w_tilde_via_snr_cut_iteration(fit=fit_0)
            else:
                self.w_tilde = self.w_tilde_for_snr_cut(fit=fit_0, snr_cut=-1.0e99)

            self.use_w_tilde = True

            logger.info("PRELOADS - W-Tilde preloaded for this model-fit.")

    def w_tilde_via_snr_cut_iteration(self, fit):

        snr_cut = conf.instance["general"]["w_tilde"]["snr_cut_start"]
        likelihood_threshold = fit.imaging.settings.w_tilde_likelihood_threshold

        fit_snr_cut = self.fit_for_snr_cut(fit=fit, snr_cut=snr_cut)
        fit_higher_snr_cut = self.fit_for_snr_cut(fit=fit, snr_cut=snr_cut * 10.0)

        if (
            abs(fit_snr_cut.figure_of_merit - fit_higher_snr_cut.figure_of_merit)
            < likelihood_threshold
        ):
            return self.increase_snr_cut_until_greater_than_likelihood_threshold(
                fit=fit_higher_snr_cut, snr_cut=snr_cut * 10.0
            )

        return self.decrease_snr_cut_until_less_than_likelihood_threshold(
            fit=fit, snr_cut=snr_cut
        )

    def decrease_snr_cut_until_less_than_likelihood_threshold(
        self, fit, snr_cut, iterations=20
    ):

        likelihood_threshold = fit.imaging.settings.w_tilde_likelihood_threshold

        for i in range(iterations):

            fit_prev = fit

            snr_cut /= 10.0

            fit = self.fit_for_snr_cut(fit=fit, snr_cut=snr_cut)

            if (
                abs(fit.figure_of_merit - fit_prev.figure_of_merit)
                < likelihood_threshold
            ):
                return fit.preloads.w_tilde

        raise exc.PreloadsException(
            f"Unable to decrease snr_cut to be less than w_tilde_likelihood_threshold after {iterations} iterations"
        )

    def increase_snr_cut_until_greater_than_likelihood_threshold(
        self, fit, snr_cut, iterations=20
    ):

        likelihood_threshold = fit.imaging.settings.w_tilde_likelihood_threshold

        for i in range(iterations):

            fit_prev = fit

            snr_cut *= 10.0

            try:
                fit = self.fit_for_snr_cut(fit=fit, snr_cut=snr_cut)
            except exc.InversionException:
                return fit_prev.preloads.w_tilde

            if (
                abs(fit.figure_of_merit - fit_prev.figure_of_merit)
                > likelihood_threshold
            ):

                return fit_prev.preloads.w_tilde

        raise exc.PreloadsException(
            f"Unable to decrease snr_cut to be greater than w_tilde_likelihood_threshold after {iterations} iterations"
        )

    def w_tilde_for_snr_cut(self, fit, snr_cut):

        from autoarray.dataset.imaging import WTildeImaging

        preload, indexes, lengths = inversion_util.w_tilde_curvature_preload_imaging_from(
            noise_map_native=fit.noise_map.native,
            signal_to_noise_map_native=fit.signal_to_noise_map.native,
            kernel_native=fit.dataset.psf.native,
            native_index_for_slim_index=fit.dataset.mask.native_index_for_slim_index,
            snr_cut=snr_cut,
        )

        return WTildeImaging(
            curvature_preload=preload,
            indexes=indexes.astype("int"),
            lengths=lengths.astype("int"),
            noise_map_value=fit.noise_map[0],
            snr_cut=snr_cut,
        )

    def fit_for_snr_cut(self, fit, snr_cut):

        w_tilde = self.w_tilde_for_snr_cut(fit=fit, snr_cut=snr_cut)

        preloads = self.__class__(w_tilde=w_tilde, use_w_tilde=True)

        return fit.refit_with_new_preloads(preloads=preloads)

    def set_relocated_grid(self, fit_0, fit_1):
        """
        If the `MassProfile`'s in a model are fixed their traced grid (which may have had coordinates relocated at
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

        mapper_0 = fit_0.inversion.mapper
        mapper_1 = fit_1.inversion.mapper

        if mapper_0.source_grid_slim.shape[0] == mapper_1.source_grid_slim.shape[0]:

            if (
                np.max(abs(mapper_0.source_grid_slim - mapper_1.source_grid_slim))
                < 1.0e-8
            ):

                self.relocated_grid = mapper_0.source_grid_slim

                logger.info(
                    "PRELOADS - Relocated grid of pxielization preloaded for this model-fit."
                )

    def set_mapper(self, fit_0, fit_1):
        """
        If the `MassProfile`'s and `Pixelization`'s in a model are fixed, the mapping of image-pixels to the
        source-pixels does not change during the model-fit and the `Mapper` containing this information can be
        preloaded. This includes preloading the `mapping_matrix`.

        This function compares the mapping matrix of two fit's corresponding to two model instances, and preloads the
        mapper if the mapping matrix of both fits are the same.

        The preload is typically used in searches where only light profiles vary (e.g. when only the lens's light is
        being fitted for).

        Parameters
        ----------
        fit_0
            The first fit corresponding to a model with a specific set of unit-values.
        fit_1
            The second fit corresponding to a model with a different set of unit-values.
        """

        self.mapper = None

        if fit_0.inversion is None:
            return

        mapper_0 = fit_0.inversion.mapper
        mapper_1 = fit_1.inversion.mapper

        if mapper_0.mapping_matrix.shape[1] == mapper_1.mapping_matrix.shape[1]:

            if np.allclose(mapper_0.mapping_matrix, mapper_1.mapping_matrix):

                self.mapper = mapper_0

                logger.info(
                    "PRELOADS - Mappers of planes preloaded for this model-fit."
                )

    def set_inversion(self, fit_0, fit_1):
        """
        If the `MassProfile`'s and `Pixelization`'s in a model are fixed, the mapping of image-pixels to the
        source-pixels does not change during the model-fit and matrices used to perform the linear algebra in an
        inversion can be preloaded, which help efficiently construct the curvature matrix.

        This function compares the blurred mapping matrix of two fit's corresponding to two model instances, and
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

        self.blurred_mapping_matrix = None
        self.curvature_matrix_sparse_preload = None
        self.curvature_matrix_preload_counts = None

        inversion_0 = fit_0.inversion
        inversion_1 = fit_1.inversion

        if inversion_0 is None:
            return

        if (
            inversion_0.blurred_mapping_matrix.shape[1]
            == inversion_1.blurred_mapping_matrix.shape[1]
        ):

            if (
                np.max(
                    abs(
                        inversion_0.blurred_mapping_matrix
                        - inversion_1.blurred_mapping_matrix
                    )
                )
                < 1e-8
            ):

                self.blurred_mapping_matrix = inversion_0.blurred_mapping_matrix
                self.curvature_matrix_sparse_preload = (
                    inversion_0.curvature_matrix_sparse_preload
                ).astype("int")
                self.curvature_matrix_preload_counts = (
                    inversion_0.curvature_matrix_preload_counts
                ).astype("int")

                logger.info(
                    "PRELOADS - Inversion linear algebra quantities preloaded for this model-fit."
                )

    def set_regularization_matrix_and_term(self, fit_0, fit_1):
        """
        If the `MassProfile`'s and `Pixelization`'s in a model are fixed, the mapping of image-pixels to the
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

        likelihood_threshold = fit.imaging.settings.w_tilde_likelihood_threshold

        fom_with_preloads = fit.refit_with_new_preloads(preloads=self).figure_of_merit

        fom_without_preloads = fit.refit_with_new_preloads(
            preloads=self.__class__(use_w_tilde=False)
        ).figure_of_merit

        if abs(fom_with_preloads - fom_without_preloads) > likelihood_threshold:

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
        self.sparse_image_plane_grids_of_planes = None
        self.relocated_grid = None
        self.mapper = None
        self.blurred_mapping_matrix = None
        self.curvature_matrix_sparse_preload = None
        self.curvature_matrix_preload_counts = None
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
        line += [f"Mapper = {self.mapper is not None}\n"]
        line += [
            f"Blurred Mapping Matrix = {self.blurred_mapping_matrix is not None}\n"
        ]
        line += [
            f"Curvature Matrix Sparse = {self.curvature_matrix_sparse_preload is not None}\n"
        ]
        line += [f"Regularization Matrix = {self.regularization_matrix is not None}\n"]
        line += [
            f"Log Det Regularization Matrix Term = {self.log_det_regularization_matrix_term is not None}\n"
        ]

        return line
