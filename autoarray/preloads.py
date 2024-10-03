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
    ):
        self.w_tilde = w_tilde
        self.use_w_tilde = use_w_tilde

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
