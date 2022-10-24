import copy
import logging
import numpy as np
from typing import List, Optional

from autoconf import cached_property

from autoarray.dataset.abstract_dataset import AbstractWTilde
from autoarray.dataset.abstract_dataset import AbstractSettingsDataset
from autoarray.dataset.abstract_dataset import AbstractDataset
from autoarray.structures.arrays.uniform_2d import Array2D
from autoarray.operators.convolver import Convolver
from autoarray.structures.grids.uniform_2d import Grid2D
from autoarray.structures.arrays.kernel_2d import Kernel2D
from autoarray.mask.mask_2d import Mask2D

from autoarray import exc
from autoarray.inversion.inversion import inversion_util
from autoarray.inversion.inversion.imaging import inversion_imaging_util
from autoarray.dataset import preprocess

logger = logging.getLogger(__name__)


class WTildeImaging(AbstractWTilde):
    def __init__(
        self,
        curvature_preload: np.ndarray,
        indexes: np.ndim,
        lengths: np.ndarray,
        noise_map_value: float,
    ):
        """
        Packages together all derived data quantities necessary to fit `Imaging` data using an ` Inversion` via the
        w_tilde formalism.

        The w_tilde formalism performs linear algebra formalism in a way that speeds up the construction of the
        simultaneous linear equations by bypassing the construction of a `mapping_matrix` and precomputing the
        blurring operations performed using the imaging's PSF.

        Parameters
        ----------
        curvature_preload
            A matrix which uses the imaging's noise-map and PSF to preload as much of the computation of the
            curvature matrix as possible.
        indexes
            The image-pixel indexes of the curvature preload matrix, which are used to compute the curvature matrix
            efficiently when performing an inversion.
        lengths
            The lengths of how many indexes each curvature preload contains, again used to compute the curvature
            matrix efficienctly.
        noise_map_value
            The first value of the noise-map used to construct the curvature preload, which is used as a sanity
            check when performing the inversion to ensure the preload corresponds to the data being fitted.
        """
        super().__init__(
            curvature_preload=curvature_preload, noise_map_value=noise_map_value
        )

        self.indexes = indexes
        self.lengths = lengths
