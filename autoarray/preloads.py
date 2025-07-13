import logging

import numpy as np

logger = logging.getLogger(__name__)

logger.setLevel(level="INFO")


class Preloads:

    def __init__(self, mapper_indices: np.ndarray = None):
        """
        Preload in memory arrays and matrices used to perform pixelized linear inversions, for both key functionality
        and speeding up the run-time of the inversion.

        Certain preloading arrays (e.g. `mapper_indices`) are stored here because JAX requires that they are
        known and defined as static arrays before sampling. During each inversion, the preloads will be inspected
        for these fixed arrays and used to change matrix shapes in an identical way for every likelihood evaluation.

        Other preloading arrays are used purely to speed up the run-time of the inversion, such as 
        the `curvature_matrix_preload` array. For certain models (e.g. if the source model is fixed and only the
        lens light is being fitted for), certain quadrants of the `curvature_matrix` are fixed 
        for every likelihood evaluation, meaning that they can be preloaded and used to speed up the inversion.


        Parameters
        ----------
        mapper_indices
            The integer indexes of the mapper pixels in a pixeized inversion, which separate their indexes from those
            of linear light profiles in the inversion. This is used to extract `_reduced` 
            matrices (e.g. `curvature_matrix_reduced`) to compute the `log_evidence` terms of the pixelized inversion
            likelihood function.
        """

        self.mapper_indices = mapper_indices