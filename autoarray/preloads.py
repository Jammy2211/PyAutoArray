import logging

import jax.numpy as jnp
import numpy as np

logger = logging.getLogger(__name__)

logger.setLevel(level="INFO")


class Preloads:

    def __init__(
        self,
        mapper_indices: np.ndarray = None,
        source_pixel_zeroed_indices: np.ndarray = None,
        linear_light_profile_blurred_mapping_matrix = None,
    ):
        """
        Stores preloaded arrays and matrices used during pixelized linear inversions, improving both performance
        and compatibility with JAX.

        Some arrays (e.g. `mapper_indices`) are required to be defined before sampling begins, because JAX demands
        that input shapes remain static. These are used during each inversion to ensure consistent matrix shapes
        for all likelihood evaluations.

        Other arrays (e.g. parts of the curvature matrix) are preloaded purely to improve performance. In cases where
        the source model is fixed (e.g. when fitting only the lens light), sections of the curvature matrix do not
        change and can be reused, avoiding redundant computation.

        Parameters
        ----------
        mapper_indices
            The integer indices of mapper pixels in the inversion. Used to extract reduced matrices (e.g.
            `curvature_matrix_reduced`) that compute the pixelized inversion's log evidence term, where the indicies
            are requirred to separate the rows and columns of matrices from linear light profiles.
        source_pixel_zeroed_indices
            Indices of source pixels that should be set to zero in the reconstruction. These typically correspond to
            outer-edge source-plane regions with no image-plane mapping (e.g. outside a circular mask), helping
            separate the lens light from the pixelized source model.
        linear_light_profile_blurred_mapping_matrix
            The evaluated images of the linear light profiles that make up the blurred mapping matrix component of the
            inversion, with the other component being the pixelization's pixels. These are fixed when the lens light
            is fixed to the maximum likelihood solution, allowing the blurred mapping matrix to be preloaded, but
            the intensity values will still be solved for during the inversion.
        """

        self.mapper_indices = None
        self.source_pixel_zeroed_indices = None
        self.source_pixel_zeroed_indices_to_keep = None
        self.linear_light_profile_blurred_mapping_matrix = None

        if mapper_indices is not None:

            self.mapper_indices = jnp.array(mapper_indices)

        if source_pixel_zeroed_indices is not None:

            self.source_pixel_zeroed_indices = jnp.array(source_pixel_zeroed_indices)

            ids_zeros = jnp.array(source_pixel_zeroed_indices, dtype=int)

            values_to_solve = jnp.ones(np.max(mapper_indices), dtype=bool)
            values_to_solve = values_to_solve.at[ids_zeros].set(False)

            # Get the indices where values_to_solve is True
            self.source_pixel_zeroed_indices_to_keep = jnp.where(values_to_solve)[0]

        if linear_light_profile_blurred_mapping_matrix is not None:

            self.linear_light_profile_blurred_mapping_matrix = jnp.array(
                linear_light_profile_blurred_mapping_matrix
            )