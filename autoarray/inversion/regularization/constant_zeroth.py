from __future__ import annotations
import jax.numpy as jnp
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from autoarray.inversion.linear_obj.linear_obj import LinearObj

from autoarray.inversion.regularization.abstract import AbstractRegularization


def constant_zeroth_regularization_matrix_from(
    coefficient: float,
    coefficient_zeroth: float,
    neighbors: jnp.ndarray,
    neighbors_sizes: jnp.ndarray[[int], jnp.int64],
) -> jnp.ndarray:
    """
    From the pixel-neighbors array, setup the regularization matrix using the instance regularization scheme.

    A complete description of regularizatin and the ``regularization_matrix`` can be found in the ``Regularization``
    class in the module ``autoarray.inversion.regularization``.

    Parameters
    ----------
    coefficients
        The regularization coefficients which controls the degree of smoothing of the inversion reconstruction.
    neighbors
        An array of length (total_pixels) which provides the index of all neighbors of every pixel in
        the Voronoi grid (entries of -1 correspond to no neighbor).
    neighbors_sizes
        An array of length (total_pixels) which gives the number of neighbors of every pixel in the
        Voronoi grid.

    Returns
    -------
    jnp.ndarray
        The regularization matrix computed using Regularization where the effective regularization
        coefficient of every source pixel is the same.
    """
    S, P = neighbors.shape
    # as the regularization matrix is S by S, S would be out of bound (any out of bound index would do)
    OUT_OF_BOUND_IDX = S
    regularization_coefficient = coefficient * coefficient

    # flatten it for feeding into the matrix as j indices
    neighbors = neighbors.flatten()
    # now create the corresponding i indices
    I_IDX = jnp.repeat(jnp.arange(S), P)
    # Entries of `-1` in `neighbors` (indicating no neighbor) are replaced with an out-of-bounds index.
    # This ensures that JAX can efficiently drop these entries during matrix updates.
    neighbors = jnp.where(neighbors == -1, OUT_OF_BOUND_IDX, neighbors)
    const = (
        jnp.diag(1e-8 + regularization_coefficient * neighbors_sizes).at[I_IDX, neighbors]
        # unique indices should be guranteed by neighbors-spec
        .add(-regularization_coefficient, mode="drop", unique_indices=True)
    )

    reg_coeff = coefficient_zeroth ** 2.0
    # Identity matrix scaled by reg_coeff does exactly ∑_i reg_coeff * e_i e_i^T
    zeroth = jnp.eye(P) * reg_coeff

    return const + zeroth


class ConstantZeroth(AbstractRegularization):
    def __init__(self, coefficient_neighbor=1.0, coefficient_zeroth=1.0):
        super().__init__()

        self.coefficient_neighbor = coefficient_neighbor
        self.coefficient_zeroth = coefficient_zeroth

    def regularization_weights_from(self, linear_obj: LinearObj) -> jnp.ndarray:
        """
        Returns the regularization weights of this regularization scheme.

        The regularization weights define the level of regularization applied to each parameter in the linear object
        (e.g. the ``pixels`` in a ``Mapper``).

        For standard regularization (e.g. ``Constant``) are weights are equal, however for adaptive schemes
        (e.g. ``AdaptiveBrightness``) they vary to adapt to the data being reconstructed.

        Parameters
        ----------
        linear_obj
            The linear object (e.g. a ``Mapper``) which uses these weights when performing regularization.

        Returns
        -------
        The regularization weights.
        """
        return self.coefficient_neighbor * jnp.ones(linear_obj.params)

    def regularization_matrix_from(self, linear_obj: LinearObj) -> jnp.ndarray:
        """
        Returns the regularization matrix with shape [pixels, pixels].

        Parameters
        ----------
        linear_obj
            The linear object (e.g. a ``Mapper``) which uses this matrix to perform regularization.

        Returns
        -------
        The regularization matrix.
        """
        return constant_zeroth_regularization_matrix_from(
            coefficient=self.coefficient_neighbor,
            coefficient_zeroth=self.coefficient_zeroth,
            neighbors=linear_obj.neighbors,
        )
