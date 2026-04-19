"""Tests for gated JAX pytree registration of ``AbstractNDArray`` subclasses.

Follows the three-step pattern from ``autolens_workspace_test/scripts/hessian_jax.py``:
1. NumPy path — confirm autoarray type with ``np.ndarray`` backing, no pytree registration.
2. JAX path outside JIT — same autoarray type with ``jax.Array`` backing; pytree registered.
3. JAX path through ``jax.jit`` — round-trip the instance and assert the output carries
   a ``jax.Array`` leaf.
"""

import numpy as np
import numpy.testing as npt
import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

from autoarray.abstract_ndarray import AbstractNDArray, _pytree_registered_classes


class _LeafArray(AbstractNDArray):
    """Minimal concrete ``AbstractNDArray`` with no nested autoarray children.

    Isolates the pytree-registration machinery from the larger autoarray
    hierarchy: a real ``Array2D`` also carries a ``Mask2D`` and other nested
    ``AbstractNDArray`` children whose own registration is covered by
    follow-up steps in the ``fit-imaging-pytree`` task.
    """

    @property
    def native(self):
        return self


def test_numpy_path_does_not_register_pytree():
    _pytree_registered_classes.discard(_LeafArray)

    arr = _LeafArray(np.array([1.0, 2.0, 3.0]))

    assert isinstance(arr._array, np.ndarray)
    assert _LeafArray not in _pytree_registered_classes


def test_jax_path_registers_pytree_once():
    _pytree_registered_classes.discard(_LeafArray)

    arr_jax = _LeafArray(jnp.array([1.0, 2.0, 3.0]), xp=jnp)

    assert isinstance(arr_jax._array, jnp.ndarray)
    assert _LeafArray in _pytree_registered_classes

    # Second construction on the JAX path is a no-op; class stays registered.
    _LeafArray(jnp.array([4.0, 5.0]), xp=jnp)
    assert _LeafArray in _pytree_registered_classes


def test_jax_jit_round_trip_returns_wrapper_with_jax_array():
    arr_jax = _LeafArray(jnp.array([1.0, 2.0, 3.0]), xp=jnp)
    assert _LeafArray in _pytree_registered_classes

    result = jax.jit(lambda a: a)(arr_jax)

    assert isinstance(result, _LeafArray)
    assert isinstance(result._array, jnp.ndarray)
    npt.assert_allclose(np.asarray(result._array), np.asarray(arr_jax._array))
