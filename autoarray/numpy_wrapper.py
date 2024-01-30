import logging

import numpy as np
from os import environ


def unwrap_arrays(args):
    from autoarray.abstract_ndarray import AbstractNDArray

    for arg in args:
        if isinstance(arg, AbstractNDArray):
            yield arg.array
        elif isinstance(arg, (list, tuple)):
            yield type(arg)(unwrap_arrays(arg))
        else:
            yield arg


class Callable:
    def __init__(self, func):
        self.func = func

    def __call__(self, *args, **kwargs):
        from autoarray.abstract_ndarray import AbstractNDArray

        try:
            first_argument = args[0]
        except IndexError:
            first_argument = None

        args = unwrap_arrays(args)
        result = self.func(*args, **kwargs)
        if isinstance(first_argument, AbstractNDArray) and not isinstance(
            result, float
        ):
            return first_argument.with_new_array(result)
        return result


class Numpy:
    def __init__(self, jnp):
        self.jnp = jnp

    def __getattr__(self, item):
        try:
            attribute = getattr(self.jnp, item)
        except AttributeError as e:
            logging.debug(e)
            attribute = getattr(np, item)
        if callable(attribute):
            return Callable(attribute)
        return attribute


use_jax = environ.get("USE_JAX", "0") == "1"

if use_jax:
    try:
        import jax.numpy as jnp

        numpy = Numpy(jnp)

        print("JAX mode enabled")
    except ImportError:
        raise ImportError(
            "JAX is not installed. Please install it with `pip install jax`."
        )
else:
    numpy = Numpy(np)

try:
    from jax._src.tree_util import register_pytree_node
    from jax._src.tree_util import register_pytree_node_class

    from jax import Array
except ImportError:

    def register_pytree_node_class(cls):
        return cls

    def register_pytree_node(*_, **__):
        pass

    Array = np.ndarray
