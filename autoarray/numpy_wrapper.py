import logging

from os import environ

use_jax = environ.get("USE_JAX", "0") == "1"

if use_jax:
    try:
        import jax
        from jax import numpy as np, jit

        print("JAX mode enabled")
    except ImportError:
        raise ImportError(
            "JAX is not installed. Please install it with `pip install jax`."
        )
else:
    import numpy as np

    def jit(function, *_, **__):
        return function


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
