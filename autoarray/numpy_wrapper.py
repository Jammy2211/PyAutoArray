import logging

import jax.numpy as jnp
import numpy as np


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
    def __getattr__(self, item):
        try:
            attribute = getattr(jnp, item)
        except AttributeError as e:
            logging.exception(e)
            attribute = getattr(np, item)
        if callable(attribute):
            return Callable(attribute)
        return attribute


numpy = Numpy()
