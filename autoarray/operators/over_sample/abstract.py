import numpy as np
from typing import List, Union


from autoarray.mask.mask_2d import Mask2D
from autoarray.structures.arrays.uniform_2d import Array2D
from autoarray.structures.grids.uniform_2d import Grid2D

from autoarray.numpy_wrapper import register_pytree_node_class


class AbstractOverSample:
    pass


@register_pytree_node_class
class AbstractOverSampleFunc:
    def tree_flatten(self):
        return (self.mask,), ()

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(mask=children[0])
