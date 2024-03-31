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

    def structure_2d_from(
        self, result: np.ndarray,
    ) -> Union[Array2D, "Grid2D"]:
        """
        Convert a result from an ndarray to an aa.Array2D or aa.Grid2D structure, where the conversion depends on
        type(result) as follows:

        - 1D np.ndarray   -> aa.Array2D
        - 2D np.ndarray   -> aa.Grid2D

        This function is used by the grid_2d_to_structure decorator to convert the output result of a function
        to an autoarray structure when a `Grid2D` instance is passed to the decorated function.

        Parameters
        ----------
        result or [np.ndarray]
            The input result (e.g. of a decorated function) that is converted to a PyAutoArray structure.
        """
        from autoarray.structures.grids.transformed_2d import Grid2DTransformed
        from autoarray.structures.grids.transformed_2d import Grid2DTransformedNumpy

        if len(result.shape) == 1:
            return Array2D(values=result, mask=self.mask)
        else:
            if isinstance(result, Grid2DTransformedNumpy):
                return Grid2DTransformed(values=result, mask=self.mask, over_sample=self.over_sample)
            return Grid2D(values=result, mask=self.mask, over_sample=self.over_sample)

    def structure_2d_list_from(
        self, result_list: List,
    ) -> List[Union[Array2D, "Grid2D"]]:
        """
        Convert a result from a list of ndarrays to a list of aa.Array2D or aa.Grid2D structure, where the conversion
        depends on type(result) as follows:

        - [1D np.ndarray] -> [aa.Array2D]
        - [2D np.ndarray] -> [aa.Grid2D]

        This function is used by the grid_like_list_to_structure-list decorator to convert the output result of a
        function to a list of autoarray structure when a `Grid2D` instance is passed to the decorated function.

        Parameters
        ----------
        result_list or [np.ndarray]
            The input result (e.g. of a decorated function) that is converted to a PyAutoArray structure.
        """
        return [
            self.structure_2d_from(result=result) for result in result_list
        ]

