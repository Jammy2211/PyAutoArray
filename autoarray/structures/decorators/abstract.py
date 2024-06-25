from typing import List, Union

from autoarray.mask.mask_1d import Mask1D
from autoarray.mask.mask_2d import Mask2D
from autoarray.operators.over_sampling.abstract import AbstractOverSampling
from autoarray.structures.grids.uniform_1d import Grid1D
from autoarray.structures.grids.irregular_2d import Grid2DIrregular
from autoarray.structures.grids.uniform_2d import Grid2D


class AbstractMaker:
    def __init__(self, func, obj, grid, *args, **kwargs):
        """
        Makes 2D data structures from an input function and grid, ensuring that the structure of the input grid is
        paired to the structure of the output data structure.

        This is used by the `to_array`, `to_grid` and `to_vector_yx` decorators to ensure that the input grid and output
        data structure are consistent.

        There are three types of consistent data structures and therefore decorated function mappings:

        - Uniform: 2D structures defined on a uniform grid of data points, for example the `Array2D` and `Grid2D`
        objects. Both structures are defined according to a `Mask2D`, which the maker object ensures is passed through
        self consistently.

        - Irregular: 2D structures defined on an irregular grid of data points, for example an `ArrayIrregular`
        and `Grid2DIrregular` objects. Neither structure is defined according to a mask and the maker sures the lack of
        a mask does not prevent the function from being evaluated.

        - 1D: 1D structures defined on a 1D grid of data points, for example the `Array1D` and `Grid1D` objects.
        These project the 1D grid to a 2D grid to ensure the function can be evaluated, and then deproject the 2D grid
        back to a 1D grid to ensure the output data structure is consistent with the input grid.

        Parameters
        ----------
        func
            The function that is being decorated, which the maker object evaluates to create the output data structure.
        obj
            The object that the function is a method of, which is passed to the function when it is evaluated. This
            is typically the self object of the class the function is a method of.
        grid
            The grid that is passed to the function when it is evaluated, which is used to evaluate the function.
            For example, this might be the `Grid2D` object that the function checks the typing of when determining
            the output data structure.
        args
            Any arguments that are passed to the function when it is evaluated.
        kwargs
            Any keyword arguments that are passed to the function when it is evaluated.
        """

        self.func = func
        self.obj = obj
        self.grid = grid
        self.args = args
        self.kwargs = kwargs

    @property
    def mask(self) -> Union[Mask1D, Mask2D]:
        return self.grid.mask

    @property
    def over_sampling(self) -> AbstractOverSampling:
        return self.grid.over_sampling

    @property
    def over_sampling_non_uniform(self) -> AbstractOverSampling:
        return self.grid.over_sampling_non_uniform

    def via_grid_2d(self, result):
        raise NotImplementedError

    def via_grid_2d_irr(self, result):
        raise NotImplementedError

    def via_grid_1d(self, result):
        raise NotImplementedError

    @property
    def evaluate_func(self):
        """
        Evaluate the function that is being decorated, using the grid that is passed to the maker object when it is
        initialized.

        In normal usage, the input grid is 2D and it is simply passed to the decorated function.

        However, if the input grid is 1D, the grid is projected to a 2D grid before being passed to the function. This
        is because the function is expected to evaluate a 2D grid, and the maker object ensures that the function can
        be evaluated by projecting the 1D grid to a 2D grid.

        Returns
        -------
        The result of the function that is being decorated, which is the output data structure that is consistent with
        the input grid.
        """

        if isinstance(self.grid, Grid1D):
            grid = self.grid.grid_2d_radial_projected_from()
            return self.func(self.obj, grid, *self.args, **self.kwargs)
        return self.func(self.obj, self.grid, *self.args, **self.kwargs)

    @property
    def result(self):
        """
        The result of the function that is being decorated, which this function converts to the output data structure
        that is consistent with the input grid.

        This function called one of three methods, depending on the type of the input grid:

        - `via_grid_2d`: If the input grid is a `Grid2D` object.
        - `via_grid_2d_irr`: If the input grid is a `Grid2DIrregular` object.
        - `via_grid_1d`: If the input grid is a `Grid1D` object.

        These functions are over written depending on whether the decorated function returns an array, grid or vector.
        The over written functions are in the child classes `ArrayMaker`, `GridMaker` and `VectorYXMaker`.
        """

        if isinstance(self.grid, Grid2D):
            return self.via_grid_2d(self.evaluate_func)
        elif isinstance(self.grid, Grid2DIrregular):
            return self.via_grid_2d_irr(self.evaluate_func)
        elif isinstance(self.grid, Grid1D):
            return self.via_grid_1d(self.evaluate_func)

        return self.evaluate_func
