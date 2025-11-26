from functools import wraps
import numpy as np
from typing import List, Union

from autoarray.structures.decorators.abstract import AbstractMaker
from autoarray.structures.grids.uniform_1d import Grid1D
from autoarray.structures.grids.irregular_2d import Grid2DIrregular
from autoarray.structures.grids.uniform_2d import Grid2D


class GridMaker(AbstractMaker):
    def via_grid_2d(self, result) -> Union[Grid2D, List[Grid2D]]:
        """
        Convert the result of a decorated function which receives as input a `Grid2D` object to an `Grid2D` object.

        If the result returns a list, a list of `Grid2D` objects is returned.

        Parameters
        ----------
        result
            The input result (e.g. of a decorated function) that is converted to a Grid2D or list of Grid2D objects.
        """
        if not isinstance(result, list):
            try:
                over_sampled = result.over_sampled
            except AttributeError:
                over_sampled = None

            try:
                over_sampler = result.over_sampler
            except AttributeError:
                over_sampler = None

            return Grid2D(
                values=result,
                mask=self.mask,
                over_sample_size=self.over_sample_size,
                over_sampled=over_sampled,
                over_sampler=over_sampler,
            )

        try:
            grid_over_sampled_list = [res.over_sampled for res in result]
            grid_over_sampler_list = [res.over_sampler for res in result]
        except AttributeError:
            grid_over_sampled_list = [None] * len(result)
            grid_over_sampler_list = [None] * len(result)

        return [
            Grid2D(
                values=res,
                mask=self.mask,
                over_sample_size=self.over_sample_size,
                over_sampled=over_sampled,
                over_sampler=over_sampler,
            )
            for res, over_sampled, over_sampler in zip(
                result, grid_over_sampled_list, grid_over_sampler_list
            )
        ]

    def via_grid_2d_irr(self, result) -> Union[Grid2DIrregular, List[Grid2DIrregular]]:
        """
        Convert the result of a decorated function which receives as input a `Grid2DIrregular` object to
        an `Grid2DIrregular` object.

        If the result returns a list, a list of `Grid2DIrregular` objects is returned.

        Parameters
        ----------
        result
            The input result (e.g. of a decorated function) that is converted to an Grid2DIrregular or list of
            `Grid2DIrregular` objects.
        """
        if not isinstance(result, list):
            return Grid2DIrregular(values=result)
        return [Grid2DIrregular(values=res) for res in result]

    def via_grid_1d(self, result) -> Union[Grid2D, List[Grid2D]]:
        """
        Convert the result of a decorated function which receives as input a `Grid1D` object to a `Grid2D` object
        where a projection is performed from 1D to 2D before the function is evaluated.

        If the result returns a list, a list of `Grid2D` objects is returned.

        Parameters
        ----------
        result
            The input result (e.g. of a decorated function) that is converted to a Grid2D or list of Grid2D objects.
        """
        if not isinstance(result, list):
            return Grid2D(values=result, mask=self.mask.derive_mask.to_mask_2d)
        return [
            Grid2D(values=res, mask=self.mask.derive_mask.to_mask_2d) for res in result
        ]


def to_grid(func):
    """
    Homogenize the inputs and outputs of functions that take 1D or 2D grids of coordinates and return a 1D ndarray
    which is converted to an `Grid2D` or `Grid2DIrregular` object.

    Parameters
    ----------
    func
        A function which computes a set of values from a 1D or 2D grid of coordinates.

    Returns
    -------
        A function that has its outputs homogenized to `Grid2D` or `Grid2DIrregular` objects.
    """

    @wraps(func)
    def wrapper(
        obj: object,
        grid: Union[np.ndarray, Grid2D, Grid2DIrregular, Grid1D],
        xp=np,
        *args,
        **kwargs,
    ) -> Union[np.ndarray, Grid2D, Grid2DIrregular, List]:
        """
        This decorator homogenizes the input of a "grid_like" 2D structure (`Grid2D`, `Grid2DIrregular` or `Grid1D`)
        into a function which outputs a grid-like structure (`Grid2D` or `Grid2DIrregular`).

        It allows these classes to be interchangeably input into a function, such that the grid is used to evaluate
        the function at every (y,x) coordinates of the grid using specific functionality of the input grid.

        The grid_like objects `Grid2D` and `Grid2DIrregular` are input into the function as a slimmed 2D ndarray array
        of shape [total_coordinates, 2] where the second dimension stores the (y,x)

        There are three types of consistent data structures and therefore decorated function mappings:

        - Uniform (`Grid2D` -> `Grid2D`): 2D structures defined on a uniform grid of data points. Both structures are
        defined according to a `Mask2D`, which the maker object ensures is passed through self consistently.

        - Irregular (`Grid2DIrregular` -> `Grid2DIrregular`: 2D structures defined on an irregular grid of data points,
        Neither structure is defined according to a mask and the maker sures the lack of a mask does not prevent the
        function from being evaluated.

        - 1D (`Grid1D` -> `Grid2D`): 1D structures defined on a 1D grid of data points. These project the 1D grid
        to a 2D grid to ensure the function can be evaluated.

        Parameters
        ----------
        obj
            An object whose function uses grid_like inputs to compute quantities at every coordinate on the grid.
        grid
            A grid_like object of coordinates on which the function values are evaluated.

        Returns
        -------
            The function values evaluated on the grid with the same structure as the input grid_like object.
        """

        return GridMaker(func=func, obj=obj, grid=grid, xp=xp, *args, **kwargs).result

    return wrapper
