import numpy as np
from functools import wraps

from autoconf import conf
from autoarray.structures import grids


def grid_like_to_structure(func):
    """ Checks whether any coordinates in the grid are radially near (0.0, 0.0), which can lead to numerical faults in \
    the evaluation of a light or mass profiles. If any coordinates are radially within the the radial minimum \
    threshold, their (y,x) coordinates are shifted to that value to ensure they are evaluated correctly.

    By default this radial minimum is not used, and users should be certain they use a value that does not impact \
    results.

    Parameters
    ----------
    func : (profile, *args, **kwargs) -> Object
        A function that takes a grid of coordinates which may have a singularity as (0.0, 0.0)

    Returns
    -------
        A function that can except cartesian or transformed coordinates
    """

    @wraps(func)
    def wrapper(profile, grid, *args, **kwargs):
        """ This decorator homogenizes the input of a "grid_like" structure (*Grid*, *GridIterate*, *GridInterpolate*
        or  *GridCoordinate*) into a function. It allows these classes to be interchangeably input into a function,
        such that the grid is used to evalaute the function as every (y,x) coordinates of the grid.

        The grid_like objects *Grid* and *GridCoordinates* are input into the function as a flattened 2D NumPy array
        of shape [total_coordinates, 2] where second dimension stores the (y,x) values. If a *GridIterate* is input,
        the function is evaluated using the appropriate iterated_*_from_func* function.

        The outputs of the function are converted from a 1D or 2D NumPy Array to an *Array*, *Grid*, *Values* or
        *GridCoordinate* objects, whichever is applicable as follows:

        - If the function returns (y,x) coordinates at every input point, the returned results are returned as a
         *Grid* or *GridCoordinates* structure - the same structure as the input.

        - If the function returns scalar values at every input point and a *Grid* is input, the returned results are
          an *Array* structure which uses the same dimensions and mask as the *Grid*.

        - If the function returns scalar values at every input point and *GridCoordinates* are input, the returned
          results are a *Values* object with structure resembling that of the *GridCoordinates*..

        If the input array is not a *Grid* structure (e.g. it is a 2D NumPy array) the output is a NumPy array.

        Parameters
        ----------
        profile : Profile
            A Profile object which uses grid_like inputs to compute quantities at every coordinate on the grid.
        grid : Grid or GridCoordinates
            A grid_like object of (y,x) coordinates on which the function values are evaluated.

        Returns
        -------
            The function values evaluated on the grid with the same structure as the input grid_like object.
        """

        if isinstance(grid, grids.GridIterate):
            return grid.iterated_result_from_func(func=func, cls=profile)
        elif isinstance(grid, grids.GridInterpolate):
            return grid.result_from_func(func=func, cls=profile)
        elif isinstance(grid, grids.GridCoordinates):
            result = func(profile, grid, *args, **kwargs)
            return grid.structure_from_result(result=result)
        elif isinstance(grid, grids.Grid):
            result = func(profile, grid, *args, **kwargs)
            return grid.structure_from_result(result=result)

        if not isinstance(grid, grids.GridCoordinates) and not isinstance(
            grid, grids.Grid
        ):
            return func(profile, grid, *args, **kwargs)

    return wrapper


def grid_like_to_structure_list(func):
    """ Checks whether any coordinates in the grid are radially near (0.0, 0.0), which can lead to numerical faults in \
    the evaluation of a light or mass profiles. If any coordinates are radially within the the radial minimum \
    threshold, their (y,x) coordinates are shifted to that value to ensure they are evaluated correctly.

    By default this radial minimum is not used, and users should be certain they use a value that does not impact \
    results.

    Parameters
    ----------
    func : (profile, *args, **kwargs) -> Object
        A function that takes a grid of coordinates which may have a singularity as (0.0, 0.0)

    Returns
    -------
        A function that can except cartesian or transformed coordinates
    """

    @wraps(func)
    def wrapper(profile, grid, *args, **kwargs):
        """ This decorator homogenizes the input of a "grid_like" structure (*Grid*, *GridIterate*, *GridInterpolate*
        or  *GridCoordinate*) into a function. It allows these classes to be interchangeably input into a function,
        such that the grid is used to evalaute the function as every (y,x) coordinates of the grid.

        The grid_like objects *Grid* and *GridCoordinates* are input into the function as a flattened 2D NumPy array
        of shape [total_coordinates, 2] where second dimension stores the (y,x) values. If a *GridIterate* is input,
        the function is evaluated using the appropriate iterated_*_from_func* function.

        If a *GridIterate* is not input the outputs of the function are converted from a list of 1D or 2D NumPy Arrays
        to a list of *Array*, *Grid*,  *Values* or  *GridCoordinate* objects, whichever is applicable as follows:

        - If the function returns (y,x) coordinates at every input point, the returned results are returned as a
         *Grid* or *GridCoordinates* structure - the same structure as the input.

        - If the function returns scalar values at every input point and a *Grid* is input, the returned results are
          an *Array* structure which uses the same dimensions and mask as the *Grid*.

        - If the function returns scalar values at every input point and *GridCoordinates* are input, the returned
          results are a *Values* object with structure resembling that of the *GridCoordinates*.

        if a *GridIterate* is input, the iterated grid calculation is not applicable. Thus, the highest resolution
        sub_size grid in the *GridIterate* is used instead.

        If the input array is not a *Grid* structure (e.g. it is a 2D NumPy array) the output is a NumPy array.

        Parameters
        ----------
        profile : Profile
            A Profile object which uses grid_like inputs to compute quantities at every coordinate on the grid.
        grid : Grid or GridCoordinates
            A grid_like object of (y,x) coordinates on which the function values are evaluated.

        Returns
        -------
            The function values evaluated on the grid with the same structure as the input grid_like object.
        """

        if isinstance(grid, grids.GridIterate):
            mask = grid.mask.mask_new_sub_size_from_mask(
                mask=grid.mask, sub_size=max(grid.sub_steps)
            )
            grid_compute = grids.Grid.from_mask(mask=mask)
            result_list = func(profile, grid_compute, *args, **kwargs)
            result_list = [
                grid_compute.structure_from_result(result=result)
                for result in result_list
            ]
            result_list = [result.in_1d_binned for result in result_list]
            return grid.grid.structure_list_from_result_list(result_list=result_list)
        elif isinstance(grid, grids.GridInterpolate):
            return func(profile, grid, *args, **kwargs)
        #     return grid.structure_list_from_result_list(result_list=result_list)
        elif isinstance(grid, grids.GridCoordinates):
            result_list = func(profile, grid, *args, **kwargs)
            return grid.structure_list_from_result_list(result_list=result_list)
        elif isinstance(grid, grids.Grid):
            result_list = func(profile, grid, *args, **kwargs)
            return grid.structure_list_from_result_list(result_list=result_list)

        if not isinstance(grid, grids.GridCoordinates) and not isinstance(
            grid, grids.Grid
        ):
            return func(profile, grid, *args, **kwargs)

    return wrapper


def transform(func):
    """Checks whether the input Grid of (y,x) coordinates have previously been transformed. If they have not \
    been transformed then they are transformed.

    Parameters
    ----------
    func : (profile, grid *args, **kwargs) -> Object
        A function where the input grid is the grid whose coordinates are transformed.

    Returns
    -------
        A function that can except cartesian or transformed coordinates
    """

    @wraps(func)
    def wrapper(cls, grid, *args, **kwargs):
        """

        Parameters
        ----------
        cls : Profile
            The class that owns the function.
        grid : grid_like
            The (y, x) coordinates in the original reference frame of the grid.

        Returns
        -------
            A grid_like object whose coordinates may be transformed.
        """

        if not isinstance(grid, (grids.GridTransformed, grids.GridTransformedNumpy)):
            result = func(
                cls, cls.transform_grid_to_reference_frame(grid), *args, **kwargs
            )

            return result

        else:
            return func(cls, grid, *args, **kwargs)

    return wrapper


def relocate_to_radial_minimum(func):
    """ Checks whether any coordinates in the grid are radially near (0.0, 0.0), which can lead to numerical faults in \
    the evaluation of a function (e.g. numerical integration reaching a singularity at (0.0, 0.0)). If any coordinates
    are radially within the the radial minimum threshold, their (y,x) coordinates are shifted to that value to ensure
    they are evaluated at that coordinate.

    The value the (y,x) coordinates are rounded to is set in the 'radial_min.ini' config.

    Parameters
    ----------
    func : (profile, *args, **kwargs) -> Object
        A function that takes a grid of coordinates which may have a singularity as (0.0, 0.0)

    Returns
    -------
        A function that can except cartesian or transformed coordinates
    """

    @wraps(func)
    def wrapper(cls, grid, *args, **kwargs):
        """

        Parameters
        ----------
        cls : Profile
            The class that owns the function.
        grid : grid_like
            The (y, x) coordinates which are to be radially moved from (0.0, 0.0).

        Returns
        -------
            The grid_like object whose coordinates are radially moved from (0.0, 0.0).
        """

        grid_radial_minimum = conf.instance.radial_min.get(
            "radial_minimum", cls.__class__.__name__, float
        )

        with np.errstate(all="ignore"):  # Division by zero fixed via isnan

            grid_radii = cls.grid_to_grid_radii(grid=grid)

            grid_radial_scale = np.where(
                grid_radii < grid_radial_minimum, grid_radial_minimum / grid_radii, 1.0
            )
            grid = np.multiply(grid, grid_radial_scale[:, None])
        grid[np.isnan(grid)] = grid_radial_minimum

        return func(cls, grid, *args, **kwargs)

    return wrapper
