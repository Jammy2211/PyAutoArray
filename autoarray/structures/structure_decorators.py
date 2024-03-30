import numpy as np
from functools import wraps

from typing import List, Union

from autoconf.exc import ConfigException

from autoarray import exc
from autoarray.structures.arrays.irregular import ArrayIrregular
from autoarray.structures.arrays.uniform_1d import Array1D
from autoarray.structures.arrays.uniform_2d import Array2D
from autoarray.structures.grids.irregular_2d import Grid2DIrregular
from autoarray.structures.grids.irregular_2d import Grid2DIrregularTransformed
from autoarray.structures.grids.over_sample.uniform import OverSampleUniform
from autoarray.structures.grids.over_sample.iterate import OverSampleIterate
from autoarray.structures.grids.transformed_2d import Grid2DTransformed
from autoarray.structures.grids.transformed_2d import Grid2DTransformedNumpy
from autoarray.structures.grids.uniform_1d import Grid1D
from autoarray.structures.grids.uniform_2d import Grid2D
from autoarray.structures.vectors.irregular import VectorYX2DIrregular
from autoarray.structures.vectors.uniform import VectorYX2D
from autoconf import conf


def grid_1d_to_structure(func):
    """
    Homogenize the inputs and outputs of functions that take 2D grids of (y,x) coordinates that return the results
    as a NumPy array.

    Parameters
    ----------
    func
        A function which computes a set of values from a 2D grid of (y,x) coordinates.

    Returns
    -------
        A function that can accept cartesian or transformed coordinates
    """

    @wraps(func)
    def wrapper(
        obj: object,
        grid: Union[Grid1D, Grid2D, Grid2DIrregular],
        *args,
        **kwargs,
    ) -> Union[Array1D, ArrayIrregular]:
        """
        This decorator homogenizes the input of a "grid_like" 2D structure (`Grid2D`, `Grid2DIrregular` or `Grid1D`)
        into a function. It allows these classes to be
        interchangeably input into a function, such that the grid is used to evaluate the function at every (y,x)
        coordinates of the grid using specific functionality of the input grid.

        If the `Grid2DLike` objects `Grid2D` and `Grid2DIrregular` are input into the function as a slimmed 2D NumPy
        array of shape [total_coordinates, 2] they are projected into 1D and evaluated on this 1D grid. If the
        decorator is wrapping an object with a `centre` or `angle`, the projected give aligns the angle at a 90
        degree offset, which for an ellipse is its major-axis.

        The outputs of the function are converted from a 1D ndarray to an `Array1D` or `ArrayIrregular`,
        whichever is applicable as follows:

        - If an object where the coordinates are on a uniformly spaced grid is input (e.g. `Grid1D`, the radially
        projected grid computed from a `Grid2D`) the returns values using an `Array1D` object which assumes
        uniform spacing.

        - If an object where the coordinates are on an irregular grid is input (e.g. `Grid2DIrregular`)`the function
        returns a `ArrayIrregular` object which is also irregular.

        Parameters
        ----------
        obj
            An object whose function uses grid_like inputs to compute quantities at every coordinate on the grid.
        grid : Grid2D or Grid2DIrregular
            A grid_like object of (y,x) coordinates on which the function values are evaluated.

        Returns
        -------
            The function values evaluated on the grid with the same structure as the input grid_like object.
        """

        centre = (0.0, 0.0)

        if hasattr(obj, "centre"):
            if obj.centre is not None:
                centre = obj.centre

        angle = 0.0

        if hasattr(obj, "angle"):
            if obj.angle is not None:
                angle = obj.angle + 90.0

        if isinstance(grid, Grid2D):
            grid_2d_projected = grid.grid_2d_radial_projected_from(
                centre=centre, angle=angle
            )
            result = func(obj, grid_2d_projected, *args, **kwargs)
            return Array1D.no_mask(values=result, pixel_scales=grid.pixel_scale)

        elif isinstance(grid, Grid2DIrregular):
            result = func(obj, grid, *args, **kwargs)
            return grid.structure_2d_from(result=result)
        elif isinstance(grid, Grid1D):
            grid_2d_radial = grid.grid_2d_radial_projected_from(angle=angle)
            result = func(obj, grid_2d_radial, *args, **kwargs)
            return Array1D.no_mask(values=result, pixel_scales=grid.pixel_scale)

        raise exc.GridException(
            "You cannot input a NumPy array to a `quantity_1d_from` method."
        )

    return wrapper


def grid_1d_output_structure(func):
    """
    Homogenize the inputs and outputs of functions that take 2D grids of (y,x) coordinates that return the results
    as a NumPy array.

    Parameters
    ----------
    func
        A function which computes a set of values from a 2D grid of (y,x) coordinates.

    Returns
    -------
        A function that can accept cartesian or transformed coordinates
    """

    @wraps(func)
    def wrapper(
        obj,
        grid: Union[Grid1D, Grid2D, Grid2DIrregular],
        *args,
        **kwargs,
    ) -> Union[Array1D, ArrayIrregular]:
        """
        This decorator homogenizes the output of functions which compute a 1D result, by inspecting the output
        and converting the result to an `Array1D` object if it is uniformly spaced and a `ArrayIrregular` object if
        it is irregular. "grid_like" 2D structure (`Grid2D`),

        Parameters
        ----------
        obj
            An object whose function uses grid_like inputs to compute quantities at every coordinate on the grid.
        grid
            A grid_like object of (y,x) coordinates on which the function values are evaluated.

        Returns
        -------
            The function values evaluated on the grid with the same structure as the input grid_like object.
        """

        result = func(obj, grid, *args, **kwargs)

        if isinstance(grid, Grid2D):
            return Array1D.no_mask(values=result, pixel_scales=grid.pixel_scale)

        elif isinstance(grid, Grid2DIrregular):
            return grid.structure_2d_from(result=result)
        elif isinstance(grid, Grid1D):
            return Array1D.no_mask(values=result, pixel_scales=grid.pixel_scale)

        raise exc.GridException(
            "You cannot input a NumPy array to a `quantity_1d_from` method."
        )

    return wrapper


def grid_2d_to_structure(func):
    """
    Homogenize the inputs and outputs of functions that take 2D grids of (y,x) coordinates that return the results
    as a NumPy array.

    Parameters
    ----------
    func
        A function which computes a set of values from a 2D grid of (y,x) coordinates.

    Returns
    -------
        A function that can accept cartesian or transformed coordinates
    """

    @wraps(func)
    def wrapper(
        obj: object,
        grid: Union[np.ndarray, Grid2D, Grid2DIrregular, Grid1D],
        *args,
        **kwargs,
    ) -> Union[np.ndarray, Array2D, ArrayIrregular, Grid2D, Grid2DIrregular]:
        """
        This decorator homogenizes the input of a "grid_like" 2D structure (`Grid2D`, `Grid2DIrregular` or `Grid1D`)
        into a function.

        It allows these classes to be interchangeably input into a function, such that the grid is used to evaluate
        the function at every (y,x) coordinates of the grid using specific functionality of the input grid.

        The grid_like objects `Grid2D` and `Grid2DIrregular` are input into the function as a slimmed 2D NumPy array
        of shape [total_coordinates, 2] where the second dimension stores the (y,x)  For a `Grid2D`, the
        function is evaluated using its `OverSample` object.

        The outputs of the function are converted from a 1D or 2D NumPy Array2D to an `Array2D`, `Grid2D`,
        `ArrayIrregular` or `Grid2DIrregular` objects, whichever is applicable as follows:

        - If the function returns (y,x) coordinates at every input point, the returned results are a `Grid2D`
        or `Grid2DIrregular` structure, the same structure as the input.

        - If the function returns scalar values at every input point and a `Grid2D` is input, the returned results are
        an `Array2D` structure which uses the same dimensions and mask as the `Grid2D`.

        - If the function returns scalar values at every input point and `Grid2DIrregular` are input, the returned
        results are a `ArrayIrregular` object with structure resembling that of the `Grid2DIrregular`.

        If the input array is not a `Grid2D` structure (e.g. it is a 2D NumPy array) the output is a NumPy array.

        Parameters
        ----------
        obj
            An object whose function uses grid_like inputs to compute quantities at every coordinate on the grid.
        grid : Grid2D or Grid2DIrregular
            A grid_like object of (y,x) coordinates on which the function values are evaluated.

        Returns
        -------
            The function values evaluated on the grid with the same structure as the input grid_like object.
        """

        if isinstance(grid, Grid2D):
            result = func(obj, grid, *args, **kwargs)
            return grid.over_sample.structure_2d_from(result=result, mask=grid.mask)
        elif isinstance(grid, Grid2DIrregular):
            result = func(obj, grid, *args, **kwargs)
            return grid.structure_2d_from(result=result)
        elif isinstance(grid, Grid1D):
            grid_2d_radial = grid.grid_2d_radial_projected_from()
            result = func(obj, grid_2d_radial, *args, **kwargs)
            return grid.structure_2d_from(result=result)
        if not isinstance(grid, Grid2DIrregular) and not isinstance(grid, Grid2D):
            return func(obj, grid, *args, **kwargs)

    return wrapper


def grid_2d_to_structure_over_sample(func):
    """
    Homogenize the inputs and outputs of functions that take 2D grids of (y,x) coordinates that return the results
    as a NumPy array.

    Parameters
    ----------
    func
        A function which computes a set of values from a 2D grid of (y,x) coordinates.

    Returns
    -------
        A function that can accept cartesian or transformed coordinates
    """

    @wraps(func)
    def wrapper(
        obj: object,
        grid: Union[np.ndarray, Grid2D, Grid2DIrregular, Grid1D],
        *args,
        **kwargs,
    ) -> Union[np.ndarray, Array2D, ArrayIrregular, Grid2D, Grid2DIrregular]:
        """
        This decorator homogenizes the input of a "grid_like" 2D structure (`Grid2D`, `Grid2DIrregular` or `Grid1D`)
        into a function.

        It allows these classes to be interchangeably input into a function, such that the grid is used to evaluate
        the function at every (y,x) coordinates of the grid using specific functionality of the input grid.

        The grid_like objects `Grid2D` and `Grid2DIrregular` are input into the function as a slimmed 2D NumPy array
        of shape [total_coordinates, 2] where the second dimension stores the (y,x)  For a `Grid2D`, the
        function is evaluated using its `OverSample` object.

        The outputs of the function are converted from a 1D or 2D NumPy Array2D to an `Array2D`, `Grid2D`,
        `ArrayIrregular` or `Grid2DIrregular` objects, whichever is applicable as follows:

        - If the function returns (y,x) coordinates at every input point, the returned results are a `Grid2D`
        or `Grid2DIrregular` structure, the same structure as the input.

        - If the function returns scalar values at every input point and a `Grid2D` is input, the returned results are
        an `Array2D` structure which uses the same dimensions and mask as the `Grid2D`.

        - If the function returns scalar values at every input point and `Grid2DIrregular` are input, the returned
        results are a `ArrayIrregular` object with structure resembling that of the `Grid2DIrregular`.

        If the input array is not a `Grid2D` structure (e.g. it is a 2D NumPy array) the output is a NumPy array.

        Parameters
        ----------
        obj
            An object whose function uses grid_like inputs to compute quantities at every coordinate on the grid.
        grid : Grid2D or Grid2DIrregular
            A grid_like object of (y,x) coordinates on which the function values are evaluated.

        Returns
        -------
            The function values evaluated on the grid with the same structure as the input grid_like object.
        """

        if hasattr(grid, "over_sample"):
            if isinstance(grid.over_sample, OverSampleUniform):
                result = grid.over_sample.evaluated_func_obj_from(
                    func=func,
                    cls=obj,
                    mask=grid.mask,
                    sub_size=grid.over_sample.sub_size,
                )
                return grid.over_sample.structure_2d_from(result=result, mask=grid.mask)

            elif isinstance(grid.over_sample, OverSampleIterate):
                return grid.over_sample.iterated_result_from(
                    func=func, cls=obj, grid=grid
                )

        if isinstance(grid, Grid2DIrregular):
            result = func(obj, grid, *args, **kwargs)
            return grid.structure_2d_from(result=result)
        if isinstance(grid, Grid1D):
            grid_2d_radial = grid.grid_2d_radial_projected_from()
            result = func(obj, grid_2d_radial, *args, **kwargs)
            return grid.structure_2d_from(result=result)

        if not isinstance(grid, Grid2DIrregular) and not isinstance(grid, Grid2D):
            return func(obj, grid, *args, **kwargs)

    return wrapper


def grid_2d_to_structure_list(func):
    """
    Homogenize the inputs and outputs of functions that take 2D grids of (y,x) coordinates and return the results as
    a list of NumPy arrays.

    Parameters
    ----------
    func
        A function which computes a set of values from a 2D grid of (y,x) coordinates.

    Returns
    -------
        A function that can accept cartesian or transformed coordinates
    """

    @wraps(func)
    def wrapper(
        obj: object,
        grid: Union[np.ndarray, Grid2D, Grid2DIrregular, Grid1D],
        *args,
        **kwargs,
    ) -> List[Union[np.ndarray, Array2D, ArrayIrregular, Grid2D, Grid2DIrregular]]:
        """
        This decorator serves the same purpose as the `grid_2d_to_structure` decorator, but it deals with functions
        whose output is a list of results as opposed to a single NumPy array. It simply iterates over these lists to
        perform the same conversions as `grid_2d_to_structure`.

        Parameters
        ----------
        obj
            An object whose function uses grid_like inputs to compute quantities at every coordinate on the grid.
        grid : Grid2D or Grid2DIrregular
            A grid_like object of (y,x) coordinates on which the function values are evaluated.

        Returns
        -------
            The function values evaluated on the grid with the same structure as the input grid_like object in a list
            of NumPy arrays.
        """

        if isinstance(grid, Grid2D):
            result_list = func(obj, grid, *args, **kwargs)
            return grid.over_sample.structure_2d_list_from(
                result_list=result_list, mask=grid.mask
            )
        elif isinstance(grid, Grid2DIrregular):
            result_list = func(obj, grid, *args, **kwargs)
            return grid.structure_2d_list_from(result_list=result_list)
        elif isinstance(grid, Grid1D):
            grid_2d_radial = grid.grid_2d_radial_projected_from()
            result_list = func(obj, grid_2d_radial, *args, **kwargs)
            return grid.structure_2d_list_from(result_list=result_list)

        if not isinstance(grid, Grid2DIrregular) and not isinstance(grid, Grid2D):
            return func(obj, grid, *args, **kwargs)

    return wrapper


def grid_2d_to_structure_over_sample_list(func):
    """
    Homogenize the inputs and outputs of functions that take 2D grids of (y,x) coordinates and return the results as
    a list of NumPy arrays.

    Parameters
    ----------
    func
        A function which computes a set of values from a 2D grid of (y,x) coordinates.

    Returns
    -------
        A function that can accept cartesian or transformed coordinates
    """

    @wraps(func)
    def wrapper(
        obj: object,
        grid: Union[np.ndarray, Grid2D, Grid2DIrregular, Grid1D],
        *args,
        **kwargs,
    ) -> List[Union[np.ndarray, Array2D, ArrayIrregular, Grid2D, Grid2DIrregular]]:
        """
        This decorator serves the same purpose as the `grid_2d_to_structure` decorator, but it deals with functions
        whose output is a list of results as opposed to a single NumPy array. It simply iterates over these lists to
        perform the same conversions as `grid_2d_to_structure`.

        Parameters
        ----------
        obj
            An object whose function uses grid_like inputs to compute quantities at every coordinate on the grid.
        grid : Grid2D or Grid2DIrregular
            A grid_like object of (y,x) coordinates on which the function values are evaluated.

        Returns
        -------
            The function values evaluated on the grid with the same structure as the input grid_like object in a list
            of NumPy arrays.
        """

        if hasattr(grid, "over_sample"):
            if isinstance(grid.over_sample, OverSampleUniform):
                result_list = grid.over_sample.evaluated_func_obj_from(
                    func=func,
                    cls=obj,
                    mask=grid.mask,
                    sub_size=grid.over_sample.sub_size,
                )

                return grid.over_sample.structure_2d_list_from(
                    result_list=result_list, mask=grid.mask
                )

            if isinstance(grid.over_sample, OverSampleIterate):
                raise Exception

        if isinstance(grid, Grid2DIrregular):
            result_list = func(obj, grid, *args, **kwargs)
            return grid.structure_2d_list_from(result_list=result_list)
        elif isinstance(grid, Grid1D):
            grid_2d_radial = grid.grid_2d_radial_projected_from()
            result_list = func(obj, grid_2d_radial, *args, **kwargs)
            return grid.structure_2d_list_from(result_list=result_list)

        if not isinstance(grid, Grid2DIrregular) and not isinstance(grid, Grid2D):
            return func(obj, grid, *args, **kwargs)

    return wrapper


def grid_2d_to_vector_yx(func):
    """
    Homogenize the inputs and outputs of functions that take 2D grids of (y,x) coordinates that return the results
    as a NumPy array which represents a (y,x) 2D vectors.

    Parameters
    ----------
    func
        A function which computes (y,x) 2D vectors from a 2D grid of (y,x) coordinates.

    Returns
    -------
        A function that can accept cartesian or transformed coordinates
    """

    @wraps(func)
    def wrapper(
        obj: object,
        grid: Union[np.ndarray, Grid2D, Grid2DIrregular, Grid1D],
        *args,
        **kwargs,
    ) -> Union[np.ndarray, Array2D, ArrayIrregular, Grid2D, Grid2DIrregular]:
        """
        This decorator homogenizes the input of a "grid_like" 2D vector_yx (`Grid2D`, `Grid2DIrregular` or `Grid1D`)
        into a function.

        It allows these classes to be interchangeably input into a function, such that the grid is used to evaluate
        the function at every (y,x) coordinates of the grid using specific functionality of the input grid.

        The grid_like objects `Grid2D` and `Grid2DIrregular` are input into the function as a slimmed 2D NumPy array
        of shape [total_coordinates, 2] where the second dimension stores the (y,x)  For a `Grid2D`, the
        function is evaluated using its `OverSample` object.

        The outputs of the function are converted from a 1D or 2D NumPy Array2D to an `Array2D`, `Grid2D`,
        `ArrayIrregular` or `Grid2DIrregular` objects, whichever is applicable as follows:

        - If the function returns (y,x) coordinates at every input point, the returned results are a `Grid2D`
        or `Grid2DIrregular` vector_yx, the same vector_yx as the input.

        - If the function returns scalar values at every input point and a `Grid2D` is input, the returned results are
        an `Array2D` vector_yx which uses the same dimensions and mask as the `Grid2D`.

        - If the function returns scalar values at every input point and `Grid2DIrregular` are input, the returned
        results are a `ArrayIrregular` object with vector_yx resembling that of the `Grid2DIrregular`.

        If the input array is not a `Grid2D` vector_yx (e.g. it is a 2D NumPy array) the output is a NumPy array.

        Parameters
        ----------
        obj
            An object whose function uses grid_like inputs to compute quantities at every coordinate on the grid.
        grid : Grid2D or Grid2DIrregular
            A grid_like object of (y,x) coordinates on which the function values are evaluated.

        Returns
        -------
            The function values evaluated on the grid with the same vector_yx as the input grid_like object.
        """

        vector_yx_2d = func(obj, grid, *args, **kwargs)

        if isinstance(grid, Grid2DIrregular):
            return VectorYX2DIrregular(values=vector_yx_2d, grid=grid)

        try:
            return VectorYX2D(values=vector_yx_2d, grid=grid, mask=grid.mask)
        except AttributeError:
            return vector_yx_2d

    return wrapper


def grid_2d_to_vector_yx_list(func):
    """
    Homogenize the inputs and outputs of functions that take 2D grids of (y,x) coordinates and return the results as
    a list of NumPy arrays.

    Parameters
    ----------
    func
        A function which computes a set of values from a 2D grid of (y,x) coordinates.

    Returns
    -------
        A function that can accept cartesian or transformed coordinates
    """

    @wraps(func)
    def wrapper(
        obj: object,
        grid: Union[np.ndarray, Grid2D, Grid2DIrregular, Grid1D],
        *args,
        **kwargs,
    ) -> List[Union[np.ndarray, Array2D, ArrayIrregular, Grid2D, Grid2DIrregular]]:
        """
        This decorator serves the same purpose as the `grid_2d_to_vector_yx` decorator, but it deals with functions
        whose output is a list of results as opposed to a single NumPy array. It simply iterates over these lists to
        perform the same conversions as `grid_2d_to_vector_yx`.

        Parameters
        ----------
        obj
            An object whose function uses grid_like inputs to compute quantities at every coordinate on the grid.
        grid : Grid2D or Grid2DIrregular
            A grid_like object of (y,x) coordinates on which the function values are evaluated.

        Returns
        -------
            The function values evaluated on the grid with the same vector_yx as the input grid_like object in a list
            of NumPy arrays.
        """

        vector_yx_2d_list = func(obj, grid, *args, **kwargs)

        if isinstance(grid, Grid2DIrregular):
            return [
                VectorYX2DIrregular(values=vectors, grid=grid)
                for vectors in vector_yx_2d_list
            ]
        else:
            return [
                VectorYX2D(values=vectors, grid=grid, mask=grid.mask)
                for vectors in vector_yx_2d_list
            ]

    return wrapper


def transform(func):
    """
    Checks whether the input Grid2D of (y,x) coordinates have previously been transformed. If they have not \
    been transformed then they are transformed.

    Parameters
    ----------
    func : (profile, grid *args, **kwargs) -> Object
        A function where the input grid is the grid whose coordinates are transformed.

    Returns
    -------
        A function that can accept cartesian or transformed coordinates
    """

    @wraps(func)
    def wrapper(
        cls,
        grid: Union[
            np.ndarray,
            Grid2D,
            Grid2DIrregular,
            Grid2DTransformed,
            Grid2DTransformedNumpy,
            Grid2DIrregularTransformed,
        ],
        *args,
        **kwargs,
    ) -> Union[
        np.ndarray,
        Grid2D,
        Grid2DIrregular,
        Grid2DTransformed,
        Grid2DTransformedNumpy,
        Grid2DIrregularTransformed,
    ]:
        """

        Parameters
        ----------
        cls : Profile
            The class that owns the function.
        grid
            The (y, x) coordinates in the original reference frame of the grid.

        Returns
        -------
            A grid_like object whose coordinates may be transformed.
        """

        if not isinstance(
            grid,
            (Grid2DTransformed, Grid2DTransformedNumpy, Grid2DIrregularTransformed),
        ):
            result = func(
                cls, cls.transformed_to_reference_frame_grid_from(grid), *args, **kwargs
            )

            return result

        else:
            return func(cls, grid, *args, **kwargs)

    return wrapper


def relocate_to_radial_minimum(func):
    """ Checks whether any coordinates in the grid are radially near (0.0, 0.0), which can lead to numerical faults in \
    the evaluation of a function (e.g. numerical integration reaching a singularity at (0.0, 0.0)). If any coordinates
    are radially within the radial minimum threshold, their (y,x) coordinates are shifted to that value to ensure
    they are evaluated at that coordinate.

    The value the (y,x) coordinates are rounded to is set in the 'radial_min.ini' config.

    Parameters
    ----------
    func : (profile, *args, **kwargs) -> Object
        A function that takes a grid of coordinates which may have a singularity as (0.0, 0.0)

    Returns
    -------
        A function that can accept cartesian or transformed coordinates
    """

    @wraps(func)
    def wrapper(
        cls,
        grid: Union[np.ndarray, Grid2D, Grid2DIrregular],
        *args,
        **kwargs,
    ) -> Union[np.ndarray, Grid2D, Grid2DIrregular]:
        """

        Parameters
        ----------
        cls : Profile
            The class that owns the function.
        grid
            The (y, x) coordinates which are to be radially moved from (0.0, 0.0).

        Returns
        -------
            The grid_like object whose coordinates are radially moved from (0.0, 0.0).
        """

        try:
            grid_radial_minimum = conf.instance["grids"]["radial_minimum"][
                "radial_minimum"
            ][cls.__class__.__name__]
        except KeyError as e:
            raise ConfigException(
                rf"""
                The {cls.__class__.__name__} profile you are using does not have a corresponding
                entry in the `config/grid.yaml` config file.
                
                When a profile is evaluated at (0.0, 0.0), they commonly break due to numericalinstabilities (e.g. 
                division by zero). To prevent this, the code relocates the (y,x) coordinates of the grid to a 
                minimum radial value, specified in the `config/grids.yaml` config file.
                
                For example, if the value in `grid.yaml` is `radial_minimum: 1e-6`, then any (y,x) coordinates
                with a radial distance less than 1e-6 to (0.0, 0.0) are relocated to 1e-6.
                
                For a profile to be used it must have an entry in the `config/grids.yaml` config file. Go to this
                file now and add your profile to the `radial_minimum` section. Adopting a value of 1e-6 is a good
                default choice.
                
                If you are going to make a pull request to add your profile to the source code, you should also
                add an entry to the `config/grids.yaml` config file of the source code itself
                (e.g. `PyAutoGalaxy/autogalaxy/config/grids.yaml`).
                """
            )

        with np.errstate(all="ignore"):  # Division by zero fixed via isnan
            grid_radii = cls.radial_grid_from(grid=grid)

            grid_radial_scale = np.where(
                grid_radii < grid_radial_minimum, grid_radial_minimum / grid_radii, 1.0
            )
            moved_grid = np.multiply(grid, grid_radial_scale[:, None])

            if hasattr(grid, "with_new_array"):
                moved_grid = grid.with_new_array(moved_grid)

        moved_grid[np.isnan(np.array(moved_grid))] = grid_radial_minimum

        return func(cls, moved_grid, *args, **kwargs)

    return wrapper
