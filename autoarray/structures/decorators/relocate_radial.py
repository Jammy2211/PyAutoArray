import os

from autofit.jax_wrapper import numpy as np
from functools import wraps

from typing import Union

from autoconf.exc import ConfigException

from autoarray.structures.grids.irregular_2d import Grid2DIrregular
from autoarray.structures.grids.uniform_2d import Grid2D
from autoconf import conf


def relocate_to_radial_minimum(func):
    """
    Checks whether any coordinates in the grid are radially near (0.0, 0.0), which can lead to numerical faults in
    the evaluation of a function (e.g. numerical integration reaching a singularity at (0.0, 0.0)).

    If any coordinates are radially within the radial minimum threshold, their (y,x) coordinates are shifted to that
    value to ensure they are evaluated at that coordinate.

    The value the (y,x) coordinates are rounded to is set in the 'radial_minimum.yaml' config.

    Parameters
    ----------
    func
        A function that takes a grid of coordinates which may have a singularity as (0.0, 0.0)

    Returns
    -------
        A function that has an input grid whose radial coordinates are relocated to the radial minimum.
    """

    @wraps(func)
    def wrapper(
        obj: object,
        grid: Union[np.ndarray, Grid2D, Grid2DIrregular],
        *args,
        **kwargs,
    ) -> Union[np.ndarray, Grid2D, Grid2DIrregular]:
        """
        Checks whether any coordinates in the grid are radially near (0.0, 0.0), which can lead to numerical faults in
        the evaluation of a function (e.g. numerical integration reaching a singularity at (0.0, 0.0)).

        If any coordinates are radially within the radial minimum threshold, their (y,x) coordinates are shifted to that
        value to ensure they are evaluated at that coordinate.

        The value the (y,x) coordinates are rounded to is set in the 'radial_minimum.yaml' config.

        Parameters
        ----------
        obj
            An object whose function uses grid_like inputs to compute quantities at every coordinate on the grid.
        grid
            The (y, x) coordinates which are to be radially moved from (0.0, 0.0).

        Returns
        -------
            The grid_like object whose coordinates are radially moved from (0.0, 0.0).
        """
        if os.environ.get("USE_JAX", "0") == "1":
            return func(obj, grid, *args, **kwargs)

        try:
            grid_radial_minimum = conf.instance["grids"]["radial_minimum"][
                "radial_minimum"
            ][obj.__class__.__name__]
        except KeyError as e:
            raise ConfigException(
                rf"""
                The {obj.__class__.__name__} profile you are using does not have a corresponding
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
            grid_radii = obj.radial_grid_from(grid=grid)

            grid_radial_scale = np.where(
                grid_radii < grid_radial_minimum, grid_radial_minimum / grid_radii, 1.0
            )
            moved_grid = np.multiply(grid, grid_radial_scale[:, None])

            if hasattr(grid, "with_new_array"):
                moved_grid = grid.with_new_array(moved_grid)

        moved_grid[np.isnan(np.array(moved_grid))] = grid_radial_minimum

        return func(obj, moved_grid, *args, **kwargs)

    return wrapper
