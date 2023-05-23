from typing import List, Optional, Tuple, Type, Union

from autoarray.structures.grids.uniform_1d import Grid1D
from autoarray.structures.grids.uniform_2d import Grid2D
from autoarray.structures.grids.iterate_2d import Grid2DIterate
from autoarray.mask.mask_1d import Mask1D
from autoarray.mask.mask_2d import Mask2D


def grid_via_grid_class_from(
    mask: Union[Mask1D, Mask2D],
    grid_class: Union[Type[Grid1D], Type[Grid2D]],
    fractional_accuracy: float,
    relative_accuracy: Optional[float],
    sub_steps: List[int],
) -> Optional[Union[Grid1D, Grid2D, Grid2DIterate]]:
    if mask.pixel_scales is None:
        return None

    if grid_class is None:
        if isinstance(mask, Mask1D):
            grid_class = Grid1D
        elif isinstance(mask, Mask2D):
            grid_class = Grid2D

    if grid_class is Grid1D:
        return Grid1D.from_mask(mask=mask)

    if grid_class is Grid2D:
        return Grid2D.from_mask(mask=mask)

    elif grid_class is Grid2DIterate:
        return Grid2DIterate.from_mask(
            mask=mask,
            fractional_accuracy=fractional_accuracy,
            relative_accuracy=relative_accuracy,
            sub_steps=sub_steps,
        )


class AbstractSettingsDataset:
    def __init__(
        self,
        grid_class: Optional[Union[Type[Grid1D], Type[Grid2D]]] = None,
        grid_pixelization_class: Optional[Union[Type[Grid1D], Type[Grid2D]]] = None,
        sub_size: int = 1,
        sub_size_pixelization: int = 4,
        fractional_accuracy: float = 0.9999,
        relative_accuracy: Optional[float] = None,
        sub_steps: Tuple[int] = (2, 4, 8, 16),
    ):
        """
        A dataset is a collection of data structures (e.g. the data, noise-map, PSF), a mask, grid, convolver
        and other utilities that are used for modeling and fitting an image of a strong lens.

        Whilst the image, noise-map, etc. are loaded in 2D, the lens dataset creates reduced 1D arrays of each \
        for lens calculations.

        Parameters
        ----------
        grid_class
            The type of grid used to create the image from the `Galaxy` and `Plane`. The options are `Grid2D` and
            `Grid2DIterate` (see the `Grid2D` documentation for a description of these options).
        grid_pixelization_class
            The type of grid used to create the grid that maps the `Inversion` source pixels to the data's image-pixels.
            The options are `Grid2D` and `Grid2DIterate` (see the `Grid2D` documentation for a
            description of these options).
        sub_size
            If the grid and / or grid_pixelization use a `Grid2D`, this sets the sub-size used by the `Grid2D`.
        fractional_accuracy
            If the grid and / or grid_pixelization use a `Grid2DIterate`, this sets the fractional accuracy it
            uses when evaluating functions, where the fraction accuracy is the ratio of the values computed using
            two grids at a higher and lower sub-grid size.
        relative_accuracy
            If the grid and / or grid_pixelization use a `Grid2DIterate`, this sets the relative accuracy it
            uses when evaluating functions, where the relative accuracy is the absolute difference of the values
            computed using two grids at a higher and lower sub-grid size.
        sub_steps : [int]
            If the grid and / or grid_pixelization use a `Grid2DIterate`, this sets the steps the sub-size is increased by
            to meet the fractional accuracy when evaluating functions.
        """

        self.grid_class = grid_class
        self.grid_pixelization_class = grid_pixelization_class
        self.sub_size = sub_size
        self.sub_size_pixelization = sub_size_pixelization
        self.fractional_accuracy = fractional_accuracy
        self.relative_accuracy = relative_accuracy
        self.sub_steps = sub_steps

    def grid_from(self, mask) -> Union[Grid1D, Grid2D]:
        return grid_via_grid_class_from(
            mask=mask,
            grid_class=self.grid_class,
            fractional_accuracy=self.fractional_accuracy,
            relative_accuracy=self.relative_accuracy,
            sub_steps=self.sub_steps,
        )

    def grid_pixelization_from(self, mask) -> Union[Grid1D, Grid2D]:
        return grid_via_grid_class_from(
            mask=mask,
            grid_class=self.grid_pixelization_class,
            fractional_accuracy=self.fractional_accuracy,
            relative_accuracy=self.relative_accuracy,
            sub_steps=self.sub_steps,
        )
