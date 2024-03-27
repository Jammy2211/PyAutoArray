from typing import List, Optional, Type, Union

from autoarray.dataset.abstract.settings import AbstractSettingsDataset
from autoarray.structures.grids.uniform_1d import Grid1D
from autoarray.structures.grids.uniform_2d import Grid2D
from autoarray.operators.transformer import TransformerNUFFT


class SettingsInterferometer(AbstractSettingsDataset):
    def __init__(
        self,
        grid_class: Optional[Union[Type[Grid1D], Type[Grid2D]]] = None,
        grid_pixelization_class: Optional[Union[Type[Grid1D], Type[Grid2D]]] = None,
        sub_size: int = 1,
        sub_size_pixelization=1,
        fractional_accuracy: float = 0.9999,
        sub_steps: List[int] = None,
        transformer_class=TransformerNUFFT,
    ):
        """
          The lens dataset is the collection of data_type (image, noise-map), a mask, grid, convolver \
          and other utilities that are used for modeling and fitting an image of a strong lens.

          Whilst the image, noise-map, etc. are loaded in 2D, the lens dataset creates reduced 1D arrays of each \
          for lens calculations.

          Parameters
          ----------
        grid_class : ag.Grid2D
            The type of grid used to create the image from the `Galaxy` and `Plane`. The options are `Grid2D`,
            and `Iterator` (see the `Grid2D` documentation for a description of these options).
        grid_pixelization_class : ag.Grid2D
            The type of grid used to create the grid that maps the `Inversion` source pixels to the data's image-pixels.
            The options are `Grid2D` and `Iterator` (see the `Grid2D` documentation for a
            description of these options).
        sub_size
            If the grid and / or grid_pixelization use a `Grid2D`, this sets the sub-size used by the `Grid2D`.
        fractional_accuracy
            If the grid and / or grid_pixelization use a `Iterator`, this sets the fractional accuracy it
            uses when evaluating functions.
        sub_steps : [int]
            If the grid and / or grid_pixelization use a `Iterator`, this sets the steps the sub-size is increased by
            to meet the fractional accuracy when evaluating functions.
        """

        super().__init__(
            grid_class=grid_class,
            grid_pixelization_class=grid_pixelization_class,
            sub_size=sub_size,
            sub_size_pixelization=sub_size_pixelization,
            fractional_accuracy=fractional_accuracy,
            sub_steps=sub_steps,
        )

        self.transformer_class = transformer_class
