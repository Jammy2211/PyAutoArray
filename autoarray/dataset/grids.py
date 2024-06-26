from typing import Optional, Union

from autoarray.structures.grids.uniform_1d import Grid1D
from autoarray.structures.grids.uniform_2d import Grid2D

from autoarray.operators.over_sampling.uniform import OverSamplingUniform
from autoarray.inversion.pixelization.border_relocator import BorderRelocator
from autoconf import cached_property



class GridsDataset:

    def __init__(self, mask, over_sampling):

        self.mask = mask
        self.over_sampling = over_sampling

    @cached_property
    def uniform(self) -> Union[Grid1D, Grid2D]:
        """
        Returns the grid of (y,x) Cartesian coordinates of every pixel in the masked data structure.

        This grid is computed based on the mask, in particular its pixel-scale and sub-grid size.

        Returns
        -------
        The (y,x) coordinates of every pixel in the data structure.
        """

        return Grid2D.from_mask(
            mask=self.mask,
            over_sampling=self.over_sampling.uniform,
        )

    @cached_property
    def non_uniform(self) -> Optional[Union[Grid1D, Grid2D]]:
        """
        Returns the grid of (y,x) Cartesian coordinates of every pixel in the masked data structure.

        This grid is computed based on the mask, in particular its pixel-scale and sub-grid size.

        Returns
        -------
        The (y,x) coordinates of every pixel in the data structure.
        """

        if self.over_sampling.non_uniform is None:
            return None

        return Grid2D.from_mask(
            mask=self.mask,
            over_sampling=self.over_sampling.non_uniform,
        )

    @cached_property
    def pixelization(self) -> Grid2D:
        """
        Returns the grid of (y,x) Cartesian coordinates of every pixel in the masked data structure which is used
        specifically for pixelization reconstructions (e.g. an `inversion`).

        This grid is computed based on the mask, in particular its pixel-scale and sub-grid size.

        A pixelization often uses a different grid of coordinates compared to the main `grid` of the data structure.
        A common example is that a pixelization may use a higher `sub_size` than the main grid, in order to better
        prevent aliasing effects.

        Returns
        -------
        The (y,x) coordinates of every pixel in the data structure, used for pixelization / inversion calculations.
        """

        over_sampling = self.over_sampling.pixelization

        if over_sampling is None:
            over_sampling = OverSamplingUniform(sub_size=4)

        return Grid2D.from_mask(
            mask=self.mask,
            over_sampling=over_sampling,
        )

    @cached_property
    def over_sampler_non_uniform(self):
        return self.non_uniform.over_sampling.over_sampler_from(mask=self.mask)

    @cached_property
    def over_sampler_pixelization(self):
        return self.pixelization.over_sampling.over_sampler_from(mask=self.mask)

    @cached_property
    def border_relocator(self) -> BorderRelocator:
        return BorderRelocator(
            mask=self.mask, sub_size=self.pixelization.over_sampling.sub_size
        )