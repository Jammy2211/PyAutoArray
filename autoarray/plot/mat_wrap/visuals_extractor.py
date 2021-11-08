from typing import Union

from autoarray.plot.mat_wrap.include import Include1D
from autoarray.plot.mat_wrap.include import Include2D
from autoarray.plot.mat_wrap.visuals import Visuals1D
from autoarray.plot.mat_wrap.visuals import Visuals2D

from autoarray.structures.grids.two_d.grid_2d import Grid2D
from autoarray.structures.grids.two_d.grid_2d_irregular import Grid2DIrregular


class AbstractExtractor:
    def __init__(
        self, include: Union[Include1D, Include2D], visuals: Union[Visuals1D, Visuals2D]
    ):

        self.include = include
        self.visuals = visuals

    def extract(self, name, value, include_name=None):
        """
        Extracts an attribute for plotting in a `Visuals1D` object based on the following criteria:

        1) If `visuals_1d` already has a value for the attribute this is returned, over-riding the input `value` of
          that attribute.

        2) If `visuals_1d` do not contain the attribute, the input `value` is returned provided its corresponding
          entry in the `Include1D` class is `True`.

        3) If the `Include1D` entry is `False` a None is returned and the attribute is therefore plotted.

        Parameters
        ----------
        name : str
            The name of the attribute which is to be extracted.
        value :
            The `value` of the attribute, which is used when criteria 2) above is met.

        Returns
        -------
            The collection of attributes that can be plotted by a `Plotter1D` object.
        """

        if include_name is None:
            include_name = name

        if getattr(self.visuals, name) is not None:
            return getattr(self.visuals, name)
        else:
            if getattr(self.include, include_name):
                return value


class VisualsExtractor1D(AbstractExtractor):
    def __init__(self, include: Include1D, visuals: Visuals1D):
        super().__init__(include=include, visuals=visuals)

    def via_array_1d_from(self, array_1d):
        return self.visuals + self.visuals.__class__(
            origin=self.extract("origin", array_1d.origin),
            mask=self.extract("mask", array_1d.mask),
        )


class VisualsExtractor2D(AbstractExtractor):
    def __init__(self, include: Include2D, visuals: Visuals2D):

        super().__init__(include=include, visuals=visuals)

    def origin_via_mask_from(self, mask):
        return self.extract("origin", Grid2DIrregular(grid=[mask.origin]))

    def via_array_1d_from(self, array_1d):

        return self.visuals + self.visuals.__class__(
            origin=self.extract("origin", array_1d.origin),
            mask=self.extract("mask", array_1d.mask),
        )

    def via_mask_from(self, mask):
        """
        Extracts from an `Array2D` attributes that can be plotted and returns them in a `Visuals` object.

        Only attributes already in `self.visuals_2d` or with `True` entries in the `Include` object are extracted
        for plotting.

        From an `Array2D` the following attributes can be extracted for plotting:

        - origin: the (y,x) origin of the structure's coordinate system.
        - mask: the mask of the structure.
        - border: the border of the structure's mask.

        Parameters
        ----------
        array : Array2D
            The array whose attributes are extracted for plotting.

        Returns
        -------
        Visuals2D
            The collection of attributes that can be plotted by a `Plotter2D` object.
        """
        origin = self.origin_via_mask_from(mask=mask)
        mask_visuals = self.extract("mask", mask)
        border = self.extract("border", mask.border_grid_sub_1.binned)

        return self.visuals + self.visuals.__class__(
            origin=origin, mask=mask_visuals, border=border
        )

    def via_grid_from(self, grid):
        """
        Extracts from a `Grid2D` attributes that can be plotted and return them in a `Visuals` object.

        Only attributes with `True` entries in the `Include` object are extracted for plotting.

        From a `Grid2D` the following attributes can be extracted for plotting:

        - origin: the (y,x) origin of the grid's coordinate system.

        Parameters
        ----------
        grid : abstract_grid_2d.AbstractGrid2D
            The grid whose attributes are extracted for plotting.

        Returns
        -------
        Visuals2D
            The collection of attributes that can be plotted by a `Plotter2D` object.
        """
        if not isinstance(grid, Grid2D):
            return self.visuals

        origin = self.origin_via_mask_from(mask=grid.mask)

        return self.visuals + self.visuals.__class__(origin=origin)

    def via_mapper_for_data_from(self, mapper):
        """
        Extracts from a `Mapper` attributes that can be plotted for figures in its data-plane (e.g. the reconstructed
        data) and return them in a `Visuals` object.

        Only attributes with `True` entries in the `Include` object are extracted for plotting.

        From a `Mapper` the following attributes can be extracted for plotting in the data-plane:

        - origin: the (y,x) origin of the `Array2D`'s coordinate system in the data plane.
        - mask : the `Mask` defined in the data-plane containing the data that is used by the `Mapper`.
        - mapper_data_pixelization_grid: the `Mapper`'s pixelization grid in the data-plane.
        - mapper_border_grid: the border of the `Mapper`'s full grid in the data-plane.

        Parameters
        ----------
        mapper : Mapper
            The mapper whose data-plane attributes are extracted for plotting.

        Returns
        -------
        Visuals2D
            The collection of attributes that can be plotted by a `Plotter2D` object.
        """

        visuals_via_mask = self.via_mask_from(mask=mapper.source_grid_slim.mask)

        pixelization_grid = self.extract(
            "pixelization_grid",
            mapper.data_pixelization_grid,
            "mapper_data_pixelization_grid",
        )

        return (
            self.visuals
            + visuals_via_mask
            + self.visuals.__class__(pixelization_grid=pixelization_grid)
        )

    def via_mapper_for_source_from(self, mapper):
        """
        Extracts from a `Mapper` attributes that can be plotted for figures in its source-plane (e.g. the reconstruction
        and return them in a `Visuals` object.

        Only attributes with `True` entries in the `Include` object are extracted for plotting.

        From a `Mapper` the following attributes can be extracted for plotting in the source-plane:

        - origin: the (y,x) origin of the coordinate system in the source plane.
        - mapper_source_pixelization_grid: the `Mapper`'s pixelization grid in the source-plane.
        - mapper_source_grid_slim: the `Mapper`'s full grid in the source-plane.
        - mapper_border_grid: the border of the `Mapper`'s full grid in the data-plane.

        Parameters
        ----------
        mapper : Mapper
            The mapper whose source-plane attributes are extracted for plotting.

        Returns
        -------
        Visuals2D
            The collection of attributes that can be plotted by a `Plotter2D` object.
        """

        origin = self.extract(
            "origin", Grid2DIrregular(grid=[mapper.source_pixelization_grid.origin])
        )

        grid = self.extract("grid", mapper.source_grid_slim, "mapper_source_grid_slim")

        border = self.extract("border", mapper.source_grid_slim.sub_border_grid)

        pixelization_grid = self.extract(
            "pixelization_grid",
            mapper.source_pixelization_grid,
            "mapper_source_pixelization_grid",
        )

        return self.visuals + self.visuals.__class__(
            origin=origin, grid=grid, border=border, pixelization_grid=pixelization_grid
        )

    def via_fit_from(self, fit):

        return self.via_mask_from(mask=fit.mask)
