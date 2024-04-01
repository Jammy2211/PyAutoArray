from typing import Union

from autoarray.fit.fit_imaging import FitImaging
from autoarray.inversion.pixelization.mappers.rectangular import (
    MapperRectangularNoInterp,
)
from autoarray.inversion.pixelization.mappers.voronoi import MapperVoronoiNoInterp
from autoarray.mask.mask_2d import Mask2D
from autoarray.plot.get_visuals.abstract import AbstractGetVisuals
from autoarray.plot.include.two_d import Include2D
from autoarray.plot.visuals.two_d import Visuals2D
from autoarray.structures.grids.uniform_2d import Grid2D
from autoarray.structures.grids.irregular_2d import Grid2DIrregular

from autoarray.type import Grid2DLike


class GetVisuals2D(AbstractGetVisuals):
    def __init__(self, include: Include2D, visuals: Visuals2D):
        """
        Class which gets 2D attributes and adds them to a `Visuals2D` objects, such that they are plotted on 2D figures.

        For a visual to be extracted and added for plotting, it must have a `True` value in its corresponding entry in
        the `Include2D` object. If this entry is `False`, the `GetVisuals2D.get` method returns a None and the
        attribute is omitted from the plot.

        The `GetVisuals2D` class adds new visuals to a pre-existing `Visuals2D` object that is passed to
        its `__init__` method. This only adds a new entry if the visual are not already in this object.

        Parameters
        ----------
        include
            Sets which 2D visuals are included on the figure that is to be plotted (only entries which are `True`
            are extracted via the `GetVisuals2D` object).
        visuals
            The pre-existing visuals of the plotter which new visuals are added too via the `GetVisuals2D` class.
        """
        super().__init__(include=include, visuals=visuals)

    def origin_via_mask_from(self, mask: Mask2D) -> Grid2DIrregular:
        """
        From a `Mask2D` get its origin for plotter, which is only extracted if an origin is not already
        in `self.visuals` and with `True` entries in the `Include2D` object are extracted for plotting.

        Parameters
        ----------
        mask
            The 2D mask whose origin is extracted for plotting.

        Returns
        -------
        Visuals2D
            The collection of attributes that are plotted by a `Plotter` object, which include the origin if it is
            extracted.
        """
        return self.get("origin", Grid2DIrregular(values=[mask.origin]))

    def via_mask_from(self, mask: Mask2D) -> Visuals2D:
        """
        From a `Mask2D` get its attributes that can be plotted and return them in a `Visuals2D` object.

        Only attributes not already in `self.visuals` and with `True` entries in the `Include2D` object are extracted
        for plotting.

        From a `Mask2D` the following attributes can be extracted for plotting:

        - origin: the (y,x) origin of the 2D coordinate system.
        - mask: the 2D mask.
        - border: the border of the 2D mask, which are all of the mask's exterior edge pixels.

        Parameters
        ----------
        mask
            The 2D mask whose attributes are extracted for plotting.

        Returns
        -------
        Visuals2D
            The collection of attributes that are plotted by a `Plotter` object.
        """
        origin = self.origin_via_mask_from(mask=mask)
        mask_visuals = self.get("mask", mask)
        border = self.get("border", mask.derive_grid.border)

        return self.visuals + self.visuals.__class__(
            origin=origin, mask=mask_visuals, border=border
        )

    def via_grid_from(self, grid: Grid2DLike) -> Visuals2D:
        """
        From a `Grid2D` get its attributes that can be plotted and return them in a `Visuals2D` object.

        Only attributes not already in `self.visuals` and with `True` entries in the `Include2D` object are extracted
        for plotting.

        From a `Grid2D` the following attributes can be extracted for plotting:

        - origin: the (y,x) origin of the grid's coordinate system.

        Parameters
        ----------
        grid : Grid2D
            The grid whose attributes are extracted for plotting.

        Returns
        -------
        Visuals2D
            The collection of attributes that can be plotted by a `Plotter` object.
        """
        if not isinstance(grid, Grid2D):
            return self.visuals

        origin = self.origin_via_mask_from(mask=grid.mask)

        return self.visuals + self.visuals.__class__(origin=origin)

    def via_mapper_for_data_from(
        self, mapper: Union[MapperRectangularNoInterp, MapperVoronoiNoInterp]
    ) -> Visuals2D:
        """
        From a `Mapper` get its attributes that can be plotted in the mapper's data-plane  (e.g. the reconstructed
        data) and return them in a `Visuals2D` object.

        Only attributes not already in `self.visuals` and with `True` entries in the `Include2D` object are extracted
        for plotting.

        From a `Mapper` the following attributes can be extracted for plotting in the data-plane:

        - origin: the (y,x) origin of the `Array2D`'s coordinate system in the data plane.
        - mask : the `Mask2D` defined in the data-plane containing the data that is used by the `Mapper`.
        - mapper_image_plane_mesh_grid: the `Mapper`'s pixelization's mesh in the data-plane.
        - mapper_border_grid: the border of the `Mapper`'s full grid in the data-plane.

        Parameters
        ----------
        mapper
            The mapper whose data-plane attributes are extracted for plotting.

        Returns
        -------
        Visuals2D
            The collection of attributes that can be plotted by a `Plotter` object.
        """

        visuals_via_mask = self.via_mask_from(mask=mapper.source_plane_data_grid.mask)

        mesh_grid = self.get(
            "mesh_grid", mapper.image_plane_mesh_grid, "mapper_image_plane_mesh_grid"
        )

        return (
            self.visuals
            + visuals_via_mask
            + self.visuals.__class__(mesh_grid=mesh_grid)
        )

    def via_mapper_for_source_from(
        self, mapper: Union[MapperRectangularNoInterp, MapperVoronoiNoInterp]
    ) -> Visuals2D:
        """
        From a `Mapper` get its attributes that can be plotted in the mapper's source-plane  (e.g. the reconstruction)
        and return them in a `Visuals2D` object.

        Only attributes not already in `self.visuals` and with `True` entries in the `Include2D` object are extracted
        for plotting.

        From a `Mapper` the following attributes can be extracted for plotting in the source-plane:

        - origin: the (y,x) origin of the coordinate system in the source plane.
        - mapper_source_plane_data_grid: the (y,x) grid of coordinates in the mapper's source-plane which are paired with
        the mapper's pixelization's mesh pixels.
        - mapper_source_plane_mesh_grid: the `Mapper`'s pixelization's mesh grid in the source-plane.
        - mapper_border_grid: the border of the `Mapper`'s full grid in the data-plane.

        Parameters
        ----------
        mapper
            The mapper whose source-plane attributes are extracted for plotting.

        Returns
        -------
        Visuals2D
            The collection of attributes that can be plotted by a `Plotter2D` object.
        """

        origin = self.get(
            "origin", Grid2DIrregular(values=[mapper.source_plane_mesh_grid.origin])
        )

        grid = self.get(
            "grid", mapper.source_plane_data_grid, "mapper_source_plane_data_grid"
        )

        border = self.get("border", mapper.mapper_tools.border_relocator.border_grid)

        mesh_grid = self.get(
            "mesh_grid", mapper.source_plane_mesh_grid, "mapper_source_plane_mesh_grid"
        )

        return self.visuals + self.visuals.__class__(
            origin=origin, grid=grid, border=border, mesh_grid=mesh_grid
        )

    def via_fit_imaging_from(self, fit: FitImaging) -> Visuals2D:
        """
        From a `FitImaging` get its attributes that can be plotted and return them in a `Visuals2D` object.

        Only attributes not already in `self.visuals` and with `True` entries in the `Include2D` object are extracted
        for plotting.

        From a `FitImaging` the following attributes can be extracted for plotting:

        - origin: the (y,x) origin of the 2D coordinate system.
        - mask: the 2D mask.
        - border: the border of the 2D mask, which are all of the mask's exterior edge pixels.

        Parameters
        ----------
        fit
            The fit imaging object whose attributes are extracted for plotting.

        Returns
        -------
        Visuals2D
            The collection of attributes that are plotted by a `Plotter` object.
        """
        return self.via_mask_from(mask=fit.mask)
