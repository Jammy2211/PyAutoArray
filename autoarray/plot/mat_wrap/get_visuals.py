from typing import Optional, Union

from autoarray.plot.mat_wrap.include import Include1D
from autoarray.plot.mat_wrap.include import Include2D
from autoarray.plot.mat_wrap.visuals import Visuals1D
from autoarray.plot.mat_wrap.visuals import Visuals2D

from autoarray.fit.fit_imaging import FitImaging
from autoarray.inversion.mappers.rectangular import MapperRectangularNoInterp
from autoarray.inversion.mappers.voronoi import MapperVoronoiNoInterp
from autoarray.mask.mask_2d import Mask2D
from autoarray.structures.one_d.array_1d import Array1D
from autoarray.structures.two_d.grids.grid_2d import Grid2D
from autoarray.structures.two_d.grids.grid_2d_irregular import Grid2DIrregular

from autoarray.type import Grid2DLike


class AbstractGetVisuals:
    def __init__(
        self, include: Union[Include1D, Include2D], visuals: Union[Visuals1D, Visuals2D]
    ):
        """
        Class which gets attributes and adds them to a `Visuals` objects, such that they are plotted on figures.

        For a visual to be extracted and added for plotting, it must have a `True` value in its corresponding entry in
        the `Include` object. If this entry is `False`, the `GetVisuals.get` method returns a None and the attribute
        is omitted from the plot.

        The `GetVisuals` class adds new visuals to a pre-existing `Visuals` object that is passed to its `__init__`
        method. This only adds a new entry if the visual are not already in this object.

        Parameters
        ----------
        include
            Sets which visuals are included on the figure that is to be plotted (only entries which are `True`
            are extracted via the `GetVisuals` object).
        visuals
            The pre-existing visuals of the plotter which new visuals are added too via the `GetVisuals` class.
        """
        self.include = include
        self.visuals = visuals

    def get(self, name: str, value, include_name: Optional[str] = None):
        """
        Get an attribute for plotting in a `Visuals1D` object based on the following criteria:

        1) If `visuals_1d` already has a value for the attribute this is returned, over-riding the input `value` of
        that attribute.

        2) If `visuals_1d` do not contain the attribute, the input `value` is returned provided its corresponding
        entry in the `Include1D` class is `True`.

        3) If the `Include1D` entry is `False` a None is returned and the attribute is therefore not plotted.

        Parameters
        ----------
        name
            The name of the attribute which is to be extracted.
        value
            The `value` of the attribute, which is used when criteria 2 above is met.

        Returns
        -------
            The collection of attributes that can be plotted by a `Plotter` object.
        """

        if include_name is None:
            include_name = name

        if getattr(self.visuals, name) is not None:
            return getattr(self.visuals, name)
        else:
            if getattr(self.include, include_name):
                return value


class GetVisuals1D(AbstractGetVisuals):
    def __init__(self, include: Include1D, visuals: Visuals1D):
        """
        Class which gets 1D attributes and adds them to a `Visuals1D` objects, such that they are plotted on 1D figures.

        For a visual to be extracted and added for plotting, it must have a `True` value in its corresponding entry in
        the `Include1D` object. If this entry is `False`, the `GetVisuals1D.get` method returns a None and the attribute
        is omitted from the plot.

        The `GetVisuals1D` class adds new visuals to a pre-existing `Visuals1D` object that is passed to its `__init__`
        method. This only adds a new entry if the visual are not already in this object.

        Parameters
        ----------
        include
            Sets which 1D visuals are included on the figure that is to be plotted (only entries which are `True`
            are extracted via the `GetVisuals1D` object).
        visuals
            The pre-existing visuals of the plotter which new visuals are added too via the `GetVisuals1D` class.
        """
        super().__init__(include=include, visuals=visuals)

    def via_array_1d_from(self, array_1d: Array1D) -> Visuals1D:
        """
        From an `Array1D` get its attributes that can be plotted and return them in a `Visuals1D` object.

        Only attributes not already in `self.visuals` and with `True` entries in the `Include1D` object are extracted
        for plotting.

        From an `Array1D` the following attributes can be extracted for plotting:

        - origin: the (y,x) origin of the 1D array's coordinate system.
        - mask: the mask of the 1D array.

        Parameters
        ----------
        array
            The 1D array whose attributes are extracted for plotting.

        Returns
        -------
        Visuals1D
            The collection of attributes that are plotted by a `Plotter` object.
        """
        return self.visuals + self.visuals.__class__(
            origin=self.get("origin", array_1d.origin),
            mask=self.get("mask", array_1d.mask),
        )


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
        return self.get("origin", Grid2DIrregular(grid=[mask.origin]))

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
        border = self.get("border", mask.border_grid_sub_1.binned)

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
        - mapper_data_pixelization_grid: the `Mapper`'s pixelization grid in the data-plane.
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

        visuals_via_mask = self.via_mask_from(mask=mapper.source_grid_slim.mask)

        pixelization_grid = self.get(
            "pixelization_grid",
            mapper.data_pixelization_grid,
            "mapper_data_pixelization_grid",
        )

        return (
            self.visuals
            + visuals_via_mask
            + self.visuals.__class__(pixelization_grid=pixelization_grid)
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
        - mapper_source_grid_slim: the (y,x) grid of coordinates in the mapper's source-plane which are paired with
        the mapper's pixelization pixels.
        - mapper_source_pixelization_grid: the `Mapper`'s pixelization grid in the source-plane.
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
            "origin", Grid2DIrregular(grid=[mapper.source_pixelization_grid.origin])
        )

        grid = self.get("grid", mapper.source_grid_slim, "mapper_source_grid_slim")

        border = self.get("border", mapper.source_grid_slim.sub_border_grid)

        pixelization_grid = self.get(
            "pixelization_grid",
            mapper.source_pixelization_grid,
            "mapper_source_pixelization_grid",
        )

        return self.visuals + self.visuals.__class__(
            origin=origin, grid=grid, border=border, pixelization_grid=pixelization_grid
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
