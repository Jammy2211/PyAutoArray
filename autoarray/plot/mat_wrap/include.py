from autoconf import conf
from autoarray.structures import abstract_structure, arrays, frames, grids, lines
from autoarray.inversion import mappers
from autoarray.plot.mat_wrap import visuals as vis
import typing


class AbstractInclude:
    def __init__(
        self, origin: typing.Optional[bool] = None, mask: typing.Optional[bool] = None
    ):
        """
        Sets which `Visuals` are included on a figure that is plotted using a `Plotter`.

        The `Include` object is used to extract the visuals of the plotted data structure (e.g. `Array`, `Grid`) s so they can be used in plot
        functions. Only visuals with a `True` entry in the `Include` object are extracted and therefore plotted.

        If an entry is not input into the class (e.g. it retains its default entry of `None`) then the bool is
        loaded from the `config/visualize/include.ini` config file. This means the default visuals of a project
        can be specified in a config file.

        Parameters
        ----------
        origin : bool
            If `True`, the `origin` of the plotted data structure (e.g. `Array`, `Grid`)  is included on the figure.
        mask : bool
            if `True`, the `mask` of the plotted data structure (e.g. `Array`, `Grid`)  is included on the figure.
        """

        self._origin = origin
        self._mask = mask

    def load(self, value, name):
        if value is True:
            return True
        elif value is False:
            return False
        elif value is None:
            return conf.instance["visualize"]["include"]["include"][name]

    @property
    def origin(self):
        return self.load(value=self._origin, name="origin")

    #
    @property
    def mask(self):
        return self.load(value=self._mask, name="mask")


class Include1D(AbstractInclude):
    def __init__(
        self, origin: typing.Optional[bool] = None, mask: typing.Optional[bool] = None
    ):
        """
        Sets which `Visuals1D` are included on a figure plotting 1D data that is plotted using a `Plotter1D`.

        The `Include` object is used to extract the visuals of the plotted 1D data structures so they can be used in 
        plot functions. Only visuals with a `True` entry in the `Include` object are extracted and therefore plotted.

        If an entry is not input into the class (e.g. it retains its default entry of `None`) then the bool is
        loaded from the `config/visualize/include.ini` config file. This means the default visuals of a project
        can be specified in a config file.

        Parameters
        ----------
        origin : bool
            If `True`, the `origin` of the plotted data structure (e.g. `Line`)  is included on the figure.
        mask : bool
            if `True`, the `mask` of the plotted data structure (e.g. `Line`)  is included on the figure.
        """
        super().__init__(origin=origin, mask=mask)

    def visuals_from_line(self, line: lines.Line) -> "vis.Visuals1D":

        origin = line.origin if self.origin else None
        mask = line.mask if self.mask else None

        return vis.Visuals1D(origin=origin, mask=mask)


class Include2D(AbstractInclude):
    def __init__(
        self,
        origin: typing.Optional[bool] = None,
        mask: typing.Optional[bool] = None,
        border: typing.Optional[bool] = None,
        mapper_data_pixelization_grid: typing.Optional[bool] = None,
        mapper_source_pixelization_grid: typing.Optional[bool] = None,
        mapper_source_full_grid: typing.Optional[bool] = None,
        mapper_source_border: typing.Optional[bool] = None,
        parallel_overscan: typing.Optional[bool] = None,
        serial_prescan: typing.Optional[bool] = None,
        serial_overscan: typing.Optional[bool] = None,
    ):
        """
        Sets which `Visuals2D` are included on a figure plotting 2D data that is plotted using a `Plotter2D`.

        The `Include` object is used to extract the visuals of the plotted 2D data structures so they can be used in 
        plot functions. Only visuals with a `True` entry in the `Include` object are extracted and therefore plotted.

        If an entry is not input into the class (e.g. it retains its default entry of `None`) then the bool is
        loaded from the `config/visualize/include.ini` config file. This means the default visuals of a project
        can be specified in a config file.

        Parameters
        ----------
        origin : bool
            If `True`, the `origin` of the plotted data structure (e.g. `Array`, `Grid`)  is included on the figure.
        mask : bool
            if `True`, the `mask` of the plotted data structure (e.g. `Array`, `Grid`)  is included on the figure.
        border : bool
            If `True`, the `border` of the plotted data structure (e.g. `Array`, `Grid`)  is included on the figure.
        mapper_data_pixelization_grid : bool
            If `True`, the pixelization grid in the data plane of a plotted `Mapper` is included on the figure.
        mapper_source_pixelization_grid : bool
            If `True`, the pixelization grid in the source plane of a plotted `Mapper` is included on the figure.
        mapper_source_border : bool
            If `True`, the border of the pixelization grid in the source plane of a plotted `Mapper` is included on 
            the figure.
        parallel_overscan : bool
            If `True`, the parallel overscan of a plotted `Frame` is included on the figure.
        serial_prescan : bool
            If `True`, the serial prescan of a plotted `Frame` is included on the figure.
        serial_overscan : bool
            If `True`, the serial overscan of a plotted `Frame` is included on the figure.
        """

        super().__init__(origin=origin, mask=mask)

        self._border = border
        self._mapper_data_pixelization_grid = mapper_data_pixelization_grid
        self._mapper_source_pixelization_grid = mapper_source_pixelization_grid
        self._mapper_source_full_grid = mapper_source_full_grid
        self._mapper_source_border = mapper_source_border
        self._parallel_overscan = parallel_overscan
        self._serial_prescan = serial_prescan
        self._serial_overscan = serial_overscan

    @property
    def border(self):
        return self.load(value=self._border, name="border")

    @property
    def mapper_data_pixelization_grid(self):
        return self.load(
            value=self._mapper_data_pixelization_grid,
            name="mapper_data_pixelization_grid",
        )

    @property
    def mapper_source_pixelization_grid(self):
        return self.load(
            value=self._mapper_source_pixelization_grid,
            name="mapper_source_pixelization_grid",
        )

    @property
    def mapper_source_full_grid(self):
        return self.load(
            value=self._mapper_source_full_grid, name="mapper_source_full_grid"
        )

    @property
    def mapper_source_border(self):
        return self.load(value=self._mapper_source_border, name="mapper_source_border")

    @property
    def parallel_overscan(self):
        return self.load(value=self._parallel_overscan, name="parallel_overscan")

    @property
    def serial_prescan(self):
        return self.load(value=self._serial_prescan, name="serial_prescan")

    @property
    def serial_overscan(self):
        return self.load(value=self._serial_overscan, name="serial_overscan")

    def visuals_from_structure(
        self, structure: abstract_structure.AbstractStructure
    ) -> "vis.Visuals2D":
        """
        Extracts from a `Structure` attributes that can be plotted and return them in a `Visuals` object.
        
        Only attributes with `True` entries in the `Include` object are extracted for plotting.

        From an `AbstractStructure` the following attributes can be extracted for plotting:

        - origin: the (y,x) origin of the structure's coordinate system.
        - mask: the mask of the structure.
        - border: the border of the structure's mask.

        Parameters
        ----------
        structure : abstract_structure.AbstractStructure
            The structure whose attributes are extracted for plotting.

        Returns
        -------
        vis.Visuals2D
            The collection of attributes that can be plotted by a `Plotter2D` object.

        """
        origin = grids.GridIrregular(grid=[structure.origin]) if self.origin else None

        mask = structure.mask if self.mask else None

        border = (
            structure.mask.geometry.border_grid_sub_1.in_1d_binned
            if self.border
            else None
        )

        return vis.Visuals2D(origin=origin, mask=mask, border=border)

    def visuals_from_array(self, array: arrays.Array) -> "vis.Visuals2D":
        """
        Extracts from an `Array` attributes that can be plotted and return them in a `Visuals` object.

        Only attributes with `True` entries in the `Include` object are extracted for plotting.

        From an `Array` the following attributes can be extracted for plotting:

        - origin: the (y,x) origin of the structure's coordinate system.
        - mask: the mask of the structure.
        - border: the border of the structure's mask.

        Parameters
        ----------
        array : arrays.Array
            The array whose attributes are extracted for plotting.

        Returns
        -------
        vis.Visuals2D
            The collection of attributes that can be plotted by a `Plotter2D` object.
        """
        return self.visuals_from_structure(structure=array)

    def visuals_from_grid(self, grid: grids.Grid) -> "vis.Visuals2D":
        """
        Extracts from a `Grid` attributes that can be plotted and return them in a `Visuals` object.

        Only attributes with `True` entries in the `Include` object are extracted for plotting.

        From a `Grid` the following attributes can be extracted for plotting:

        - origin: the (y,x) origin of the grid's coordinate system.
        - mask: the mask of the grid.
        - border: the border of the grid's mask.

        Parameters
        ----------
        grid : abstract_grid.AbstractGrid
            The grid whose attributes are extracted for plotting.

        Returns
        -------
        vis.Visuals2D
            The collection of attributes that can be plotted by a `Plotter2D` object.
        """
        if isinstance(grid, grids.Grid):
            return self.visuals_from_structure(structure=grid)
        return vis.Visuals2D()

    def visuals_from_frame(self, frame: frames.Frame) -> "vis.Visuals2D":
        """
        Extracts from a `Frame` attributes that can be plotted and return them in a `Visuals` object.

        Only attributes with `True` entries in the `Include` object are extracted for plotting.

        From an `Frame` the following attributes can be extracted for plotting:

        - origin: the (y,x) origin of the structure's coordinate system.
        - mask: the mask of the structure.
        - border: the border of the structure's mask.
        - parallel_overscan: the parallel overscan of the frame data.
        - serial_prescan: the serial prescan of the frame data.
        - serial_overscan: the serial overscan of the frame data.

        Parameters
        ----------
        frame : frames.Frame
            The frame whose attributes are extracted for plotting.

        Returns
        -------
        vis.Visuals2D
            The collection of attributes that can be plotted by a `Plotter2D` object.
        """
        visuals_structure = self.visuals_from_structure(structure=frame)

        parallel_overscan = (
            frame.scans.parallel_overscan if self.parallel_overscan else None
        )
        serial_prescan = frame.scans.serial_prescan if self.serial_prescan else None
        serial_overscan = frame.scans.serial_overscan if self.serial_overscan else None

        return visuals_structure + vis.Visuals2D(
            parallel_overscan=parallel_overscan,
            serial_prescan=serial_prescan,
            serial_overscan=serial_overscan,
        )

    def visuals_of_data_from_mapper(self, mapper: mappers.Mapper) -> "vis.Visuals2D":
        """
        Extracts from a `Mapper` attributes that can be plotted for figures in its data-plane (e.g. the reconstructed
        data) and return them in a `Visuals` object.

        Only attributes with `True` entries in the `Include` object are extracted for plotting.

        From a `Mapper` the following attributes can be extracted for plotting in the data-plane:

        - origin: the (y,x) origin of the `Array`'s coordinate system in the data plane.
        - mask : the `Mask` defined in the data-plane containing the data that is used by the `Mapper`.
        - mapper_data_pixelization_grid: the `Mapper`'s pixelization grid in the data-plane.
        - mapper_border_grid: the border of the `Mapper`'s full grid in the data-plane.

        Parameters
        ----------
        mapper : mappers.Mapper
            The mapper whose data-plane attributes are extracted for plotting.

        Returns
        -------
        vis.Visuals2D
            The collection of attributes that can be plotted by a `Plotter2D` object.
        """
        origin = (
            grids.GridIrregular(grid=[mapper.source_full_grid.mask.origin])
            if self.origin
            else None
        )
        mask = mapper.source_full_grid.mask if self.mask else None
        data_pixelization_grid = (
            mapper.data_pixelization_grid
            if self.mapper_data_pixelization_grid
            else None
        )

        border = (
            mapper.source_full_grid.mask.geometry.border_grid_sub_1.in_1d_binned
            if self.border
            else None
        )

        return vis.Visuals2D(
            origin=origin,
            mask=mask,
            pixelization_grid=data_pixelization_grid,
            border=border,
        )

    def visuals_of_source_from_mapper(self, mapper: mappers.Mapper) -> "vis.Visuals2D":
        """
        Extracts from a `Mapper` attributes that can be plotted for figures in its source-plane (e.g. the reconstruction
        and return them in a `Visuals` object.

        Only attributes with `True` entries in the `Include` object are extracted for plotting.

        From a `Mapper` the following attributes can be extracted for plotting in the source-plane:

        - origin: the (y,x) origin of the coordinate system in the source plane.
        - mapper_source_pixelization_grid: the `Mapper`'s pixelization grid in the source-plane.
        - mapper_source_full_grid: the `Mapper`'s full grid in the source-plane.
        - mapper_border_grid: the border of the `Mapper`'s full grid in the data-plane.

        Parameters
        ----------
        mapper : mappers.Mapper
            The mapper whose source-plane attributes are extracted for plotting.

        Returns
        -------
        vis.Visuals2D
            The collection of attributes that can be plotted by a `Plotter2D` object.
        """
        origin = (
            grids.GridIrregular(grid=[mapper.source_pixelization_grid.origin])
            if self.origin
            else None
        )
        source_full_grid = (
            mapper.source_full_grid if self.mapper_source_full_grid else None
        )
        source_pixelization_grid = (
            mapper.source_pixelization_grid
            if self.mapper_source_pixelization_grid
            else None
        )
        #     border = mapper.source_grid.sub_border_grid if self.mapper_source_pixelization_grid else None

        return vis.Visuals2D(
            origin=origin,
            grid=source_full_grid,
            pixelization_grid=source_pixelization_grid,
        )  # , border=border)
