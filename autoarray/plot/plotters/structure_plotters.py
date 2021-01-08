from autoarray.plot.plotters import abstract_plotters
from autoarray.plot.mat_wrap import visuals as vis
from autoarray.plot.mat_wrap import include as inc
from autoarray.plot.mat_wrap import mat_plot
from autoarray.structures import arrays, frames, grids, lines
from autoarray.inversion import mappers
import typing


class ArrayPlotter(abstract_plotters.AbstractPlotter):
    def __init__(
        self,
        array: arrays.Array,
        mat_plot_2d: mat_plot.MatPlot2D = mat_plot.MatPlot2D(),
        visuals_2d: vis.Visuals2D = vis.Visuals2D(),
        include_2d: inc.Include2D = inc.Include2D(),
    ):

        super().__init__(
            visuals_2d=visuals_2d, include_2d=include_2d, mat_plot_2d=mat_plot_2d
        )

        self.array = array

    @property
    def visuals_with_include_2d(self):
        """
        Extracts from an `Array` attributes that can be plotted and returns them in a `Visuals` object.

        Only attributes already in `self.visuals_2d` or with `True` entries in the `Include` object are extracted
        for plotting.

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
        return self.visuals_2d + self.visuals_2d.__class__(
            origin=self.extract_2d(
                "origin", grids.GridIrregular(grid=[self.array.origin])
            ),
            mask=self.extract_2d("mask", self.array.mask),
            border=self.extract_2d(
                "border", self.array.mask.geometry.border_grid_sub_1.in_1d_binned
            ),
        )

    @abstract_plotters.for_figure
    def figure_array(self, extent_manual=None):

        self.mat_plot_2d.plot_array(
            array=self.array,
            visuals_2d=self.visuals_with_include_2d,
            extent_manual=extent_manual,
        )


class FramePlotter(abstract_plotters.AbstractPlotter):
    def __init__(
        self,
        frame: frames.Frame,
        mat_plot_2d: mat_plot.MatPlot2D = mat_plot.MatPlot2D(),
        visuals_2d: vis.Visuals2D = vis.Visuals2D(),
        include_2d: inc.Include2D = inc.Include2D(),
    ):
        super().__init__(
            visuals_2d=visuals_2d, include_2d=include_2d, mat_plot_2d=mat_plot_2d
        )

        self.frame = frame

    @property
    def visuals_with_include_2d(self):
        """
        Extracts from a `Frame` attributes that can be plotted and return them in a `Visuals` object.

        Only attributes already in `self.visuals_2d` or with `True` entries in the `Include` object are extracted
        for plotting.

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
        return self.visuals_2d + self.visuals_2d.__class__(
            origin=self.extract_2d(
                "origin", grids.GridIrregular(grid=[self.frame.origin])
            ),
            mask=self.extract_2d("mask", self.frame.mask),
            border=self.extract_2d(
                "border", self.frame.mask.geometry.border_grid_sub_1.in_1d_binned
            ),
            parallel_overscan=self.extract_2d(
                "parallel_overscan", self.frame.scans.parallel_overscan
            ),
            serial_prescan=self.extract_2d(
                "serial_prescan", self.frame.scans.serial_prescan
            ),
            serial_overscan=self.extract_2d(
                "serial_overscan", self.frame.scans.serial_overscan
            ),
        )

    @abstract_plotters.for_figure
    def figure_frame(self):

        self.mat_plot_2d.plot_frame(
            frame=self.frame, visuals_2d=self.visuals_with_include_2d
        )


class GridPlotter(abstract_plotters.AbstractPlotter):
    def __init__(
        self,
        grid: grids.Grid,
        mat_plot_2d: mat_plot.MatPlot2D = mat_plot.MatPlot2D(),
        visuals_2d: vis.Visuals2D = vis.Visuals2D(),
        include_2d: inc.Include2D = inc.Include2D(),
    ):
        super().__init__(
            visuals_2d=visuals_2d, include_2d=include_2d, mat_plot_2d=mat_plot_2d
        )

        self.grid = grid

    @property
    def visuals_with_include_2d(self):
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
        if not isinstance(self.grid, grids.Grid):
            return self.visuals_2d

        return self.visuals_2d + self.visuals_2d.__class__(
            origin=self.extract_2d(
                "origin", grids.GridIrregular(grid=[self.grid.origin])
            )
        )

    @abstract_plotters.for_figure
    def figure_grid(
        self,
        color_array=None,
        axis_limits=None,
        indexes=None,
        symmetric_around_centre=True,
    ):

        self.mat_plot_2d.plot_grid(
            grid=self.grid,
            visuals_2d=self.visuals_with_include_2d,
            color_array=color_array,
            axis_limits=axis_limits,
            indexes=indexes,
            symmetric_around_centre=symmetric_around_centre,
        )


class MapperPlotter(abstract_plotters.AbstractPlotter):
    def __init__(
        self,
        mapper: typing.Union[mappers.MapperRectangular, mappers.MapperVoronoi],
        mat_plot_2d: mat_plot.MatPlot2D = mat_plot.MatPlot2D(),
        visuals_2d: vis.Visuals2D = vis.Visuals2D(),
        include_2d: inc.Include2D = inc.Include2D(),
    ):
        super().__init__(
            visuals_2d=visuals_2d, include_2d=include_2d, mat_plot_2d=mat_plot_2d
        )

        self.mapper = mapper

    @property
    def visuals_data_with_include_2d(self):
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
        return self.visuals_2d + self.visuals_2d.__class__(
            origin=self.extract_2d(
                "origin",
                grids.GridIrregular(grid=[self.mapper.source_full_grid.mask.origin]),
            ),
            mask=self.extract_2d("mask", self.mapper.source_full_grid.mask),
            border=self.extract_2d(
                "border",
                self.mapper.source_full_grid.mask.geometry.border_grid_sub_1.in_1d_binned,
            ),
            pixelization_grid=self.extract_2d(
                "pixelization_grid",
                self.mapper.data_pixelization_grid,
                "mapper_data_pixelization_grid",
            ),
        )

    @property
    def visuals_source_with_include_2d(self):
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

        return self.visuals_2d + self.visuals_2d.__class__(
            origin=self.extract_2d(
                "origin",
                grids.GridIrregular(grid=[self.mapper.source_pixelization_grid.origin]),
            ),
            grid=self.extract_2d(
                "grid", self.mapper.source_full_grid, "mapper_source_full_grid"
            ),
            border=self.extract_2d(
                "border", self.mapper.source_full_grid.sub_border_grid
            ),
            pixelization_grid=self.extract_2d(
                "pixelization_grid",
                self.mapper.source_pixelization_grid,
                "mapper_source_pixelization_grid",
            ),
        )

    @abstract_plotters.for_figure
    def figure_mapper(
        self,
        source_pixelilzation_values=None,
        full_indexes=None,
        pixelization_indexes=None,
    ):

        self.mat_plot_2d.plot_mapper(
            mapper=self.mapper,
            visuals_2d=self.visuals_source_with_include_2d,
            source_pixelilzation_values=source_pixelilzation_values,
            full_indexes=full_indexes,
            pixelization_indexes=pixelization_indexes,
        )

    @abstract_plotters.for_subplot
    def subplot_image_and_mapper(
        self, image, full_indexes=None, pixelization_indexes=None
    ):

        number_subplots = 2

        self.open_subplot_figure(number_subplots=number_subplots)

        self.setup_subplot(number_subplots=number_subplots, subplot_index=1)

        self.mat_plot_2d.plot_array(
            array=image, visuals_2d=self.visuals_data_with_include_2d
        )

        if full_indexes is not None:

            self.mat_plot_2d.index_scatter.scatter_grid_indexes(
                grid=self.mapper.source_full_grid.geometry.masked_grid,
                indexes=full_indexes,
            )

        if pixelization_indexes is not None:

            indexes = self.mapper.full_indexes_from_pixelization_indexes(
                pixelization_indexes=pixelization_indexes
            )

            self.mat_plot_2d.index_scatter.scatter_grid_indexes(
                grid=self.mapper.source_full_grid.geometry.masked_grid, indexes=indexes
            )

        self.setup_subplot(number_subplots=number_subplots, subplot_index=2)

        self.figure_mapper(
            full_indexes=full_indexes, pixelization_indexes=pixelization_indexes
        )

        self.mat_plot_2d.output.subplot_to_figure()
        self.mat_plot_2d.figure.close()


class LinePlotter(abstract_plotters.AbstractPlotter):
    def __init__(
        self,
        mat_plot_1d: mat_plot.MatPlot1D = mat_plot.MatPlot1D(),
        visuals_1d: vis.Visuals1D = vis.Visuals1D(),
        include_1d: inc.Include1D = inc.Include1D(),
    ):

        super().__init__(
            visuals_1d=visuals_1d, include_1d=include_1d, mat_plot_1d=mat_plot_1d
        )

    def visuals_from_line(self, line: lines.Line) -> "vis.Visuals1D":

        origin = line.origin if self.include_1d.origin else None
        mask = line.mask if self.include_1d.mask else None

        return vis.Visuals1D(origin=origin, mask=mask)

    @abstract_plotters.for_figure
    def line(
        self,
        y,
        x,
        label=None,
        plot_axis_type="semilogy",
        vertical_lines=None,
        vertical_line_labels=None,
    ):

        self.mat_plot_1d.plot_line(
            y=y,
            x=x,
            label=label,
            plot_axis_type=plot_axis_type,
            vertical_lines=vertical_lines,
            vertical_line_labels=vertical_line_labels,
        )
