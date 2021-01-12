from autoarray.plot.mat_wrap.wrap import wrap_base

wrap_base.set_backend()

import matplotlib.pyplot as plt
import numpy as np

from autoarray import exc
from autoarray.structures import arrays
from autoarray.plot.mat_wrap import wrap
from autoarray.plot.mat_wrap import visuals as vis
from autoarray.inversion import mappers

import typing


class AbstractMatPlot:
    def __init__(
        self,
        units: wrap.Units = wrap.Units(),
        figure: wrap.Figure = wrap.Figure(),
        cmap: wrap.Cmap = wrap.Cmap(),
        colorbar: wrap.Colorbar = wrap.Colorbar(),
        colorbar_tickparams: wrap.ColorbarTickParams = wrap.ColorbarTickParams(),
        tickparams: wrap.TickParams = wrap.TickParams(),
        yticks: wrap.YTicks = wrap.YTicks(),
        xticks: wrap.XTicks = wrap.XTicks(),
        title: wrap.Title = wrap.Title(),
        ylabel: wrap.YLabel = wrap.YLabel(),
        xlabel: wrap.XLabel = wrap.XLabel(),
        legend: wrap.Legend = wrap.Legend(),
        output: wrap.Output = wrap.Output(),
    ):
        """
        Visualizes data structures (e.g an `Array`, `Grid`, `VectorField`, etc.) using Matplotlib.
        
        The `Plotter` is passed objects from the `mat_wrap` package which wrap matplotlib plot functions and customize
        the appearance of the plots of the data structure. If the values of these matplotlib wrapper objects are not
        manually specified, they assume the default values provided in the `config.visualize.mat_*` `.ini` config files.
        
        The following data structures can be plotted using the following matplotlib functions:
        
        - `Array`:, using `plt.imshow`.
        - `Grid`: using `plt.scatter`.
        - `Line`: using `plt.plot`, `plt.semilogy`, `plt.loglog` or `plt.scatter`.
        - `VectorField`: using `plt.quiver`.
        - `RectangularMapper`: using `plt.imshow`.
        - `VoronoiMapper`: using `plt.fill`.
        
        Parameters
        ----------
        units : mat_wrap.Units
            The units of the figure used to plot the data structure which sets the y and x ticks and labels.
        figure : mat_wrap.Figure
            Opens the matplotlib figure before plotting via `plt.figure` and closes it once plotting is complete
            via `plt.close`
        cmap : mat_wrap.Cmap
            Customizes the colormap of the plot and its normalization via matplotlib `colors` objects such 
            as `colors.Normalize` and `colors.LogNorm`.
        colorbar : mat_wrap.Colorbar
            Plots the colorbar of the plot via `plt.colorbar` and customizes its tick labels and values using method
            like `cb.set_yticklabels`.
        colorbar_tickparams : mat_wrap.ColorbarTickParams
            Customizes the yticks of the colorbar plotted via `plt.colorbar`.
        tickparams : mat_wrap.TickParams
            Customizes the appearances of the y and x ticks on the plot (e.g. the fontsize) using `plt.tick_params`.
        yticks : mat_wrap.YTicks
            Sets the yticks of the plot, including scaling them to new units depending on the `Units` object, via
            `plt.yticks`.
        xticks : mat_wrap.XTicks
            Sets the xticks of the plot, including scaling them to new units depending on the `Units` object, via
            `plt.xticks`.
        title : mat_wrap.Title
            Sets the figure title and customizes its appearance using `plt.title`.        
        ylabel : mat_wrap.YLabel
            Sets the figure ylabel and customizes its appearance using `plt.ylabel`.
        xlabel : mat_wrap.XLabel
            Sets the figure xlabel and customizes its appearance using `plt.xlabel`.
        legend : mat_wrap.Legend
            Sets whether the plot inclues a legend and customizes its appearance and labels using `plt.legend`.
        output : mat_wrap.Output
            Sets if the figure is displayed on the user's screen or output to `.png` using `plt.show` and `plt.savefig`
        """

        self.units = units
        self.figure = figure
        self.cmap = cmap
        self.colorbar = colorbar
        self.colorbar_tickparams = colorbar_tickparams
        self.tickparams = tickparams
        self.title = title
        self.yticks = yticks
        self.xticks = xticks
        self.ylabel = ylabel
        self.xlabel = xlabel
        self.legend = legend
        self.output = output

    def set_axis_limits(self, axis_limits, grid, symmetric_around_centre):
        """
        Set the axis limits of the figure the grid is plotted on.

        Parameters
        -----------
        axis_limits : []
            The axis limits of the figure on which the grid is plotted, following [xmin, xmax, ymin, ymax].
        """
        if axis_limits is not None:
            plt.axis(axis_limits)
        elif symmetric_around_centre:
            ymin = np.min(grid[:, 0])
            ymax = np.max(grid[:, 0])
            xmin = np.min(grid[:, 1])
            xmax = np.max(grid[:, 1])
            x = np.max([np.abs(xmin), np.abs(xmax)])
            y = np.max([np.abs(ymin), np.abs(ymax)])
            axis_limits = [-x, x, -y, y]
            plt.axis(axis_limits)

    def set_for_subplot(self, for_subplot: bool):
        """
        Sets the `for_subplot` attribute for every `MatWrap` object in this `MatPlot` object by updating
        the `for_subplot`. By changing this tag:

            - The [subplot] section of the config file of every `MatWrap` object is used instead of [figure].
            - Calls which output or close the matplotlib figure are over-ridden so that the subplot is not removed.

        Parameters
        ----------
        for_subplot : bool
            The entry the `for_subplot` attribute of every `MatWrap` object is set too.
        """
        self.for_subplot = for_subplot
        self.output.bypass = for_subplot

        for attr, value in self.__dict__.items():
            if hasattr(value, "for_subplot"):
                value.for_subplot = for_subplot


class MatPlot1D(AbstractMatPlot):
    def __init__(
        self,
        units: wrap.Units = wrap.Units(),
        figure: wrap.Figure = wrap.Figure(),
        cmap: wrap.Cmap = wrap.Cmap(),
        colorbar: wrap.Colorbar = wrap.Colorbar(),
        colorbar_tickparams: wrap.ColorbarTickParams = wrap.ColorbarTickParams(),
        tickparams: wrap.TickParams = wrap.TickParams(),
        yticks: wrap.YTicks = wrap.YTicks(),
        xticks: wrap.XTicks = wrap.XTicks(),
        title: wrap.Title = wrap.Title(),
        ylabel: wrap.YLabel = wrap.YLabel(),
        xlabel: wrap.XLabel = wrap.XLabel(),
        legend: wrap.Legend = wrap.Legend(),
        output: wrap.Output = wrap.Output(),
        line_plot: wrap.LinePlot = wrap.LinePlot(),
    ):
        """
        Visualizes 1D data structures (e.g a `Line`, etc.) using Matplotlib.

        The `Plotter` is passed objects from the `mat_wrap` package which wrap matplotlib plot functions and customize
        the appearance of the plots of the data structure. If the values of these matplotlib wrapper objects are not
        manually specified, they assume the default values provided in the `config.visualize.mat_*` `.ini` config files.

        The following 1D data structures can be plotted using the following matplotlib functions:

        - `Line` using `plt.plot`.

        Parameters
        ----------
        units : mat_wrap.Units
            The units of the figure used to plot the data structure which sets the y and x ticks and labels.
        figure : mat_wrap.Figure
            Opens the matplotlib figure before plotting via `plt.figure` and closes it once plotting is complete
            via `plt.close`
        cmap : mat_wrap.Cmap
            Customizes the colormap of the plot and its normalization via matplotlib `colors` objects such 
            as `colors.Normalize` and `colors.LogNorm`.
        colorbar : mat_wrap.Colorbar
            Plots the colorbar of the plot via `plt.colorbar` and customizes its tick labels and values using method
            like `cb.set_yticklabels`.
        colorbar_tickparams : mat_wrap.ColorbarTickParams
            Customizes the yticks of the colorbar plotted via `plt.colorbar`.
        tickparams : mat_wrap.TickParams
            Customizes the appearances of the y and x ticks on the plot, (e.g. the fontsize), using `plt.tick_params`.
        yticks : mat_wrap.YTicks
            Sets the yticks of the plot, including scaling them to new units depending on the `Units` object, via
            `plt.yticks`.
        xticks : mat_wrap.XTicks
            Sets the xticks of the plot, including scaling them to new units depending on the `Units` object, via
            `plt.xticks`.
        title : mat_wrap.Title
            Sets the figure title and customizes its appearance using `plt.title`.        
        ylabel : mat_wrap.YLabel
            Sets the figure ylabel and customizes its appearance using `plt.ylabel`.
        xlabel : mat_wrap.XLabel
            Sets the figure xlabel and customizes its appearance using `plt.xlabel`.
        legend : mat_wrap.Legend
            Sets whether the plot inclues a legend and customizes its appearance and labels using `plt.legend`.
        output : mat_wrap.Output
            Sets if the figure is displayed on the user's screen or output to `.png` using `plt.show` and `plt.savefig`
        """

        super().__init__(
            units=units,
            figure=figure,
            cmap=cmap,
            colorbar=colorbar,
            colorbar_tickparams=colorbar_tickparams,
            tickparams=tickparams,
            yticks=yticks,
            xticks=xticks,
            title=title,
            ylabel=ylabel,
            xlabel=xlabel,
            legend=legend,
            output=output,
        )

        self.line_plot = line_plot

        self.for_subplot = False

    def plot_line(
        self,
        y,
        x,
        label=None,
        plot_axis_type="semilogy",
        vertical_lines=None,
        vertical_line_labels=None,
    ):

        if y is None:
            return

        self.figure.open()
        self.title.set()

        if x is None:
            x = np.arange(len(y))

        self.line_plot.plot_y_vs_x(y=y, x=x, plot_axis_type=plot_axis_type, label=label)

        self.ylabel.set(units=self.units, include_brackets=False)
        self.xlabel.set(units=self.units, include_brackets=False)

        self.line_plot.plot_vertical_lines(
            vertical_lines=vertical_lines, vertical_line_labels=vertical_line_labels
        )

        if label is not None or vertical_line_labels is not None:
            self.legend.set()

        self.tickparams.set()
        self.xticks.set(
            array=None, min_value=np.min(x), max_value=np.max(x), units=self.units
        )

        if not self.for_subplot:
            self.output.to_figure(structure=None)
            self.figure.close()


class MatPlot2D(AbstractMatPlot):
    def __init__(
        self,
        units: wrap.Units = wrap.Units(),
        figure: wrap.Figure = wrap.Figure(),
        cmap: wrap.Cmap = wrap.Cmap(),
        colorbar: wrap.Colorbar = wrap.Colorbar(),
        colorbar_tickparams: wrap.ColorbarTickParams = wrap.ColorbarTickParams(),
        tickparams: wrap.TickParams = wrap.TickParams(),
        yticks: wrap.YTicks = wrap.YTicks(),
        xticks: wrap.XTicks = wrap.XTicks(),
        title: wrap.Title = wrap.Title(),
        ylabel: wrap.YLabel = wrap.YLabel(),
        xlabel: wrap.XLabel = wrap.XLabel(),
        legend: wrap.Legend = wrap.Legend(),
        output: wrap.Output = wrap.Output(),
        array_overlay: wrap.ArrayOverlay = wrap.ArrayOverlay(),
        grid_scatter: wrap.GridScatter = wrap.GridScatter(),
        grid_plot: wrap.GridPlot = wrap.GridPlot(),
        vector_field_quiver: wrap.VectorFieldQuiver = wrap.VectorFieldQuiver(),
        patch_overlay: wrap.PatchOverlay = wrap.PatchOverlay(),
        voronoi_drawer: wrap.VoronoiDrawer = wrap.VoronoiDrawer(),
        origin_scatter: wrap.OriginScatter = wrap.OriginScatter(),
        mask_scatter: wrap.MaskScatter = wrap.MaskScatter(),
        border_scatter: wrap.BorderScatter = wrap.BorderScatter(),
        positions_scatter: wrap.PositionsScatter = wrap.PositionsScatter(),
        index_scatter: wrap.IndexScatter = wrap.IndexScatter(),
        pixelization_grid_scatter: wrap.PixelizationGridScatter = wrap.PixelizationGridScatter(),
        parallel_overscan_plot: wrap.ParallelOverscanPlot = wrap.ParallelOverscanPlot(),
        serial_prescan_plot: wrap.SerialPrescanPlot = wrap.SerialPrescanPlot(),
        serial_overscan_plot: wrap.SerialOverscanPlot = wrap.SerialOverscanPlot(),
    ):
        """
        Visualizes 2D data structures (e.g an `Array`, `Grid`, `VectorField`, etc.) using Matplotlib.

        The `Plotter` is passed objects from the `mat_wrap` package which wrap matplotlib plot functions and customize
        the appearance of the plots of the data structure. If the values of these matplotlib wrapper objects are not
        manually specified, they assume the default values provided in the `config.visualize.mat_*` `.ini` config files.

        The following 2D data structures can be plotted using the following matplotlib functions:

        - `Array`:, using `plt.imshow`.
        - `Grid`: using `plt.scatter`.
        - `Line`: using `plt.plot`, `plt.semilogy`, `plt.loglog` or `plt.scatter`.
        - `VectorField`: using `plt.quiver`.
        - `RectangularMapper`: using `plt.imshow`.
        - `VoronoiMapper`: using `plt.fill`.

        Parameters
        ----------
        units : mat_wrap.Units
            The units of the figure used to plot the data structure which sets the y and x ticks and labels.
        figure : mat_wrap.Figure
            Opens the matplotlib figure before plotting via `plt.figure` and closes it once plotting is complete
            via `plt.close`
        cmap : mat_wrap.Cmap
            Customizes the colormap of the plot and its normalization via matplotlib `colors` objects such 
            as `colors.Normalize` and `colors.LogNorm`.
        colorbar : mat_wrap.Colorbar
            Plots the colorbar of the plot via `plt.colorbar` and customizes its tick labels and values using method
            like `cb.set_yticklabels`.
        colorbar_tickparams : mat_wrap.ColorbarTickParams
            Customizes the yticks of the colorbar plotted via `plt.colorbar`.
        tickparams : mat_wrap.TickParams
            Customizes the appearances of the y and x ticks on the plot, (e.g. the fontsize), using `plt.tick_params`.
        yticks : mat_wrap.YTicks
            Sets the yticks of the plot, including scaling them to new units depending on the `Units` object, via
            `plt.yticks`.
        xticks : mat_wrap.XTicks
            Sets the xticks of the plot, including scaling them to new units depending on the `Units` object, via
            `plt.xticks`.
        title : mat_wrap.Title
            Sets the figure title and customizes its appearance using `plt.title`.        
        ylabel : mat_wrap.YLabel
            Sets the figure ylabel and customizes its appearance using `plt.ylabel`.
        xlabel : mat_wrap.XLabel
            Sets the figure xlabel and customizes its appearance using `plt.xlabel`.
        legend : mat_wrap.Legend
            Sets whether the plot inclues a legend and customizes its appearance and labels using `plt.legend`.
        output : mat_wrap.Output
            Sets if the figure is displayed on the user's screen or output to `.png` using `plt.show` and `plt.savefig`
        array_overlay: wrappers.ArrayOverlay
            Overlays an input `Array` over the figure using `plt.imshow`.
        grid_scatter : wrappers.GridScatter
            Scatters a `Grid` of (y,x) coordinates over the figure using `plt.scatter`.
        grid_plot: wrappers.LinePlot
            Plots lines of data (e.g. a y versus x plot via `plt.plot`, vertical lines via `plt.avxline`, etc.)
        vector_field_quiver: wrappers.VectorFieldQuiver
            Plots a `VectorField` object using the matplotlib function `plt.quiver`.
        patch_overlay: wrappers.PatchOverlay
            Overlays matplotlib `patches.Patch` objects over the figure, such as an `Ellipse`.
        voronoi_drawer: wrappers.VoronoiDrawer
            Draws a colored Voronoi mesh of pixels using `plt.fill`.
        origin_scatter : wrappers.OriginScatter
            Scatters the (y,x) origin of the data structure on the figure.
        mask_scatter : wrappers.MaskScatter
            Scatters an input `Mask2d` over the plotted data structure's figure.
        border_scatter : wrappers.BorderScatter
            Scatters the border of an input `Mask2d` over the plotted data structure's figure.
        positions_scatter : wrappers.PositionsScatter
            Scatters specific (y,x) coordinates input as a `GridIrregular` object over the figure.
        index_scatter : wrappers.IndexScatter
            Scatters specific coordinates of an input `Grid` based on input values of the `Grid`'s 1D or 2D indexes.
        pixelization_grid_scatter : wrappers.PixelizationGridScatter
            Scatters the `PixelizationGrid` of a `Pixelization` object.
        parallel_overscan_plot : wrappers.ParallelOverscanPlot
            Plots the parallel overscan on an `Array` data structure representing a CCD imaging via `plt.plot`.
        serial_prescan_plot : wrappers.SerialPrescanPlot
            Plots the serial prescan on an `Array` data structure representing a CCD imaging via `plt.plot`.
        serial_overscan_plot : wrappers.SerialOverscanPlot
            Plots the serial overscan on an `Array` data structure representing a CCD imaging via `plt.plot`.
        """

        super().__init__(
            units=units,
            figure=figure,
            cmap=cmap,
            colorbar=colorbar,
            colorbar_tickparams=colorbar_tickparams,
            tickparams=tickparams,
            yticks=yticks,
            xticks=xticks,
            title=title,
            ylabel=ylabel,
            xlabel=xlabel,
            legend=legend,
            output=output,
        )

        self.origin_scatter = origin_scatter
        self.mask_scatter = mask_scatter
        self.border_scatter = border_scatter
        self.grid_scatter = grid_scatter
        self.positions_scatter = positions_scatter
        self.index_scatter = index_scatter
        self.pixelization_grid_scatter = pixelization_grid_scatter
        self.grid_plot = grid_plot
        self.vector_field_quiver = vector_field_quiver
        self.patch_overlay = patch_overlay
        self.array_overlay = array_overlay
        self.voronoi_drawer = voronoi_drawer
        self.parallel_overscan_plot = parallel_overscan_plot
        self.serial_prescan_plot = serial_prescan_plot
        self.serial_overscan_plot = serial_overscan_plot

        self.for_subplot = False

    def plot_array(
        self,
        array: arrays.Array,
        visuals_2d: vis.Visuals2D,
        extent_manual: np.ndarray = None,
        bypass: bool = False,
    ):
        """
        Plot an `Array` data structure as a figure using the matplotlib wrapper objects and tools.

        This `Array` is plotted using `plt.imshow`.

        Parameters
        -----------
        array : arrays.Array
            The 2D array of data_type which is plotted.
        visuals_2d : vis.Visuals2D
            Contains all the visuals that are plotted over the `Array` (e.g. the origin, mask, grids, etc.).
        extent_manual : np.ndarray
            Manually specify the extent of the figure yticks and xticks using the format [xmin, xmax, ymin, ymax].
        bypass : bool
            If `True`, `plt.close` is omitted and the matplotlib figure remains open. This is used when making subplots.
        """

        if array is None or np.all(array == 0):
            return

        if array.pixel_scales is None and self.units.use_scaled:
            raise exc.ArrayException(
                "You cannot plot an array using its scaled unit_label if the input array does not have "
                "a pixel scales attribute."
            )

        array = array.binned

        if array.mask.is_all_false:
            buffer = 0
        else:
            buffer = 1

        if array.zoom_for_plot:

            extent = array.extent_of_zoomed_array(buffer=buffer)
            array = array.zoomed_around_mask(buffer=buffer)

        else:

            extent = array.extent

        self.figure.open()
        aspect = self.figure.aspect_from_shape_2d(shape_2d=array.shape_2d)
        norm_scale = self.cmap.norm_from_array(array=array)

        plt.imshow(
            X=array.in_2d,
            aspect=aspect,
            cmap=self.cmap.config_dict["cmap"],
            norm=norm_scale,
            extent=extent,
        )

        if extent_manual is not None:
            extent = extent_manual

        if visuals_2d.array_overlay is not None:
            self.array_overlay.overlay_array(
                array=visuals_2d.array_overlay, figure=self.figure
            )

        plt.axis(extent)

        self.tickparams.set()
        self.yticks.set(
            array=array, min_value=extent[2], max_value=extent[3], units=self.units
        )
        self.xticks.set(
            array=array, min_value=extent[0], max_value=extent[1], units=self.units
        )

        self.title.set()
        self.ylabel.set(units=self.units, include_brackets=True)
        self.xlabel.set(units=self.units, include_brackets=True)

        cb = self.colorbar.set()
        self.colorbar_tickparams.set(cb=cb)

        visuals_2d.plot_via_plotter(plotter=self)

        if not self.for_subplot and not bypass:
            self.output.to_figure(structure=array)
            self.figure.close()

    def plot_frame(self, frame, visuals_2d: vis.Visuals2D = vis.Visuals2D()):
        """Plot an array of data_type as a figure.

        """

        if frame is None or np.all(frame == 0):
            return

        if frame.pixel_scales is None and self.units.use_scaled:
            raise exc.FrameException(
                "You cannot plot an array using its scaled unit_label if the input array does not have "
                "a pixel scales attribute."
            )

        self.figure.open()
        aspect = self.figure.aspect_from_shape_2d(shape_2d=frame.shape_2d)
        norm_scale = self.cmap.norm_from_array(array=frame)

        plt.imshow(
            X=frame,
            aspect=aspect,
            cmap=self.cmap.config_dict["cmap"],
            norm=norm_scale,
            extent=frame.mask.geometry.extent,
        )

        plt.axis(frame.mask.geometry.extent)

        extent = frame.mask.geometry.extent

        self.tickparams.set()
        self.yticks.set(
            array=frame, min_value=extent[2], max_value=extent[3], units=self.units
        )
        self.xticks.set(
            array=frame, min_value=extent[0], max_value=extent[1], units=self.units
        )

        self.title.set()
        self.ylabel.set(units=self.units, include_brackets=True)
        self.xlabel.set(units=self.units, include_brackets=True)

        cb = self.colorbar.set()
        self.colorbar_tickparams.set(cb=cb)

        visuals_2d.plot_via_plotter(plotter=self)

        if not self.for_subplot:
            self.output.to_figure(structure=frame)
            self.figure.close()

    def plot_grid(
        self,
        grid,
        visuals_2d: vis.Visuals2D = vis.Visuals2D(),
        color_array=None,
        axis_limits=None,
        indexes=None,
        symmetric_around_centre=True,
    ):
        """Plot a grid of (y,x) Cartesian coordinates as a scatter plotter of points.

        Parameters
        -----------
        grid : Grid
            The (y,x) coordinates of the grid, in an array of shape (total_coordinates, 2).
        axis_limits : []
            The axis limits of the figure on which the grid is plotted, following [xmin, xmax, ymin, ymax].
        indexes : []
            A set of points that are plotted in a different colour for emphasis (e.g. to show the mappings between \
            different planes).
        """

        self.figure.open()

        if color_array is None:

            self.grid_scatter.scatter_grid(grid=grid)

        elif color_array is not None:

            plt.cm.get_cmap(self.cmap.config_dict["cmap"])
            self.grid_scatter.scatter_grid_colored(
                grid=grid, color_array=color_array, cmap=self.cmap.config_dict["cmap"]
            )
            cb = self.colorbar.set()
            self.colorbar_tickparams.set(cb=cb)

        self.title.set()
        self.ylabel.set(units=self.units, include_brackets=True)
        self.xlabel.set(units=self.units, include_brackets=True)

        if axis_limits is not None:

            self.set_axis_limits(
                axis_limits=axis_limits,
                grid=grid,
                symmetric_around_centre=symmetric_around_centre,
            )

        else:

            plt.axis(grid.extent)

        self.tickparams.set()
        if not symmetric_around_centre:
            self.yticks.set(
                array=None,
                min_value=grid.extent[2],
                max_value=grid.extent[3],
                units=self.units,
            )
            self.xticks.set(
                array=None,
                min_value=grid.extent[0],
                max_value=grid.extent[1],
                units=self.units,
            )

        visuals_2d.plot_via_plotter(plotter=self)

        if not self.for_subplot:
            self.output.to_figure(structure=grid)
            self.figure.close()

    def plot_mapper(
        self,
        mapper: typing.Union[mappers.MapperRectangular, mappers.MapperVoronoi],
        visuals_2d: vis.Visuals2D = vis.Visuals2D(),
        source_pixelilzation_values=None,
        full_indexes=None,
        pixelization_indexes=None,
    ):

        if isinstance(mapper, mappers.MapperRectangular):

            self._plot_rectangular_mapper(
                mapper=mapper,
                visuals_2d=visuals_2d,
                source_pixelilzation_values=source_pixelilzation_values,
                full_indexes=full_indexes,
                pixelization_indexes=pixelization_indexes,
            )

        else:

            self._plot_voronoi_mapper(
                mapper=mapper,
                visuals_2d=visuals_2d,
                source_pixelilzation_values=source_pixelilzation_values,
                full_indexes=full_indexes,
                pixelization_indexes=pixelization_indexes,
            )

    def _plot_rectangular_mapper(
        self,
        mapper: mappers.MapperRectangular,
        visuals_2d: vis.Visuals2D = vis.Visuals2D(),
        source_pixelilzation_values=None,
        full_indexes=None,
        pixelization_indexes=None,
    ):

        self.figure.open()

        if source_pixelilzation_values is not None:
            self.plot_array(
                array=source_pixelilzation_values, visuals_2d=visuals_2d, bypass=True
            )

        self.set_axis_limits(
            axis_limits=mapper.source_pixelization_grid.extent,
            grid=None,
            symmetric_around_centre=False,
        )

        self.yticks.set(
            array=None,
            min_value=mapper.source_pixelization_grid.extent[2],
            max_value=mapper.source_pixelization_grid.extent[3],
            units=self.units,
        )
        self.xticks.set(
            array=None,
            min_value=mapper.source_pixelization_grid.extent[0],
            max_value=mapper.source_pixelization_grid.extent[1],
            units=self.units,
        )

        self.grid_plot.plot_rectangular_grid_lines(
            extent=mapper.source_pixelization_grid.extent, shape_2d=mapper.shape_2d
        )

        self.title.set()
        self.tickparams.set()
        self.ylabel.set(units=self.units, include_brackets=True)
        self.xlabel.set(units=self.units, include_brackets=True)

        visuals_2d.plot_via_plotter(plotter=self)

        if full_indexes is not None:
            self.index_scatter.scatter_grid_indexes(
                grid=mapper.source_full_grid, indexes=full_indexes
            )

        if pixelization_indexes is not None:
            indexes = mapper.full_indexes_from_pixelization_indexes(
                pixelization_indexes=pixelization_indexes
            )

            self.index_scatter.scatter_grid_indexes(
                grid=mapper.source_full_grid, indexes=indexes
            )

        if not self.for_subplot:
            self.output.to_figure(structure=None)
            self.figure.close()

    def _plot_voronoi_mapper(
        self,
        mapper: mappers.MapperVoronoi,
        visuals_2d: vis.Visuals2D = vis.Visuals2D(),
        source_pixelilzation_values=None,
        full_indexes=None,
        pixelization_indexes=None,
    ):

        self.figure.open()

        self.set_axis_limits(
            axis_limits=mapper.source_pixelization_grid.extent,
            grid=None,
            symmetric_around_centre=False,
        )

        self.tickparams.set()
        self.yticks.set(
            array=None,
            min_value=mapper.source_pixelization_grid.extent[2],
            max_value=mapper.source_pixelization_grid.extent[3],
            units=self.units,
        )
        self.xticks.set(
            array=None,
            min_value=mapper.source_pixelization_grid.extent[0],
            max_value=mapper.source_pixelization_grid.extent[1],
            units=self.units,
        )

        self.voronoi_drawer.draw_voronoi_pixels(
            mapper=mapper,
            values=source_pixelilzation_values,
            cmap=self.cmap.config_dict["cmap"],
            colorbar=self.colorbar,
            colorbar_tickparams=self.colorbar_tickparams,
        )

        self.title.set()
        self.ylabel.set(units=self.units, include_brackets=True)
        self.xlabel.set(units=self.units, include_brackets=True)

        visuals_2d.plot_via_plotter(plotter=self)

        if full_indexes is not None:
            self.index_scatter.scatter_grid_indexes(
                grid=mapper.source_full_grid, indexes=full_indexes
            )

        if pixelization_indexes is not None:
            indexes = mapper.full_indexes_from_pixelization_indexes(
                pixelization_indexes=pixelization_indexes
            )

            self.index_scatter.scatter_grid_indexes(
                grid=mapper.source_full_grid, indexes=indexes
            )

        if not self.for_subplot:
            self.output.to_figure(structure=None)
            self.figure.close()
