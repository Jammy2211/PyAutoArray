from autoarray.plot.mat_wrap.wrap import wrap_base

wrap_base.set_backend()

import matplotlib.pyplot as plt
import numpy as np

from autoarray import exc
from autoarray.structures.arrays.two_d import array_2d
from autoarray.plot.mat_wrap import wrap
from autoarray.plot.mat_wrap import visuals as vis
from autoarray.inversion import mappers

import typing


class AutoLabels:
    def __init__(
        self, title=None, ylabel=None, xlabel=None, legend=None, filename=None
    ):

        self.title = title
        self.ylabel = ylabel
        self.xlabel = xlabel
        self.legend = legend
        self.filename = filename


class AbstractMatPlot:
    def __init__(
        self,
        units: wrap.Units = wrap.Units(),
        figure: wrap.Figure = wrap.Figure(),
        axis: wrap.Axis = wrap.Axis(),
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
        Visualizes data structures (e.g an `Array2D`, `Grid2D`, `VectorField`, etc.) using Matplotlib.
        
        The `Plotter` is passed objects from the `mat_wrap` package which wrap matplotlib plot functions and customize
        the appearance of the plots of the data structure. If the values of these matplotlib wrapper objects are not
        manually specified, they assume the default values provided in the `config.visualize.mat_*` `.ini` config files.
        
        The following data structures can be plotted using the following matplotlib functions:
        
        - `Array2D`:, using `plt.imshow`.
        - `Grid2D`: using `plt.scatter`.
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
        axis : mat_wrap.Axis
            Sets the extent of the figure axis via `plt.axis` and allows for a manual axis range.
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
        self.axis = axis
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

        self.number_subplots = None
        self.subplot_index = None

    def set_for_subplot(self, is_for_subplot: bool):
        """
        Sets the `is_for_subplot` attribute for every `MatWrap` object in this `MatPlot` object by updating
        the `is_for_subplot`. By changing this tag:

            - The [subplot] section of the config file of every `MatWrap` object is used instead of [figure].
            - Calls which output or close the matplotlib figure are over-ridden so that the subplot is not removed.

        Parameters
        ----------
        is_for_subplot : bool
            The entry the `is_for_subplot` attribute of every `MatWrap` object is set too.
        """
        self.is_for_subplot = is_for_subplot
        self.output.bypass = is_for_subplot

        for attr, value in self.__dict__.items():
            if hasattr(value, "is_for_subplot"):
                value.is_for_subplot = is_for_subplot


class MatPlot1D(AbstractMatPlot):
    def __init__(
        self,
        units: wrap.Units = wrap.Units(),
        figure: wrap.Figure = wrap.Figure(),
        axis: wrap.Axis = wrap.Axis(),
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
            via `plt.close`.
        axis : mat_wrap.Axis
            Sets the extent of the figure axis via `plt.axis` and allows for a manual axis range.
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
            axis=axis,
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

        self.is_for_subplot = False

    def plot_line(
        self,
        y,
        x,
        visuals_1d: vis.Visuals1D,
        auto_labels: AutoLabels,
        plot_axis_type="semilogy",
        vertical_lines=None,
        vertical_line_labels=None,
    ):

        if y is None:
            return

        self.figure.open()
        self.title.set(auto_title=auto_labels.title)

        if x is None:
            x = np.arange(len(y))

        self.line_plot.plot_y_vs_x(
            y=y, x=x, plot_axis_type=plot_axis_type, label=auto_labels.legend
        )

        self.ylabel.set(units=self.units, include_brackets=False)
        self.xlabel.set(units=self.units, include_brackets=False)

        # self.line_plot.plot_vertical_lines(
        #     vertical_lines=vertical_lines, vertical_line_labels=vertical_line_labels
        # )

        if auto_labels.legend is not None:  # or vertical_line_labels is not None:
            self.legend.set()

        self.tickparams.set()
        self.xticks.set(
            array=None, min_value=np.min(x), max_value=np.max(x), units=self.units
        )

        self.title.set(auto_title=auto_labels.title)
        self.ylabel.set(units=self.units, auto_label=auto_labels.ylabel)
        self.xlabel.set(units=self.units, auto_label=auto_labels.xlabel)

        if not self.is_for_subplot:
            self.output.to_figure(structure=None, auto_filename=auto_labels.filename)
            self.figure.close()


class MatPlot2D(AbstractMatPlot):
    def __init__(
        self,
        units: wrap.Units = wrap.Units(),
        figure: wrap.Figure = wrap.Figure(),
        axis: wrap.Axis = wrap.Axis(),
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
        Visualizes 2D data structures (e.g an `Array2D`, `Grid2D`, `VectorField`, etc.) using Matplotlib.

        The `Plotter` is passed objects from the `mat_wrap` package which wrap matplotlib plot functions and customize
        the appearance of the plots of the data structure. If the values of these matplotlib wrapper objects are not
        manually specified, they assume the default values provided in the `config.visualize.mat_*` `.ini` config files.

        The following 2D data structures can be plotted using the following matplotlib functions:

        - `Array2D`:, using `plt.imshow`.
        - `Grid2D`: using `plt.scatter`.
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
            via `plt.close`.
        axis : mat_wrap.Axis
            Sets the extent of the figure axis via `plt.axis` and allows for a manual axis range.
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
            Overlays an input `Array2D` over the figure using `plt.imshow`.
        grid_scatter : wrappers.GridScatter
            Scatters a `Grid2D` of (y,x) coordinates over the figure using `plt.scatter`.
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
            Scatters specific (y,x) coordinates input as a `Grid2DIrregular` object over the figure.
        index_scatter : wrappers.IndexScatter
            Scatters specific coordinates of an input `Grid2D` based on input values of the `Grid2D`'s 1D or 2D indexes.
        pixelization_grid_scatter : wrappers.PixelizationGridScatter
            Scatters the `PixelizationGrid` of a `Pixelization` object.
        parallel_overscan_plot : wrappers.ParallelOverscanPlot
            Plots the parallel overscan on an `Array2D` data structure representing a CCD imaging via `plt.plot`.
        serial_prescan_plot : wrappers.SerialPrescanPlot
            Plots the serial prescan on an `Array2D` data structure representing a CCD imaging via `plt.plot`.
        serial_overscan_plot : wrappers.SerialOverscanPlot
            Plots the serial overscan on an `Array2D` data structure representing a CCD imaging via `plt.plot`.
        """

        super().__init__(
            units=units,
            figure=figure,
            axis=axis,
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

        self.is_for_subplot = False

    def get_subplot_rows_columns(self, number_subplots):
        """Get the size of a sub plotter in (total_y_pixels, total_x_pixels), based on the number of subplots that are going to be plotted.

        Parameters
        -----------
        number_subplots : int
            The number of subplots that are to be plotted in the figure.
        """
        if number_subplots <= 2:
            return 1, 2
        elif number_subplots <= 4:
            return 2, 2
        elif number_subplots <= 6:
            return 2, 3
        elif number_subplots <= 9:
            return 3, 3
        elif number_subplots <= 12:
            return 3, 4
        elif number_subplots <= 16:
            return 4, 4
        elif number_subplots <= 20:
            return 4, 5
        else:
            return 6, 6

    def setup_subplot(self, aspect=None, subplot_rows_columns=None):

        if subplot_rows_columns is None:
            rows, columns = self.get_subplot_rows_columns(
                number_subplots=self.number_subplots
            )
        else:
            rows = subplot_rows_columns[0]
            columns = subplot_rows_columns[1]

        if aspect is None:
            plt.subplot(rows, columns, self.subplot_index)
        else:
            plt.subplot(rows, columns, self.subplot_index, aspect=float(aspect))

        self.subplot_index += 1

    def plot_array(
        self,
        array: array_2d.Array2D,
        visuals_2d: vis.Visuals2D,
        auto_labels: AutoLabels,
        bypass: bool = False,
    ):
        """
        Plot an `Array2D` data structure as a figure using the matplotlib wrapper objects and tools.

        This `Array2D` is plotted using `plt.imshow`.

        Parameters
        -----------
        array : array_2d.Array2D
            The 2D array of data_type which is plotted.
        visuals_2d : vis.Visuals2D
            Contains all the visuals that are plotted over the `Array2D` (e.g. the origin, mask, grids, etc.).
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

            extent_imshow = array.extent_of_zoomed_array(buffer=buffer)
            array = array.zoomed_around_mask(buffer=buffer)

        else:

            extent_imshow = array.extent

        if not self.is_for_subplot:
            self.figure.open()
        else:
            if not bypass:
                self.setup_subplot()

        aspect = self.figure.aspect_from_shape_native(shape_native=array.shape_native)
        norm_scale = self.cmap.norm_from_array(array=array)

        plt.imshow(
            X=array.native,
            aspect=aspect,
            cmap=self.cmap.config_dict["cmap"],
            norm=norm_scale,
            extent=extent_imshow,
        )

        if visuals_2d.array_overlay is not None:
            self.array_overlay.overlay_array(
                array=visuals_2d.array_overlay, figure=self.figure
            )

        extent_axis = self.axis.config_dict.get("extent")
        extent_axis = extent_axis if extent_axis is not None else extent_imshow

        self.axis.set(extent=extent_axis)

        self.tickparams.set()

        self.yticks.set(
            array=array,
            min_value=extent_axis[2],
            max_value=extent_axis[3],
            units=self.units,
        )
        self.xticks.set(
            array=array,
            min_value=extent_axis[0],
            max_value=extent_axis[1],
            units=self.units,
        )

        self.title.set(auto_title=auto_labels.title)
        self.ylabel.set(units=self.units, include_brackets=True)
        self.xlabel.set(units=self.units, include_brackets=True)

        cb = self.colorbar.set()
        self.colorbar_tickparams.set(cb=cb)

        visuals_2d.plot_via_plotter(plotter=self, grid_indexes=array.mask.masked_grid)

        if not self.is_for_subplot and not bypass:
            self.output.to_figure(structure=array, auto_filename=auto_labels.filename)
            self.figure.close()

    def plot_frame(self, frame, visuals_2d: vis.Visuals2D, auto_labels: AutoLabels):
        """Plot an array of data_type as a figure.

        """

        if frame is None or np.all(frame == 0):
            return

        if frame.pixel_scales is None and self.units.use_scaled:
            raise exc.FrameException(
                "You cannot plot an array using its scaled unit_label if the input array does not have "
                "a pixel scales attribute."
            )

        if not self.is_for_subplot:
            self.figure.open()
        else:
            self.setup_subplot()

        aspect = self.figure.aspect_from_shape_native(shape_native=frame.shape_native)
        norm_scale = self.cmap.norm_from_array(array=frame)

        extent_imshow = frame.mask.extent

        plt.imshow(
            X=frame,
            aspect=aspect,
            cmap=self.cmap.config_dict["cmap"],
            norm=norm_scale,
            extent=extent_imshow,
        )

        extent_axis = self.axis.config_dict.get("extent")
        extent_axis = extent_axis if extent_axis is not None else extent_imshow

        self.axis.set(extent=extent_axis)

        self.tickparams.set()

        self.yticks.set(
            array=frame,
            min_value=extent_axis[2],
            max_value=extent_axis[3],
            units=self.units,
        )
        self.xticks.set(
            array=frame,
            min_value=extent_axis[0],
            max_value=extent_axis[1],
            units=self.units,
        )

        self.title.set(auto_title=auto_labels.title)
        self.ylabel.set(units=self.units, include_brackets=True)
        self.xlabel.set(units=self.units, include_brackets=True)

        cb = self.colorbar.set()
        self.colorbar_tickparams.set(cb=cb)

        visuals_2d.plot_via_plotter(plotter=self)

        if not self.is_for_subplot:
            self.output.to_figure(structure=frame, auto_filename=auto_labels.filename)
            self.figure.close()

    def plot_grid(
        self, grid, visuals_2d: vis.Visuals2D, auto_labels: AutoLabels, color_array=None
    ):
        """Plot a grid of (y,x) Cartesian coordinates as a scatter plotter of points.

        Parameters
        -----------
        grid : Grid2D
            The (y,x) coordinates of the grid, in an array of shape (total_coordinates, 2).
        indexes : []
            A set of points that are plotted in a different colour for emphasis (e.g. to show the mappings between \
            different planes).
        """

        if not self.is_for_subplot:
            self.figure.open()
        else:
            self.setup_subplot()

        if color_array is None:

            self.grid_scatter.scatter_grid(grid=grid)

        elif color_array is not None:

            plt.cm.get_cmap(self.cmap.config_dict["cmap"])
            self.grid_scatter.scatter_grid_colored(
                grid=grid, color_array=color_array, cmap=self.cmap.config_dict["cmap"]
            )
            cb = self.colorbar.set()
            self.colorbar_tickparams.set(cb=cb)

        self.title.set(auto_title=auto_labels.title)
        self.ylabel.set(units=self.units, include_brackets=True)
        self.xlabel.set(units=self.units, include_brackets=True)

        extent_axis = self.axis.config_dict.get("extent")
        extent_axis = extent_axis if extent_axis is not None else grid.extent

        self.axis.set(extent=extent_axis, grid=grid)

        self.tickparams.set()

        if not self.axis.symmetric_around_centre:
            self.yticks.set(
                array=None,
                min_value=extent_axis[2],
                max_value=extent_axis[3],
                units=self.units,
            )
            self.xticks.set(
                array=None,
                min_value=extent_axis[0],
                max_value=extent_axis[1],
                units=self.units,
            )

        visuals_2d.plot_via_plotter(plotter=self, grid_indexes=grid)

        if not self.is_for_subplot:
            self.output.to_figure(structure=grid, auto_filename=auto_labels.filename)
            self.figure.close()

    def plot_mapper(
        self,
        mapper: typing.Union[mappers.MapperRectangular, mappers.MapperVoronoi],
        visuals_2d: vis.Visuals2D,
        auto_labels: AutoLabels,
        source_pixelilzation_values=None,
    ):

        if isinstance(mapper, mappers.MapperRectangular):

            self._plot_rectangular_mapper(
                mapper=mapper,
                visuals_2d=visuals_2d,
                auto_labels=auto_labels,
                source_pixelilzation_values=source_pixelilzation_values,
            )

        else:

            self._plot_voronoi_mapper(
                mapper=mapper,
                visuals_2d=visuals_2d,
                auto_labels=auto_labels,
                source_pixelilzation_values=source_pixelilzation_values,
            )

    def _plot_rectangular_mapper(
        self,
        mapper: mappers.MapperRectangular,
        visuals_2d: vis.Visuals2D,
        auto_labels: AutoLabels,
        source_pixelilzation_values=None,
    ):

        if not self.is_for_subplot:
            self.figure.open()
        else:

            aspect_inv = self.figure.aspect_for_subplot_from_grid(
                grid=mapper.source_grid_slim
            )

            self.setup_subplot(aspect=aspect_inv)

        if source_pixelilzation_values is not None:
            self.plot_array(
                array=source_pixelilzation_values,
                visuals_2d=visuals_2d,
                auto_labels=auto_labels,
                bypass=True,
            )

        extent_axis = self.axis.config_dict.get("extent")
        extent_axis = (
            extent_axis
            if extent_axis is not None
            else mapper.source_pixelization_grid.extent
        )

        self.axis.set(extent=extent_axis, grid=mapper.source_pixelization_grid)

        self.yticks.set(
            array=None,
            min_value=extent_axis[2],
            max_value=extent_axis[3],
            units=self.units,
        )
        self.xticks.set(
            array=None,
            min_value=extent_axis[0],
            max_value=extent_axis[1],
            units=self.units,
        )

        self.grid_plot.plot_rectangular_grid_lines(
            extent=mapper.source_pixelization_grid.extent,
            shape_native=mapper.shape_native,
        )

        self.title.set(auto_title=auto_labels.title)
        self.tickparams.set()
        self.ylabel.set(units=self.units, include_brackets=True)
        self.xlabel.set(units=self.units, include_brackets=True)

        visuals_2d.plot_via_plotter(
            plotter=self, grid_indexes=mapper.source_grid_slim, mapper=mapper
        )

        if not self.is_for_subplot:
            self.output.to_figure(structure=None, auto_filename=auto_labels.filename)
            self.figure.close()

    def _plot_voronoi_mapper(
        self,
        mapper: mappers.MapperVoronoi,
        visuals_2d: vis.Visuals2D,
        auto_labels: AutoLabels,
        source_pixelilzation_values=None,
    ):

        if not self.is_for_subplot:
            self.figure.open()
        else:

            aspect_inv = self.figure.aspect_for_subplot_from_grid(
                grid=mapper.source_grid_slim
            )

            self.setup_subplot(aspect=aspect_inv)

        extent_axis = self.axis.config_dict.get("extent")
        extent_axis = (
            extent_axis
            if extent_axis is not None
            else mapper.source_pixelization_grid.extent
        )

        self.axis.set(extent=extent_axis, grid=mapper.source_pixelization_grid)

        self.tickparams.set()
        self.yticks.set(
            array=None,
            min_value=extent_axis[2],
            max_value=extent_axis[3],
            units=self.units,
        )
        self.xticks.set(
            array=None,
            min_value=extent_axis[0],
            max_value=extent_axis[1],
            units=self.units,
        )

        self.voronoi_drawer.draw_voronoi_pixels(
            mapper=mapper,
            values=source_pixelilzation_values,
            cmap=self.cmap,
            colorbar=self.colorbar,
            colorbar_tickparams=self.colorbar_tickparams,
        )

        self.title.set(auto_title=auto_labels.title)
        self.ylabel.set(units=self.units, include_brackets=True)
        self.xlabel.set(units=self.units, include_brackets=True)

        visuals_2d.plot_via_plotter(
            plotter=self, grid_indexes=mapper.source_grid_slim, mapper=mapper
        )

        if not self.is_for_subplot:
            self.output.to_figure(structure=None, auto_filename=auto_labels.filename)
            self.figure.close()
