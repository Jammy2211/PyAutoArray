from autoarray.plot.mat_wrap.wrap import wrap_base

wrap_base.set_backend()

import matplotlib.pyplot as plt
import numpy as np
import copy
import typing

from autoarray import exc
from autoarray.plot.mat_wrap import wrap
from autoarray.plot.mat_wrap import visuals as vis
import os
from autoarray.inversion import mappers


class AbstractPlotter:
    def __init__(
        self,
        units: wrap.Units = wrap.Units(),
        figure: wrap.Figure = wrap.Figure(),
        cmap: wrap.Cmap = wrap.Cmap(),
        colorbar: wrap.Colorbar = wrap.Colorbar(),
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
            Plots the colorbar of the plot via `plt.colorbar` and customizes its appearance and ticks using methods
            like `cb.set_yticklabels` and `cb.ax.tick_params`.
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

        self.units = units
        self.figure = figure
        self.cmap = cmap
        self.colorbar = colorbar
        self.tickparams = tickparams
        self.title = title
        self.yticks = yticks
        self.xticks = xticks
        self.ylabel = ylabel
        self.xlabel = xlabel
        self.legend = legend
        self.output = output

    def plotter_for_subplot_from(self, func=None):

        plotter = copy.deepcopy(self)

        plotter.for_subplot = True
        plotter.output.bypass = True

        for attr, value in plotter.__dict__.items():
            if hasattr(value, "for_subplot"):
                value.for_subplot = True

        if func is not None:
            filename = plotter.output.filename_from_func(func=func)
            return plotter.plotter_with_new_output(filename=filename)
        else:
            return plotter

    def open_subplot_figure(self, number_subplots):
        """Setup a figure for plotting an image.

        Parameters
        -----------
        figsize : (int, int)
            The size of the figure in (total_y_pixels, total_x_pixels).
        as_subplot : bool
            If the figure is a subplot, the setup_figure function is omitted to ensure that each subplot does not create a \
            new figure and so that it can be output using the *output.output_figure(structure=None)* function.
        """
        figsize = self.get_subplot_figsize(number_subplots=number_subplots)
        plt.figure(figsize=figsize)

    def setup_subplot(
        self, number_subplots, subplot_index, aspect=None, subplot_rows_columns=None
    ):
        if subplot_rows_columns is None:
            rows, columns = self.get_subplot_rows_columns(
                number_subplots=number_subplots
            )
        else:
            rows = subplot_rows_columns[0]
            columns = subplot_rows_columns[1]
        if aspect is None:
            plt.subplot(rows, columns, subplot_index)
        else:
            plt.subplot(rows, columns, subplot_index, aspect=float(aspect))

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

    def get_subplot_figsize(self, number_subplots):
        """Get the size of a sub plotter in (total_y_pixels, total_x_pixels), based on the number of subplots that are going to be plotted.

        Parameters
        -----------
        number_subplots : int
            The number of subplots that are to be plotted in the figure.
        """

        if self.figure.config_dict_figure["figsize"] is not None:
            return self.figure.config_dict_figure["figsize"]

        if number_subplots <= 2:
            return (18, 8)
        elif number_subplots <= 4:
            return (13, 10)
        elif number_subplots <= 6:
            return (18, 12)
        elif number_subplots <= 9:
            return (25, 20)
        elif number_subplots <= 12:
            return (25, 20)
        elif number_subplots <= 16:
            return (25, 20)
        elif number_subplots <= 20:
            return (25, 20)
        else:
            return (25, 20)

    def set_axis_limits(self, axis_limits, grid, symmetric_around_centre):
        """Set the axis limits of the figure the grid is plotted on.

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

    def plotter_with_new_labels(
        self,
        title_label=None,
        title_fontsize=None,
        ylabel_units=None,
        xlabel_units=None,
        tick_params_labelsize=None,
    ):

        plotter = copy.deepcopy(self)

        plotter.title.kwargs["label"] = (
            title_label if title_label is not None else self.title.config_dict["label"]
        )
        plotter.title.kwargs["fontsize"] = (
            title_fontsize
            if title_fontsize is not None
            else self.title.config_dict["fontsize"]
        )

        plotter.ylabel._units = (
            ylabel_units if ylabel_units is not None else self.ylabel._units
        )
        plotter.xlabel._units = (
            xlabel_units if xlabel_units is not None else self.xlabel._units
        )

        plotter.tickparams.kwargs["labelsize"] = (
            tick_params_labelsize
            if tick_params_labelsize is not None
            else self.tickparams.config_dict["labelsize"]
        )

        return plotter

    def plotter_with_new_cmap(
        self, cmap=None, norm=None, vmax=None, vmin=None, linthresh=None, linscale=None
    ):

        plotter = copy.deepcopy(self)

        plotter.cmap.kwargs["cmap"] = (
            cmap if cmap is not None else self.cmap.config_dict["cmap"]
        )
        plotter.cmap.kwargs["norm"] = (
            norm if norm is not None else self.cmap.config_dict["norm"]
        )
        plotter.cmap.kwargs["vmax"] = (
            vmax if vmax is not None else self.cmap.config_dict["vmax"]
        )
        plotter.cmap.kwargs["vmin"] = (
            vmin if vmin is not None else self.cmap.config_dict["vmin"]
        )
        plotter.cmap.kwargs["linthresh"] = (
            linthresh if linthresh is not None else self.cmap.config_dict["linthresh"]
        )
        plotter.cmap.kwargs["linscale"] = (
            linscale if linscale is not None else self.cmap.config_dict["linscale"]
        )

        return plotter

    def plotter_with_new_units(
        self, use_scaled=None, conversion_factor=None, in_kpc=None
    ):

        plotter = copy.deepcopy(self)

        plotter.units.use_scaled = (
            use_scaled if use_scaled is not None else self.units.use_scaled
        )

        plotter.units.in_kpc = in_kpc if in_kpc is not None else self.units.in_kpc

        plotter.units.conversion_factor = (
            conversion_factor
            if conversion_factor is not None
            else self.units.conversion_factor
        )

        return plotter

    def plotter_with_new_output(self, path=None, filename=None, format=None):

        plotter = copy.deepcopy(self)

        plotter.output.path = path if path is not None else self.output.path

        if path is not None and path:
            try:
                os.makedirs(path)
            except FileExistsError:
                pass

        plotter.output.filename = (
            filename if filename is not None else self.output.filename
        )

        plotter.output._format = format if format is not None else self.output._format

        return plotter


class Plotter1D(AbstractPlotter):
    def __init__(
        self,
        units: wrap.Units = wrap.Units(),
        figure: wrap.Figure = wrap.Figure(),
        cmap: wrap.Cmap = wrap.Cmap(),
        colorbar: wrap.Colorbar = wrap.Colorbar(),
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
            Plots the colorbar of the plot via `plt.colorbar` and customizes its appearance and ticks using methods
            like `cb.set_yticklabels` and `cb.ax.tick_params`.
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

    def _plot_line(
        self,
        y,
        x,
        label=None,
        plot_axis_type="semilogy",
        vertical_lines=None,
        vertical_line_labels=None,
        bypass_output=False,
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

        if not bypass_output:
            self.output.to_figure(structure=None)

        if not self.for_subplot and not bypass_output:
            self.figure.close()


class Plotter2D(AbstractPlotter):
    def __init__(
        self,
        units: wrap.Units = wrap.Units(),
        figure: wrap.Figure = wrap.Figure(),
        cmap: wrap.Cmap = wrap.Cmap(),
        colorbar: wrap.Colorbar = wrap.Colorbar(),
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
        line_plot: wrap.GridPlot = wrap.GridPlot(),
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
            Plots the colorbar of the plot via `plt.colorbar` and customizes its appearance and ticks using methods
            like `cb.set_yticklabels` and `cb.ax.tick_params`.
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
        line_plot: wrappers.LinePlot
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
        self.line_plot = line_plot
        self.vector_field_quiver = vector_field_quiver
        self.patch_overlay = patch_overlay
        self.array_overlay = array_overlay
        self.voronoi_drawer = voronoi_drawer
        self.parallel_overscan_plot = parallel_overscan_plot
        self.serial_prescan_plot = serial_prescan_plot
        self.serial_overscan_plot = serial_overscan_plot

        self.for_subplot = False

    def _plot_array(self, array, visuals_2d, extent_manual=None, bypass_output=False):
        """Plot an array of data_type as a figure.

        Parameters
        -----------
        array : data_type.array.aa.Scaled
            The 2D array of data_type which is plotted.
        include : Include
            Include            
        mask : data_type.array.mask.Mask2D
            The mask applied to the array, the edge of which is plotted as a set of points over the plotted array.
        positions : [[]]
            Lists of (y,x) coordinates on the image which are plotted as colored dots, to highlight specific pixels.
        grid : data_type.array.aa.Grid
            A grid of (y,x) coordinates which may be plotted over the plotted array.

        Returns
        --------
        None
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

        self.colorbar.set()

        visuals_2d.plot_via_plotter(plotter=self)

        if not bypass_output:
            self.output.to_figure(structure=array)

        if not self.for_subplot and not bypass_output:
            self.figure.close()

    def _plot_frame(
        self, frame, visuals_2d: vis.Visuals2D = vis.Visuals2D(), bypass_output=False
    ):
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

        self.colorbar.set()

        visuals_2d.plot_via_plotter(plotter=self)

        if not bypass_output:
            self.output.to_figure(structure=frame)

        if not self.for_subplot and not bypass_output:
            self.figure.close()

    def _plot_grid(
        self,
        grid,
        visuals_2d: vis.Visuals2D = vis.Visuals2D(),
        color_array=None,
        axis_limits=None,
        indexes=None,
        symmetric_around_centre=True,
        bypass_output=False,
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
            self.colorbar.set()

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
        self.yticks.set(
            array=None,
            min_value=grid.extent[2],
            max_value=grid.extent[3],
            units=self.units,
            use_defaults=symmetric_around_centre,
        )
        self.xticks.set(
            array=None,
            min_value=grid.extent[0],
            max_value=grid.extent[1],
            units=self.units,
            use_defaults=symmetric_around_centre,
        )

        visuals_2d.plot_via_plotter(plotter=self)

        if not bypass_output:
            self.output.to_figure(structure=grid)

        if not self.for_subplot and not bypass_output:
            self.figure.close()

    def _plot_mapper(
        self,
        mapper: mappers.Mapper,
        visuals_2d: vis.Visuals2D = vis.Visuals2D(),
        source_pixelilzation_values=None,
        full_indexes=None,
        pixelization_indexes=None,
        bypass_output=False,
    ):

        if isinstance(mapper, mappers.MapperRectangular):

            self._plot_rectangular_mapper(
                mapper=mapper,
                visuals_2d=visuals_2d,
                source_pixelilzation_values=source_pixelilzation_values,
                full_indexes=full_indexes,
                pixelization_indexes=pixelization_indexes,
                bypass_output=bypass_output,
            )

        else:

            self._plot_voronoi_mapper(
                mapper=mapper,
                visuals_2d=visuals_2d,
                source_pixelilzation_values=source_pixelilzation_values,
                full_indexes=full_indexes,
                pixelization_indexes=pixelization_indexes,
                bypass_output=bypass_output,
            )

    def _plot_rectangular_mapper(
        self,
        mapper: mappers.MapperRectangular,
        visuals_2d: vis.Visuals2D = vis.Visuals2D(),
        source_pixelilzation_values=None,
        full_indexes=None,
        pixelization_indexes=None,
        bypass_output=False,
    ):

        self.figure.open()

        if source_pixelilzation_values is not None:
            self._plot_array(
                array=source_pixelilzation_values,
                visuals_2d=visuals_2d,
                bypass_output=True,
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

        self.line_plot.plot_rectangular_grid_lines(
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

        if not bypass_output:
            self.output.to_figure(structure=None)

        if not self.for_subplot and not bypass_output:
            self.figure.close()

    def _plot_voronoi_mapper(
        self,
        mapper: mappers.MapperVoronoi,
        visuals_2d: vis.Visuals2D = vis.Visuals2D(),
        source_pixelilzation_values=None,
        full_indexes=None,
        pixelization_indexes=None,
        bypass_output=False,
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

        # self.voronoi_drawer.draw_voronoi_pixels(
        #     mapper=mapper,
        #     values=source_pixelilzation_values,
        #     cmap=self.cmap.config_dict["cmap"],
        #     cb=self.colorbar,
        # )

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

        if not bypass_output:
            self.output.to_figure(structure=None)

        if not self.for_subplot and not bypass_output:
            self.figure.close()
