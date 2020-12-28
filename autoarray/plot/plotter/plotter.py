import matplotlib

from autoconf import conf

backend = conf.get_matplotlib_backend()

if backend not in "default":
    matplotlib.use(backend)

try:
    hpc_mode = conf.instance["general"]["hpc"]["hpc_mode"]
except KeyError:
    hpc_mode = False

if hpc_mode:
    matplotlib.use("Agg")

import matplotlib.pyplot as plt

import numpy as np
from functools import wraps
import copy

from autoarray import exc
from autoarray.plot.mat_wrap import mat_base, mat_structure, mat_obj
import inspect
import os
from autoarray.plot.plotter import include as inc
from autoarray.inversion import mappers


class Plotter:
    def __init__(
        self,
        units: mat_base.Units = mat_base.Units(),
        figure: mat_base.Figure = mat_base.Figure(),
        cmap: mat_base.Cmap = mat_base.Cmap(),
        colorbar: mat_base.Colorbar = mat_base.Colorbar(),
        tickparams: mat_base.TickParams = mat_base.TickParams(),
        yticks: mat_base.YTicks = mat_base.YTicks(),
        xticks: mat_base.XTicks = mat_base.XTicks(),
        title: mat_base.Title = mat_base.Title(),
        ylabel: mat_base.YLabel = mat_base.YLabel(),
        xlabel: mat_base.XLabel = mat_base.XLabel(),
        legend: mat_base.Legend = mat_base.Legend(),
        output: mat_base.Output = mat_base.Output(),
        array_overlay: mat_structure.ArrayOverlay = mat_structure.ArrayOverlay(),
        grid_scatter: mat_structure.GridScatter = mat_structure.GridScatter(),
        line_plot: mat_structure.LinePlot = mat_structure.LinePlot(),
        vector_field_quiver: mat_structure.VectorFieldQuiver = mat_structure.VectorFieldQuiver(),
        patch_overlay: mat_structure.PatchOverlay = mat_structure.PatchOverlay(),
        voronoi_drawer: mat_structure.VoronoiDrawer = mat_structure.VoronoiDrawer(),
        origin_scatter: mat_obj.OriginScatter = mat_obj.OriginScatter(),
        mask_scatter: mat_obj.MaskScatter = mat_obj.MaskScatter(),
        border_scatter: mat_obj.BorderScatter = mat_obj.BorderScatter(),
        positions_scatter: mat_obj.PositionsScatter = mat_obj.PositionsScatter(),
        index_scatter: mat_obj.IndexScatter = mat_obj.IndexScatter(),
        pixelization_grid_scatter: mat_obj.PixelizationGridScatter = mat_obj.PixelizationGridScatter(),
        parallel_overscan_plot: mat_obj.ParallelOverscanPlot = mat_obj.ParallelOverscanPlot(),
        serial_prescan_plot: mat_obj.SerialPrescanPlot = mat_obj.SerialPrescanPlot(),
        serial_overscan_plot: mat_obj.SerialOverscanPlot = mat_obj.SerialOverscanPlot(),
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
        units : mat_base.Units
            The units of the figure used to plot the data structure which sets the y and x ticks and labels.
        figure : mat_base.Figure
            Opens the matplotlib figure before plotting via `plt.figure` and closes it once plotting is complete
            via `plt.close`
        cmap : mat_base.Cmap
            Customizes the colormap of the plot and its normalization via matplotlib `colors` objects such 
            as `colors.Normalize` and `colors.LogNorm`.
        colorbar : mat_base.Colorbar
            Plots the colorbar of the plot via `plt.colorbar` and customizes its appearance and ticks using methods
            like `cb.set_yticklabels` and `cb.ax.tick_params`.
        tickparams : mat_base.TickParams
            Customizes the appearances of the y and x ticks on the plot, (e.g. the fontsize), using `plt.tick_params`.
        yticks : mat_base.YTicks
            Sets the yticks of the plot, including scaling them to new units depending on the `Units` object, via
            `plt.yticks`.
        xticks : mat_base.XTicks
            Sets the xticks of the plot, including scaling them to new units depending on the `Units` object, via
            `plt.xticks`.
        title : mat_base.Title
            Sets the figure title and customizes its appearance using `plt.title`.        
        ylabel : mat_base.YLabel
            Sets the figure ylabel and customizes its appearance using `plt.ylabel`.
        xlabel : mat_base.XLabel
            Sets the figure xlabel and customizes its appearance using `plt.xlabel`.
        legend : mat_base.Legend
            Sets whether the plot inclues a legend and customizes its appearance and labels using `plt.legend`.
        output : mat_base.Output
            Sets if the figure is displayed on the user's screen or output to `.png` using `plt.show` and `plt.savefig`
        array_overlay: mat_structure.ArrayOverlay
            Overlays an input `Array` over the figure using `plt.imshow`.
        grid_scatter : mat_structure.GridScatter
            Scatters a `Grid` of (y,x) coordinates over the figure using `plt.scatter`.
        line_plot: mat_structure.LinePlot
            Plots lines of data (e.g. a y versus x plot via `plt.plot`, vertical lines via `plt.avxline`, etc.)
        vector_field_quiver: mat_structure.VectorFieldQuiver
            Plots a `VectorField` object using the matplotlib function `plt.quiver`.
        patch_overlay: mat_structure.PatchOverlay
            Overlays matplotlib `patches.Patch` objects over the figure, such as an `Ellipse`.
        voronoi_drawer: mat_structure.VoronoiDrawer
            Draws a colored Voronoi mesh of pixels using `plt.fill`.
        origin_scatter : mat_obj.OriginScatter
            Scatters the (y,x) origin of the data structure on the figure.
        mask_scatter : mat_obj.MaskScatter
            Scatters an input `Mask2d` over the plotted data structure's figure.
        border_scatter : mat_obj.BorderScatter
            Scatters the border of an input `Mask2d` over the plotted data structure's figure.
        positions_scatter : mat_obj.PositionsScatter
            Scatters specific (y,x) coordinates input as a `GridIrregular` object over the figure.
        index_scatter : mat_obj.IndexScatter
            Scatters specific coordinates of an input `Grid` based on input values of the `Grid`'s 1D or 2D indexes.
        pixelization_grid_scatter : mat_obj.PixelizationGridScatter
            Scatters the `PixelizationGrid` of a `Pixelization` object.
        parallel_overscan_plot : mat_obj.ParallelOverscanPlot
            Plots the parallel overscan on an `Array` data structure representing a CCD imaging via `plt.plot`.
        serial_prescan_plot : mat_obj.SerialPrescanPlot
            Plots the serial prescan on an `Array` data structure representing a CCD imaging via `plt.plot`.
        serial_overscan_plot : mat_obj.SerialOverscanPlot
            Plots the serial overscan on an `Array` data structure representing a CCD imaging via `plt.plot`.
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

        self.set_for_subplot(for_subplot=False)

    def set_for_subplot(self, for_subplot):

        self.for_subplot = for_subplot
        self.output.bypass = for_subplot

        for attr, value in self.__dict__.items():
            if hasattr(value, "for_subplot"):
                value.for_subplot = for_subplot

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

    def _plot_array(self, array, visuals, extent_manual=None, bypass_output=False):
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

        if visuals.array_overlay is not None:
            self.array_overlay.overlay_array(
                array=visuals.array_overlay, figure=self.figure
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

        visuals.plot_via_plotter(plotter=self)

        if not bypass_output:
            self.output.to_figure(structure=array)

        if not self.for_subplot and not bypass_output:
            self.figure.close()

    def _plot_frame(self, frame, visuals=None, bypass_output=False):
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

        visuals.plot_via_plotter(plotter=self)

        if not bypass_output:
            self.output.to_figure(structure=frame)

        if not self.for_subplot and not bypass_output:
            self.figure.close()

    def _plot_grid(
        self,
        grid,
        visuals=None,
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

        visuals.plot_via_plotter(plotter=self)

        if not bypass_output:
            self.output.to_figure(structure=grid)

        if not self.for_subplot and not bypass_output:
            self.figure.close()

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

    def _plot_mapper(
        self,
        mapper,
        visuals=None,
        source_pixel_values=None,
        image_pixel_indexes=None,
        source_pixel_indexes=None,
        bypass_output=False,
    ):

        if isinstance(mapper, mappers.MapperRectangular):

            self._plot_rectangular_mapper(
                mapper=mapper,
                visuals=visuals,
                source_pixel_values=source_pixel_values,
                image_pixel_indexes=image_pixel_indexes,
                source_pixel_indexes=source_pixel_indexes,
                bypass_output=bypass_output,
            )

        else:

            self._plot_voronoi_mapper(
                mapper=mapper,
                visuals=visuals,
                source_pixel_values=source_pixel_values,
                image_pixel_indexes=image_pixel_indexes,
                source_pixel_indexes=source_pixel_indexes,
                bypass_output=bypass_output,
            )

    def _plot_rectangular_mapper(
        self,
        mapper,
        visuals=None,
        source_pixel_values=None,
        image_pixel_indexes=None,
        source_pixel_indexes=None,
        bypass_output=False,
    ):

        self.figure.open()

        if source_pixel_values is not None:

            self._plot_array(
                array=source_pixel_values,
                lines=lines,
                positions=positions,
                include_origin=include_origin,
                bypass_output=True,
            )

        self.set_axis_limits(
            axis_limits=mapper.pixelization_grid.extent,
            grid=None,
            symmetric_around_centre=False,
        )

        self.yticks.set(
            array=None,
            min_value=mapper.pixelization_grid.extent[2],
            max_value=mapper.pixelization_grid.extent[3],
            units=self.units,
        )
        self.xticks.set(
            array=None,
            min_value=mapper.pixelization_grid.extent[0],
            max_value=mapper.pixelization_grid.extent[1],
            units=self.units,
        )

        self.line_plot.plot_rectangular_grid_lines(
            extent=mapper.pixelization_grid.extent, shape_2d=mapper.shape_2d
        )

        self.title.set()
        self.tickparams.set()
        self.ylabel.set(units=self.units, include_brackets=True)
        self.xlabel.set(units=self.units, include_brackets=True)

        visuals.plot_via_plotter(plotter=self)

        if image_pixel_indexes is not None:
            self.index_scatter.scatter_grid_indexes(
                grid=mapper.grid, indexes=image_pixel_indexes
            )

        if source_pixel_indexes is not None:

            indexes = mapper.image_pixel_indexes_from_source_pixel_indexes(
                source_pixel_indexes=source_pixel_indexes
            )

            self.index_scatter.scatter_grid_indexes(grid=mapper.grid, indexes=indexes)

        if not bypass_output:
            self.output.to_figure(structure=None)

        if not self.for_subplot and not bypass_output:
            self.figure.close()

    def _plot_voronoi_mapper(
        self,
        mapper,
        visuals=None,
        source_pixel_values=None,
        image_pixel_indexes=None,
        source_pixel_indexes=None,
        bypass_output=False,
    ):

        self.figure.open()

        self.set_axis_limits(
            axis_limits=mapper.pixelization_grid.extent,
            grid=None,
            symmetric_around_centre=False,
        )

        self.tickparams.set()
        self.yticks.set(
            array=None,
            min_value=mapper.pixelization_grid.extent[2],
            max_value=mapper.pixelization_grid.extent[3],
            units=self.units,
        )
        self.xticks.set(
            array=None,
            min_value=mapper.pixelization_grid.extent[0],
            max_value=mapper.pixelization_grid.extent[1],
            units=self.units,
        )

        self.voronoi_drawer.draw_voronoi_pixels(
            mapper=mapper,
            values=source_pixel_values,
            cmap=self.cmap.config_dict["cmap"],
            cb=self.colorbar,
        )

        self.title.set()
        self.ylabel.set(units=self.units, include_brackets=True)
        self.xlabel.set(units=self.units, include_brackets=True)

        visuals.plot_via_plotter(plotter=self)

        if image_pixel_indexes is not None:
            self.index_scatter.scatter_grid_indexes(
                grid=mapper.grid, indexes=image_pixel_indexes
            )

        if source_pixel_indexes is not None:

            indexes = mapper.image_pixel_indexes_from_source_pixel_indexes(
                source_pixel_indexes=source_pixel_indexes
            )

            self.index_scatter.scatter_grid_indexes(grid=mapper.grid, indexes=indexes)

        if not bypass_output:
            self.output.to_figure(structure=None)

        if not self.for_subplot and not bypass_output:
            self.figure.close()

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


def plotter_key_from_dictionary(dictionary):
    plotter_key = None

    for key, value in dictionary.items():
        if isinstance(value, Plotter):
            plotter_key = key

    return plotter_key


def plotter_and_plotter_key_from_func(func):
    defaults = inspect.getfullargspec(func).defaults
    plotter = [value for value in defaults if isinstance(value, Plotter)][0]

    if isinstance(plotter, Plotter):
        plotter_key = "plotter"
    else:
        plotter_key = "plotter"

    return plotter, plotter_key


def kpc_per_scaled_of_object_from_dictionary(dictionary):

    kpc_per_scaled = None

    for key, value in dictionary.items():
        if hasattr(value, "kpc_per_scaled"):
            return value.kpc_per_scaled

    return kpc_per_scaled


def set_plotter_for_figure(func):
    @wraps(func)
    def wrapper(*args, **kwargs):

        plotter_key = plotter_key_from_dictionary(dictionary=kwargs)

        if plotter_key is not None:
            plotter = kwargs[plotter_key]
        else:
            plotter = Plotter()
            plotter_key = "plotter"

        plotter.set_for_subplot(for_subplot=False)

        kwargs[plotter_key] = plotter

        return func(*args, **kwargs)

    return wrapper


def set_plotter_for_subplot(func):
    @wraps(func)
    def wrapper(*args, **kwargs):

        plotter_key = plotter_key_from_dictionary(dictionary=kwargs)

        if plotter_key is not None:
            plotter = kwargs[plotter_key]
        else:
            plotter = Plotter()
            plotter_key = "plotter"

        plotter.set_for_subplot(for_subplot=True)

        kwargs[plotter_key] = plotter

        return func(*args, **kwargs)

    return wrapper


def set_subplot_filename(func):
    """
    Decorate a profile method that accepts a coordinate grid and returns a data_type grid.

    If an interpolator attribute is associated with the input grid then that interpolator is used to down sample the
    coordinate grid prior to calling the function and up sample the result of the function.

    If no interpolator attribute is associated with the input grid then the function is called as hyper.

    Parameters
    ----------
    func
        Some method that accepts a grid

    Returns
    -------
    decorated_function
        The function with optional interpolation
    """

    @wraps(func)
    def wrapper(*args, **kwargs):

        plotter_key = plotter_key_from_dictionary(dictionary=kwargs)
        plotter = kwargs[plotter_key]

        if not isinstance(plotter, Plotter):
            raise exc.PlottingException(
                "The decorator set_subplot_title was applied to a function without a Plotter class"
            )

        filename = plotter.output.filename_from_func(func=func)

        plotter = plotter.plotter_with_new_output(filename=filename)

        kwargs[plotter_key] = plotter

        return func(*args, **kwargs)

    return wrapper


def set_labels(func):
    """
    Decorate a profile method that accepts a coordinate grid and returns a data_type grid.

    If an interpolator attribute is associated with the input grid then that interpolator is used to down sample the
    coordinate grid prior to calling the function and up sample the result of the function.

    If no interpolator attribute is associated with the input grid then the function is called as hyper.

    Parameters
    ----------
    func
        Some method that accepts a grid

    Returns
    -------
    decorated_function
        The function with optional interpolation
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        plotter_key = plotter_key_from_dictionary(dictionary=kwargs)
        plotter = kwargs[plotter_key]

        title = plotter.title.title_from_func(func=func)
        yunits = plotter.ylabel.units_from_func(func=func, for_ylabel=True)
        xunits = plotter.xlabel.units_from_func(func=func, for_ylabel=False)

        plotter = plotter.plotter_with_new_labels(
            title_label=title, ylabel_units=yunits, xlabel_units=xunits
        )

        filename = plotter.output.filename_from_func(func=func)

        plotter = plotter.plotter_with_new_output(filename=filename)

        kpc_per_scaled = kpc_per_scaled_of_object_from_dictionary(dictionary=kwargs)

        plotter = plotter.plotter_with_new_units(conversion_factor=kpc_per_scaled)

        kwargs[plotter_key] = plotter

        return func(*args, **kwargs)

    return wrapper
