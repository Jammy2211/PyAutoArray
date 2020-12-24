import matplotlib

from autoconf import conf
from autoarray.plot.plotter import include as inc

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
from autoarray.structures import grids
from autoarray.inversion import mappers


def setting(section, name, python_type):
    return conf.instance.visualize_figures.get(section, name, python_type)


def load_setting(section, value, name, python_type):
    return (
        value
        if value is not None
        else setting(section=section, name=name, python_type=python_type)
    )


def load_figure_setting(section, name, python_type):
    return conf.instance.visualize_figures.get(section, name, python_type)


def load_subplot_setting(section, name, python_type):
    return conf.instance.visualize_subplots.get(section, name, python_type)


class AbstractPlotter:
    def __init__(
        self,
        module: str = None,
        units: mat_base.Units = None,
        figure: mat_base.Figure = None,
        cmap: mat_base.Cmap = None,
        colorbar: mat_base.Colorbar = None,
        tickparams: mat_base.TickParams = None,
        yticks: mat_base.YTicks = None,
        xticks: mat_base.XTicks = None,
        title: mat_base.Title = None,
        ylabel: mat_base.YLabel = None,
        xlabel: mat_base.XLabel = None,
        legend: mat_base.Legend = None,
        output: mat_base.Output = None,
        array_overlay: mat_structure.ArrayOverlay = None,
        grid_scatter: mat_structure.GridScatter = None,
        line_plot: mat_structure.LinePlot = None,
        vector_field_quiver: mat_structure.VectorFieldQuiver = None,
        patch_overlay: mat_structure.PatchOverlay = None,
        voronoi_drawer: mat_structure.VoronoiDrawer = None,
        origin_scatter: mat_obj.OriginScatter = None,
        mask_scatter: mat_obj.MaskScatter = None,
        border_scatter: mat_obj.BorderScatter = None,
        positions_scatter: mat_obj.PositionsScatter = None,
        index_scatter: mat_obj.IndexScatter = None,
        pixelization_grid_scatter: mat_obj.PixelizationGridScatter = None,
        parallel_overscan_plot: mat_obj.ParallelOverscanPlot = None,
        serial_prescan_plot: mat_obj.SerialPrescanPlot = None,
        serial_overscan_plot: mat_obj.SerialOverscanPlot = None,
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
        module : str
            The name of the `plots.plot` module the plot function was called from, which is used to color the figure
            according to its module and therefore what it is plotting.
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
        if isinstance(self, Plotter):
            use_subplot_defaults = False
        else:
            use_subplot_defaults = True

        self.units = units if units is not None else mat_base.Units()

        self.figure = (
            figure
            if figure is not None
            else mat_base.Figure(use_subplot_defaults=use_subplot_defaults)
        )

        self.cmap = (
            cmap
            if cmap is not None
            else mat_base.Cmap(module=module, use_subplot_defaults=use_subplot_defaults)
        )

        self.cb = (
            colorbar
            if colorbar is not None
            else mat_base.Colorbar(use_subplot_defaults=use_subplot_defaults)
        )

        self.title = (
            title
            if title is not None
            else mat_base.Title(use_subplot_defaults=use_subplot_defaults)
        )

        self.tickparams = (
            tickparams
            if tickparams is not None
            else mat_base.TickParams(use_subplot_defaults=use_subplot_defaults)
        )

        self.yticks = (
            yticks
            if yticks is not None
            else mat_base.YTicks(use_subplot_defaults=use_subplot_defaults)
        )

        self.xticks = (
            xticks
            if xticks is not None
            else mat_base.XTicks(use_subplot_defaults=use_subplot_defaults)
        )

        self.ylabel = (
            ylabel
            if ylabel is not None
            else mat_base.YLabel(use_subplot_defaults=use_subplot_defaults)
        )

        self.xlabel = (
            xlabel
            if xlabel is not None
            else mat_base.XLabel(use_subplot_defaults=use_subplot_defaults)
        )

        self.legend = (
            legend
            if legend is not None
            else mat_base.Legend(use_subplot_defaults=use_subplot_defaults)
        )

        self.output = (
            output
            if output is not None
            else mat_base.Output(bypass=isinstance(self, SubPlotter))
        )

        self.origin_scatter = (
            origin_scatter
            if origin_scatter is not None
            else mat_obj.OriginScatter(use_subplot_defaults=use_subplot_defaults)
        )
        self.mask_scatter = (
            mask_scatter
            if mask_scatter is not None
            else mat_obj.MaskScatter(use_subplot_defaults=use_subplot_defaults)
        )
        self.border_scatter = (
            border_scatter
            if border_scatter is not None
            else mat_obj.BorderScatter(use_subplot_defaults=use_subplot_defaults)
        )
        self.grid_scatter = (
            grid_scatter
            if grid_scatter is not None
            else mat_structure.GridScatter(use_subplot_defaults=use_subplot_defaults)
        )
        self.positions_scatter = (
            positions_scatter
            if positions_scatter is not None
            else mat_obj.PositionsScatter(use_subplot_defaults=use_subplot_defaults)
        )
        self.index_scatter = (
            index_scatter
            if index_scatter is not None
            else mat_obj.IndexScatter(use_subplot_defaults=use_subplot_defaults)
        )
        self.pixelization_grid_scatter = (
            pixelization_grid_scatter
            if pixelization_grid_scatter is not None
            else mat_obj.PixelizationGridScatter(
                use_subplot_defaults=use_subplot_defaults
            )
        )

        self.line_plot = (
            line_plot
            if line_plot is not None
            else mat_structure.LinePlot(use_subplot_defaults=use_subplot_defaults)
        )

        self.vector_field_quiver = (
            vector_field_quiver
            if vector_field_quiver is not None
            else mat_structure.VectorFieldQuiver(
                use_subplot_defaults=use_subplot_defaults
            )
        )

        self.patch_overlay = (
            patch_overlay
            if patch_overlay is not None
            else mat_structure.PatchOverlay(use_subplot_defaults=use_subplot_defaults)
        )

        self.array_overlay = (
            array_overlay
            if array_overlay is not None
            else mat_structure.ArrayOverlay(use_subplot_defaults=use_subplot_defaults)
        )

        self.voronoi_drawer = (
            voronoi_drawer
            if voronoi_drawer is not None
            else mat_structure.VoronoiDrawer(use_subplot_defaults=use_subplot_defaults)
        )

        self.parallel_overscan_plot = (
            parallel_overscan_plot
            if parallel_overscan_plot is not None
            else mat_obj.ParallelOverscanPlot(use_subplot_defaults=use_subplot_defaults)
        )

        self.serial_prescan_plot = (
            serial_prescan_plot
            if serial_prescan_plot is not None
            else mat_obj.SerialPrescanPlot(use_subplot_defaults=use_subplot_defaults)
        )

        self.serial_overscan_plot = (
            serial_overscan_plot
            if serial_overscan_plot is not None
            else mat_obj.SerialOverscanPlot(use_subplot_defaults=use_subplot_defaults)
        )

    def plot_array(
        self,
        array,
        mask=None,
        lines=None,
        positions=None,
        grid=None,
        vector_field=None,
        patches=None,
        array_overlay=None,
        include_origin=False,
        include_border=False,
        extent_manual=None,
        bypass_output=False,
    ):
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
            cmap=self.cmap.kwargs["cmap"],
            norm=norm_scale,
            extent=extent,
        )

        if extent_manual is not None:
            extent = extent_manual

        if array_overlay is not None:
            self.array_overlay.overlay_array(array=array_overlay, figure=self.figure)

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

        self.cb.set()
        if include_origin:
            self.origin_scatter.scatter_grid(grid=[array.origin])

        if mask is not None:
            self.mask_scatter.scatter_grid(
                grid=mask.geometry.edge_grid_sub_1.in_1d_binned
            )

        if include_border and mask is not None:
            self.border_scatter.scatter_grid(
                grid=mask.geometry.border_grid_sub_1.in_1d_binned
            )

        if grid is not None:
            self.grid_scatter.scatter_grid(grid=grid)

        if positions is not None:
            if not isinstance(positions, grids.GridIrregularGrouped):
                self.positions_scatter.scatter_grid(grid=positions)
            else:
                self.positions_scatter.scatter_grid_grouped(grid_grouped=positions)

        if vector_field is not None:
            self.vector_field_quiver.quiver_vector_field(vector_field=vector_field)

        if patches is not None:
            self.patch_overlay.overlay_patches(patches=patches)

        if lines is not None:
            self.line_plot.plot_grid_grouped(grid_grouped=lines)

        if not bypass_output:
            self.output.to_figure(structure=array)

        if not isinstance(self, SubPlotter) and not bypass_output:
            self.figure.close()

    def plot_frame(
        self,
        frame,
        lines=None,
        include_origin=False,
        include_parallel_overscan=False,
        include_serial_prescan=False,
        include_serial_overscan=False,
        bypass_output=False,
    ):
        """Plot an array of data_type as a figure.

        Parameters
        -----------
        settings : PlotterSettings
            Settings
        include : PlotterInclude
            Include
        labels : PlotterLabels
            labels
        outputs : PlotterOutputs
            outputs
        array : data_type.array.aa.Scaled
            The 2D array of data_type which is plotted.
        origin : (float, float).
            The origin of the coordinate system of the array, which is plotted as an 'x' on the image if input.
        mask : data_type.array.mask.Mask2D
            The mask applied to the array, the edge of which is plotted as a set of points over the plotted array.
        extract_array_from_mask : bool
            The plotter array is extracted using the mask, such that masked values are plotted as zeros. This ensures \
            bright features outside the mask do not impact the color map of the plotter.
        zoom_around_mask : bool
            If True, the 2D region of the array corresponding to the rectangle encompassing all unmasked values is \
            plotted, thereby zooming into the region of interest.
        border : bool
            If a mask is supplied, its borders pixels (e.g. the exterior edge) is plotted if this is `True`.
        positions : [[]]
            Lists of (y,x) coordinates on the image which are plotted as colored dots, to highlight specific pixels.
        grid : data_type.array.aa.Grid
            A grid of (y,x) coordinates which may be plotted over the plotted array.
        as_subplot : bool
            Whether the array is plotted as part of a subplot, in which case the grid figure is not opened / closed.
        unit_label : str
            The label for the unit_label of the y / x axis of the plots.
        unit_conversion_factor : float or None
            The conversion factor between arc-seconds and kiloparsecs, required to plotter the unit_label in kpc.
        figsize : (int, int)
            The size of the figure in (total_y_pixels, total_x_pixels).
        aspect : str
            The aspect ratio of the array, specifically whether it is forced to be square ('equal') or adapts its size to \
            the figure size ('auto').
        cmap : str
            The colormap the array is plotted using, which may be chosen from the standard matplotlib colormaps.
        norm : str
            The normalization of the colormap used to plotter the image, specifically whether it is linear ('linear'), log \
            ('log') or a symmetric log normalization ('symmetric_log').
        vmin : float or None
            The minimum array value the colormap map spans (all values below this value are plotted the same color).
        vmax : float or None
            The maximum array value the colormap map spans (all values above this value are plotted the same color).
        linthresh : float
            For the 'symmetric_log' colormap normalization ,this specifies the range of values within which the colormap \
            is linear.
        linscale : float
            For the 'symmetric_log' colormap normalization, this allowws the linear range set by linthresh to be stretched \
            relative to the logarithmic range.
        cb_ticksize : int
            The size of the tick labels on the colorbar.
        cb_fraction : float
            The fraction of the figure that the colorbar takes up, which resizes the colorbar relative to the figure.
        cb_pad : float
            Pads the color bar in the figure, which resizes the colorbar relative to the figure.
        labelsize : int
            The fontsize of the x axes label.
        labelsize : int
            The fontsize of the y axes label.
        xyticksize : int
            The font size of the x and y ticks on the figure axes.
        mask_scatter : int
            The size of the points plotted to show the mask.
        xticks_manual :  [] or None
            If input, the xticks do not use the array's default xticks but instead overwrite them as these values.
        yticks_manual :  [] or None
            If input, the yticks do not use the array's default yticks but instead overwrite them as these values.
        output_path : str
            The path on the hard-disk where the figure is output.
        output_filename : str
            The filename of the figure that is output.
        output_format : str
            The format the figue is output:
            'show' - display on computer screen.
            'png' - output to hard-disk as a png.
            'fits' - output to hard-disk as a fits file.'

        Returns
        --------
        None

        Examples
        --------
            plotter.plot_frame(
            array=image, origin=(0.0, 0.0), mask=circular_mask,
            border=False, points=[[1.0, 1.0], [2.0, 2.0]], grid=None, as_subplot=False,
            unit_label='scaled', kpc_per_scaled=None, figsize=(7,7), aspect='auto',
            cmap='jet', norm='linear, vmin=None, vmax=None, linthresh=None, linscale=None,
            cb_ticksize=10, cb_fraction=0.047, cb_pad=0.01, cb_tick_values=None, cb_tick_labels=None,
            title='Image', titlesize=16, labelsize=16, labelsize=16, xyticksize=16,
            mask_scatter=10, border_pointsize=2, position_pointsize=10, grid_pointsize=10,
            xticks_manual=None, yticks_manual=None,
            output_path='/path/to/output', output_format='png', output_filename='image')
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
            cmap=self.cmap.kwargs["cmap"],
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

        self.cb.set()
        if include_origin:
            self.origin_scatter.scatter_grid(grid=[frame.origin])

        if (
            include_parallel_overscan is not None
            and frame.scans.parallel_overscan is not None
        ):
            self.parallel_overscan_plot.plot_rectangular_grid_lines(
                extent=frame.scans.parallel_overscan, shape_2d=frame.shape_2d
            )

        if (
            include_serial_prescan is not None
            and frame.scans.serial_prescan is not None
        ):
            self.serial_prescan_plot.plot_rectangular_grid_lines(
                extent=frame.scans.serial_prescan, shape_2d=frame.shape_2d
            )

        if (
            include_serial_overscan is not None
            and frame.scans.serial_overscan is not None
        ):
            self.serial_overscan_plot.plot_rectangular_grid_lines(
                extent=frame.scans.serial_overscan, shape_2d=frame.shape_2d
            )

        if not bypass_output:
            self.output.to_figure(structure=frame)

        if not isinstance(self, SubPlotter) and not bypass_output:
            self.figure.close()

    def plot_grid(
        self,
        grid,
        color_array=None,
        axis_limits=None,
        indexes=None,
        positions=None,
        lines=None,
        symmetric_around_centre=True,
        include_origin=False,
        include_border=False,
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

            plt.cm.get_cmap(self.cmap.kwargs["cmap"])
            self.grid_scatter.scatter_grid_colored(
                grid=grid, color_array=color_array, cmap=self.cmap.kwargs["cmap"]
            )
            self.cb.set()

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

        if include_origin:
            self.origin_scatter.scatter_grid(grid=[grid.origin])

        if include_border:
            self.border_scatter.scatter_grid(grid=grid.sub_border_grid)

        if indexes is not None:
            self.index_scatter.scatter_grid_indexes(grid=grid, indexes=indexes)

        if positions is not None:
            if not isinstance(positions, grids.GridIrregularGrouped):
                self.positions_scatter.scatter_grid(grid=positions)
            else:
                self.positions_scatter.scatter_grid_grouped(grid_grouped=positions)

        if lines is not None:
            self.line_plot.plot_grid_grouped(grid_grouped=lines)

        if not bypass_output:
            self.output.to_figure(structure=grid)

        if not isinstance(self, SubPlotter) and not bypass_output:
            self.figure.close()

    def plot_line(
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

        if not isinstance(self, SubPlotter) and not bypass_output:
            self.figure.close()

    def plot_mapper(
        self,
        mapper,
        source_pixel_values=None,
        positions=None,
        lines=None,
        include_origin=False,
        include_pixelization_grid=False,
        include_grid=False,
        include_border=False,
        image_pixel_indexes=None,
        source_pixel_indexes=None,
        bypass_output=False,
    ):

        if isinstance(mapper, mappers.MapperRectangular):

            self.plot_rectangular_mapper(
                mapper=mapper,
                source_pixel_values=source_pixel_values,
                positions=positions,
                lines=lines,
                include_origin=include_origin,
                include_grid=include_grid,
                include_pixelization_grid=include_pixelization_grid,
                include_border=include_border,
                image_pixel_indexes=image_pixel_indexes,
                source_pixel_indexes=source_pixel_indexes,
            )

        else:

            self.plot_voronoi_mapper(
                mapper=mapper,
                source_pixel_values=source_pixel_values,
                positions=positions,
                lines=lines,
                include_origin=include_origin,
                include_grid=include_grid,
                include_pixelization_grid=include_pixelization_grid,
                include_border=include_border,
                image_pixel_indexes=image_pixel_indexes,
                source_pixel_indexes=source_pixel_indexes,
            )

    def plot_rectangular_mapper(
        self,
        mapper,
        source_pixel_values=None,
        positions=None,
        lines=None,
        include_origin=False,
        include_pixelization_grid=False,
        include_grid=False,
        include_border=False,
        image_pixel_indexes=None,
        source_pixel_indexes=None,
        bypass_output=False,
    ):

        self.figure.open()

        if source_pixel_values is not None:

            self.plot_array(
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

        if include_origin:
            self.origin_scatter.scatter_grid(grid=[mapper.grid.origin])

        if include_pixelization_grid:
            self.pixelization_grid_scatter.scatter_grid(grid=mapper.pixelization_grid)

        if include_grid:
            self.grid_scatter.scatter_grid(grid=mapper.grid)

        if include_border:
            sub_border_1d_indexes = mapper.grid.mask.regions._sub_border_1d_indexes
            sub_border_grid = mapper.grid[sub_border_1d_indexes, :]
            self.border_scatter.scatter_grid(grid=sub_border_grid)

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

        if not isinstance(self, SubPlotter) and not bypass_output:
            self.figure.close()

    def plot_voronoi_mapper(
        self,
        mapper,
        source_pixel_values=None,
        positions=None,
        lines=None,
        include_origin=False,
        include_pixelization_grid=False,
        include_grid=False,
        include_border=False,
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
            cmap=self.cmap.kwargs["cmap"],
            cb=self.cb,
        )

        self.title.set()
        self.ylabel.set(units=self.units, include_brackets=True)
        self.xlabel.set(units=self.units, include_brackets=True)

        if include_origin:
            self.origin_scatter.scatter_grid(grid=[mapper.grid.origin])

        if include_pixelization_grid:
            self.pixelization_grid_scatter.scatter_grid(grid=mapper.pixelization_grid)

        if include_grid:
            self.grid_scatter.scatter_grid(grid=mapper.grid)

        if include_border:
            sub_border_1d_indexes = mapper.grid.mask.regions._sub_border_1d_indexes
            sub_border_grid = mapper.grid[sub_border_1d_indexes, :]
            self.border_scatter.scatter_grid(grid=sub_border_grid)

        if positions is not None:
            if not isinstance(positions, grids.GridIrregularGrouped):
                self.positions_scatter.scatter_grid(grid=positions)
            else:
                self.positions_scatter.scatter_grid_grouped(grid_grouped=positions)

        if lines is not None:
            self.line_plot.plot_grid_grouped(grid_grouped=lines)

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

        if not isinstance(self, SubPlotter) and not bypass_output:
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
            title_label if title_label is not None else self.title.kwargs["label"]
        )
        plotter.title.kwargs["fontsize"] = (
            title_fontsize
            if title_fontsize is not None
            else self.title.kwargs["fontsize"]
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
            else self.tickparams.kwargs["labelsize"]
        )

        return plotter

    def plotter_with_new_cmap(
        self, cmap=None, norm=None, vmax=None, vmin=None, linthresh=None, linscale=None
    ):

        plotter = copy.deepcopy(self)

        plotter.cmap.kwargs["cmap"] = (
            cmap if cmap is not None else self.cmap.kwargs["cmap"]
        )
        plotter.cmap.kwargs["norm"] = (
            norm if norm is not None else self.cmap.kwargs["norm"]
        )
        plotter.cmap.kwargs["vmax"] = (
            vmax if vmax is not None else self.cmap.kwargs["vmax"]
        )
        plotter.cmap.kwargs["vmin"] = (
            vmin if vmin is not None else self.cmap.kwargs["vmin"]
        )
        plotter.cmap.kwargs["linthresh"] = (
            linthresh if linthresh is not None else self.cmap.kwargs["linthresh"]
        )
        plotter.cmap.kwargs["linscale"] = (
            linscale if linscale is not None else self.cmap.kwargs["linscale"]
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


class Plotter(AbstractPlotter):
    def __init__(
        self,
        module=None,
        units=None,
        figure=None,
        cmap=None,
        colorbar=None,
        title=None,
        tickparams=None,
        yticks=None,
        xticks=None,
        ylabel=None,
        xlabel=None,
        legend=None,
        output=None,
        origin_scatter=None,
        mask_scatter=None,
        border_scatter=None,
        grid_scatter=None,
        positions_scatter=None,
        index_scatter=None,
        pixelization_grid_scatter=None,
        line_plot=None,
        vector_field_quiver=None,
        patch_overlay=None,
        array_overlay=None,
        voronoi_drawer=None,
        parallel_overscan_plot=None,
        serial_prescan_plot=None,
        serial_overscan_plot=None,
    ):
        super(Plotter, self).__init__(
            module=module,
            units=units,
            figure=figure,
            cmap=cmap,
            colorbar=colorbar,
            legend=legend,
            title=title,
            tickparams=tickparams,
            yticks=yticks,
            xticks=xticks,
            ylabel=ylabel,
            xlabel=xlabel,
            output=output,
            origin_scatter=origin_scatter,
            mask_scatter=mask_scatter,
            border_scatter=border_scatter,
            grid_scatter=grid_scatter,
            positions_scatter=positions_scatter,
            index_scatter=index_scatter,
            pixelization_grid_scatter=pixelization_grid_scatter,
            line_plot=line_plot,
            vector_field_quiver=vector_field_quiver,
            patch_overlay=patch_overlay,
            array_overlay=array_overlay,
            voronoi_drawer=voronoi_drawer,
            parallel_overscan_plot=parallel_overscan_plot,
            serial_prescan_plot=serial_prescan_plot,
            serial_overscan_plot=serial_overscan_plot,
        )


class SubPlotter(AbstractPlotter):
    def __init__(
        self,
        module=None,
        units=None,
        figure=None,
        cmap=None,
        colorbar=None,
        title=None,
        tickparams=None,
        yticks=None,
        xticks=None,
        ylabel=None,
        xlabel=None,
        legend=None,
        output=None,
        origin_scatter=None,
        mask_scatter=None,
        border_scatter=None,
        grid_scatter=None,
        positions_scatter=None,
        index_scatter=None,
        pixelization_grid_scatter=None,
        line_plot=None,
        vector_field_quiver=None,
        patch_overlay=None,
        array_overlay=None,
        voronoi_drawer=None,
        parallel_overscan_plot=None,
        serial_prescan_plot=None,
        serial_overscan_plot=None,
    ):

        super(SubPlotter, self).__init__(
            module=module,
            units=units,
            figure=figure,
            cmap=cmap,
            colorbar=colorbar,
            legend=legend,
            title=title,
            tickparams=tickparams,
            yticks=yticks,
            xticks=xticks,
            ylabel=ylabel,
            xlabel=xlabel,
            output=output,
            origin_scatter=origin_scatter,
            mask_scatter=mask_scatter,
            border_scatter=border_scatter,
            grid_scatter=grid_scatter,
            positions_scatter=positions_scatter,
            index_scatter=index_scatter,
            pixelization_grid_scatter=pixelization_grid_scatter,
            line_plot=line_plot,
            vector_field_quiver=vector_field_quiver,
            patch_overlay=patch_overlay,
            array_overlay=array_overlay,
            voronoi_drawer=voronoi_drawer,
            parallel_overscan_plot=parallel_overscan_plot,
            serial_prescan_plot=serial_prescan_plot,
            serial_overscan_plot=serial_overscan_plot,
        )

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

        if self.figure.kwargs["figsize"] is not None:
            return self.figure.kwargs["figsize"]

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


def include_key_from_dictionary(dictionary):
    include_key = None

    for key, value in dictionary.items():
        if isinstance(value, inc.Include):
            include_key = key

    return include_key


def plotter_key_from_dictionary(dictionary):
    plotter_key = None

    for key, value in dictionary.items():
        if isinstance(value, AbstractPlotter):
            plotter_key = key

    return plotter_key


def plotter_and_plotter_key_from_func(func):
    defaults = inspect.getfullargspec(func).defaults
    plotter = [value for value in defaults if isinstance(value, AbstractPlotter)][0]

    if isinstance(plotter, Plotter):
        plotter_key = "plotter"
    else:
        plotter_key = "sub_plotter"

    return plotter, plotter_key


def kpc_per_scaled_of_object_from_dictionary(dictionary):

    kpc_per_scaled = None

    for key, value in dictionary.items():
        if hasattr(value, "kpc_per_scaled"):
            return value.kpc_per_scaled

    return kpc_per_scaled


def set_include_and_plotter(func):
    @wraps(func)
    def wrapper(*args, **kwargs):

        include_key = include_key_from_dictionary(dictionary=kwargs)

        if include_key is not None:
            include = kwargs[include_key]
        else:
            include = inc.Include()
            include_key = "include"

        kwargs[include_key] = include

        plotter_key = plotter_key_from_dictionary(dictionary=kwargs)

        if plotter_key is not None:
            plotter = kwargs[plotter_key]
        else:
            plotter = Plotter(module=inspect.getmodule(func))
            plotter_key = "plotter"

        kwargs[plotter_key] = plotter

        return func(*args, **kwargs)

    return wrapper


def set_include_and_sub_plotter(func):
    @wraps(func)
    def wrapper(*args, **kwargs):

        include_key = include_key_from_dictionary(dictionary=kwargs)

        if include_key is not None:
            include = kwargs[include_key]
        else:
            include = inc.Include()
            include_key = "include"

        kwargs[include_key] = include

        sub_plotter_key = plotter_key_from_dictionary(dictionary=kwargs)

        if sub_plotter_key is not None:
            sub_plotter = kwargs[sub_plotter_key]
        else:
            sub_plotter = SubPlotter(module=inspect.getmodule(func))
            sub_plotter_key = "sub_plotter"

        kwargs[sub_plotter_key] = sub_plotter

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

        if not isinstance(plotter, SubPlotter):
            raise exc.PlottingException(
                "The decorator set_subplot_title was applied to a function without a SubPlotter class"
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


def plot_array(
    array,
    mask=None,
    lines=None,
    positions=None,
    grid=None,
    vector_field=None,
    patches=None,
    array_overlay=None,
    extent_manual=None,
    include=None,
    plotter=None,
):
    if include is None:
        include = inc.Include()

    if plotter is None:
        plotter = Plotter()

    plotter.plot_array(
        array=array,
        mask=mask,
        lines=lines,
        positions=positions,
        grid=grid,
        vector_field=vector_field,
        patches=patches,
        extent_manual=extent_manual,
        array_overlay=array_overlay,
        include_origin=include.origin,
        include_border=include.border,
    )


def plot_frame(frame, include=None, plotter=None):
    if include is None:
        include = inc.Include()

    if plotter is None:
        plotter = Plotter()

    plotter.plot_frame(
        frame=frame,
        include_origin=include.origin,
        include_parallel_overscan=include.parallel_overscan,
        include_serial_prescan=include.serial_prescan,
        include_serial_overscan=include.serial_overscan,
    )


def plot_grid(
    grid,
    color_array=None,
    axis_limits=None,
    indexes=None,
    positions=None,
    lines=None,
    symmetric_around_centre=True,
    include=None,
    plotter=None,
):
    if include is None:
        include = inc.Include()

    if plotter is None:
        plotter = Plotter()

    plotter.plot_grid(
        grid=grid,
        color_array=color_array,
        axis_limits=axis_limits,
        indexes=indexes,
        positions=positions,
        lines=lines,
        symmetric_around_centre=symmetric_around_centre,
        include_origin=include.origin,
        include_border=include.border,
    )


def plot_line(
    y,
    x,
    label=None,
    plot_axis_type="semilogy",
    vertical_lines=None,
    vertical_line_labels=None,
    plotter=None,
):
    if plotter is None:
        plotter = Plotter()

    plotter.plot_line(
        y=y,
        x=x,
        label=label,
        plot_axis_type=plot_axis_type,
        vertical_lines=vertical_lines,
        vertical_line_labels=vertical_line_labels,
    )


def plot_mapper_obj(
    mapper,
    image_pixel_indexes=None,
    source_pixel_indexes=None,
    include=None,
    plotter=None,
):
    if include is None:
        include = inc.Include()

    if plotter is None:
        plotter = Plotter()

    plotter.plot_mapper(
        mapper=mapper,
        include_pixelization_grid=include.inversion_pixelization_grid,
        include_grid=include.inversion_grid,
        include_border=include.inversion_border,
        image_pixel_indexes=image_pixel_indexes,
        source_pixel_indexes=source_pixel_indexes,
    )
