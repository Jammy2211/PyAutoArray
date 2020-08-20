from autoconf import conf
import matplotlib

backend = conf.get_matplotlib_backend()
if not backend in "default":
    matplotlib.use(backend)

if conf.instance.general.get("hpc", "hpc_mode", bool):
    matplotlib.use("Agg")

import matplotlib.pyplot as plt

import numpy as np
from functools import wraps
import copy

from autoarray import exc
from autoarray.structures import grids
from autoarray.plot import mat_objs
import inspect
import os
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
        module=None,
        units=None,
        figure=None,
        cmap=None,
        cb=None,
        ticks=None,
        labels=None,
        legend=None,
        output=None,
        origin_scatterer=None,
        mask_scatterer=None,
        border_scatterer=None,
        grid_scatterer=None,
        positions_scatterer=None,
        index_scatterer=None,
        pixelization_grid_scatterer=None,
        liner=None,
        array_overlayer=None,
        voronoi_drawer=None,
    ):

        if isinstance(self, Plotter):
            from_subplot_config = False
        else:
            from_subplot_config = True

        self.units = units if units is not None else mat_objs.Units()

        self.figure = (
            figure
            if figure is not None
            else mat_objs.Figure(from_subplot_config=from_subplot_config)
        )

        self.cmap = (
            cmap
            if cmap is not None
            else mat_objs.ColorMap(
                module=module, from_subplot_config=from_subplot_config
            )
        )

        self.cb = (
            cb
            if cb is not None
            else mat_objs.ColorBar(from_subplot_config=from_subplot_config)
        )

        self.ticks = (
            ticks
            if ticks is not None
            else mat_objs.Ticks(from_subplot_config=from_subplot_config)
        )

        self.labels = (
            labels
            if labels is not None
            else mat_objs.Labels(from_subplot_config=from_subplot_config)
        )

        self.legend = (
            legend
            if legend is not None
            else mat_objs.Legend(from_subplot_config=from_subplot_config)
        )

        self.output = (
            output
            if output is not None
            else mat_objs.Output(bypass=isinstance(self, SubPlotter))
        )

        self.origin_scatterer = (
            origin_scatterer
            if origin_scatterer is not None
            else mat_objs.OriginScatterer(from_subplot_config=from_subplot_config)
        )
        self.mask_scatterer = (
            mask_scatterer
            if mask_scatterer is not None
            else mat_objs.MaskScatterer(from_subplot_config=from_subplot_config)
        )
        self.border_scatterer = (
            border_scatterer
            if border_scatterer is not None
            else mat_objs.BorderScatterer(from_subplot_config=from_subplot_config)
        )
        self.grid_scatterer = (
            grid_scatterer
            if grid_scatterer is not None
            else mat_objs.GridScatterer(from_subplot_config=from_subplot_config)
        )
        self.positions_scatterer = (
            positions_scatterer
            if positions_scatterer is not None
            else mat_objs.PositionsScatterer(from_subplot_config=from_subplot_config)
        )
        self.index_scatterer = (
            index_scatterer
            if index_scatterer is not None
            else mat_objs.IndexScatterer(from_subplot_config=from_subplot_config)
        )
        self.pixelization_grid_scatterer = (
            pixelization_grid_scatterer
            if pixelization_grid_scatterer is not None
            else mat_objs.PixelizationGridScatterer(
                from_subplot_config=from_subplot_config
            )
        )

        self.liner = (
            liner
            if liner is not None
            else mat_objs.Liner(
                section="liner", from_subplot_config=from_subplot_config
            )
        )

        self.array_overlayer = (
            array_overlayer
            if array_overlayer is not None
            else mat_objs.ArrayOverlayer(from_subplot_config=from_subplot_config)
        )

        self.voronoi_drawer = (
            voronoi_drawer
            if voronoi_drawer is not None
            else mat_objs.VoronoiDrawer(from_subplot_config=from_subplot_config)
        )

    def plot_array(
        self,
        array,
        mask=None,
        lines=None,
        positions=None,
        grid=None,
        array_overlay=None,
        include_origin=False,
        include_border=False,
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
        mask : data_type.array.mask.Mask
            The mask applied to the array, the edge of which is plotted as a set of points over the plotted array.
        extract_array_from_mask : bool
            The plotter array is extracted using the mask, such that masked values are plotted as zeros. This ensures \
            bright features outside the mask do not impact the color map of the plotters.
        zoom_around_mask : bool
            If True, the 2D region of the array corresponding to the rectangle encompassing all unmasked values is \
            plotted, thereby zooming into the region of interest.
        border : bool
            If a mask is supplied, its borders pixels (e.g. the exterior edge) is plotted if this is *True*.
        positions : [[]]
            Lists of (y,x) coordinates on the image which are plotted as colored dots, to highlight specific pixels.
        grid : data_type.array.aa.Grid
            A grid of (y,x) coordinates which may be plotted over the plotted array.
        as_subplot : bool
            Whether the array is plotted as part of a subplot, in which case the grid figure is not opened / closed.
        unit_label : str
            The label for the unit_label of the y / x axis of the plots.
        unit_conversion_factor : float or None
            The conversion factor between arc-seconds and kiloparsecs, required to plotters the unit_label in kpc.
        figsize : (int, int)
            The size of the figure in (rows, columns).
        aspect : str
            The aspect ratio of the array, specifically whether it is forced to be square ('equal') or adapts its size to \
            the figure size ('auto').
        cmap : str
            The colormap the array is plotted using, which may be chosen from the standard matplotlib colormaps.
        norm : str
            The normalization of the colormap used to plotters the image, specifically whether it is linear ('linear'), log \
            ('log') or a symmetric log normalization ('symmetric_log').
        norm_min : float or None
            The minimum array value the colormap map spans (all values below this value are plotted the same color).
        norm_max : float or None
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
        xsize : int
            The fontsize of the x axes label.
        ysize : int
            The fontsize of the y axes label.
        xyticksize : int
            The font size of the x and y ticks on the figure axes.
        mask_scatterer : int
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
            plotters.plot_array(
            array=image, origin=(0.0, 0.0), mask=circular_mask,
            border=False, points=[[1.0, 1.0], [2.0, 2.0]], grid=None, as_subplot=False,
            unit_label='scaled', kpc_per_arcsec=None, figsize=(7,7), aspect='auto',
            cmap='jet', norm='linear, norm_min=None, norm_max=None, linthresh=None, linscale=None,
            cb_ticksize=10, cb_fraction=0.047, cb_pad=0.01, cb_tick_values=None, cb_tick_labels=None,
            title='Image', titlesize=16, xsize=16, ysize=16, xyticksize=16,
            mask_scatterer=10, border_pointsize=2, position_pointsize=10, grid_pointsize=10,
            xticks_manual=None, yticks_manual=None,
            output_path='/path/to/output', output_format='png', output_filename='image')
        """

        if array is None or np.all(array == 0):
            return

        if array.pixel_scales is None and self.units.use_scaled:
            raise exc.ArrayException(
                "You cannot plot an array using its scaled unit_label if the input array does not have "
                "a pixel scales attribute."
            )

        array = array.in_1d_binned

        if array.mask.is_all_false:
            buffer = 0
        else:
            buffer = 1

        extent = array.extent_of_zoomed_array(buffer=buffer)
        array = array.zoomed_around_mask(buffer=buffer)

        self.figure.open()
        aspect = self.figure.aspect_from_shape_2d(shape_2d=array.shape_2d)
        norm_scale = self.cmap.norm_from_array(array=array)

        plt.imshow(
            X=array.in_2d,
            aspect=aspect,
            cmap=self.cmap.cmap,
            norm=norm_scale,
            extent=extent,
        )

        if array_overlay is not None:
            self.array_overlayer.overlay_array(
                array_overlay=array_overlay, figure=self.figure
            )

        plt.axis(extent)

        self.ticks.set_yticks(array=array, extent=extent, units=self.units)
        self.ticks.set_xticks(array=array, extent=extent, units=self.units)

        self.labels.set_title()
        self.labels.set_yunits(units=self.units, include_brackets=True)
        self.labels.set_xunits(units=self.units, include_brackets=True)

        self.cb.set()
        if include_origin:
            self.origin_scatterer.scatter_grid(grid=[array.origin])

        if mask is not None:
            self.mask_scatterer.scatter_grid(
                grid=mask.geometry.edge_grid_sub_1.in_1d_binned
            )

        if include_border and mask is not None:
            self.border_scatterer.scatter_grid(
                grid=mask.geometry.border_grid_sub_1.in_1d_binned
            )

        if grid is not None:
            self.grid_scatterer.scatter_grid(grid=grid)

        if positions is not None:
            self.positions_scatterer.scatter_coordinates(coordinates=positions)

        if lines is not None:
            self.liner.draw_coordinates(coordinates=lines)

        if not bypass_output:
            self.output.to_figure(structure=array)

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
        """Plot a grid of (y,x) Cartesian coordinates as a scatter plotters of points.

        Parameters
        -----------
        grid : data_type.array.aa.Grid
            The (y,x) coordinates of the grid, in an array of shape (total_coordinates, 2).
        axis_limits : []
            The axis limits of the figure on which the grid is plotted, following [xmin, xmax, ymin, ymax].
        indexes : []
            A set of points that are plotted in a different colour for emphasis (e.g. to show the mappings between \
            different planes).
        as_subplot : bool
            Whether the grid is plotted as part of a subplot, in which case the grid figure is not opened / closed.
        label_yunits : str
            The label of the unit_label of the y / x axis of the plots.
        unit_conversion_factor : float
            The conversion factor between arc-seconds and kiloparsecs, required to plotters the unit_label in kpc.
        figsize : (int, int)
            The size of the figure in (rows, columns).
        pointsize : int
            The size of the points plotted on the grid.
        xyticksize : int
            The font size of the x and y ticks on the figure axes.
        title : str
            The text of the title.
        titlesize : int
            The size of of the title of the figure.
        xsize : int
            The fontsize of the x axes label.
        ysize : int
            The fontsize of the y axes label.
        output_path : str
            The path on the hard-disk where the figure is output.
        output_filename : str
            The filename of the figure that is output.
        output_format : str
            The format the figue is output:
            'show' - display on computer screen.
            'png' - output to hard-disk as a png.
        """

        self.figure.open()

        if color_array is None:

            self.grid_scatterer.scatter_grid(grid=grid)

        elif color_array is not None:

            plt.cm.get_cmap(self.cmap.cmap)
            self.grid_scatterer.scatter_colored_grid(
                grid=grid, color_array=color_array, cmap=self.cmap.cmap
            )
            self.cb.set()

        self.labels.set_title()
        self.labels.set_yunits(units=self.units, include_brackets=True)
        self.labels.set_xunits(units=self.units, include_brackets=True)

        if axis_limits is not None:

            self.set_axis_limits(
                axis_limits=axis_limits,
                grid=grid,
                symmetric_around_centre=symmetric_around_centre,
            )

        else:
            plt.axis(grid.extent)

        self.ticks.set_yticks(
            array=None,
            extent=grid.extent,
            units=self.units,
            symmetric_around_centre=symmetric_around_centre,
        )
        self.ticks.set_xticks(
            array=None,
            extent=grid.extent,
            units=self.units,
            symmetric_around_centre=symmetric_around_centre,
        )

        if include_origin:
            self.origin_scatterer.scatter_grid(grid=[grid.origin])

        if include_border:
            self.border_scatterer.scatter_grid(grid=grid.sub_border_grid)

        if indexes is not None:
            self.index_scatterer.scatter_grid_indexes(grid=grid, indexes=indexes)

        if positions is not None:
            self.positions_scatterer.scatter_grid(grid=positions)

        if lines is not None:
            self.liner.draw_grid(grid=lines)

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
        self.labels.set_title()

        if x is None:
            x = np.arange(len(y))

        self.liner.draw_y_vs_x(y=y, x=x, plot_axis_type=plot_axis_type, label=label)

        self.labels.set_yunits(units=self.units, include_brackets=False)
        self.labels.set_xunits(units=self.units, include_brackets=False)

        self.liner.draw_vertical_lines(
            vertical_lines=vertical_lines, vertical_line_labels=vertical_line_labels
        )

        if label is not None or vertical_line_labels is not None:
            self.legend.set()

        self.ticks.set_xticks(
            array=None, extent=[np.min(x), np.max(x)], units=self.units
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

        self.ticks.set_yticks(
            array=None, extent=mapper.pixelization_grid.extent, units=self.units
        )
        self.ticks.set_xticks(
            array=None, extent=mapper.pixelization_grid.extent, units=self.units
        )

        self.liner.draw_rectangular_grid_lines(
            extent=mapper.pixelization_grid.extent, shape_2d=mapper.shape_2d
        )

        self.labels.set_title()
        self.labels.set_yunits(units=self.units, include_brackets=True)
        self.labels.set_xunits(units=self.units, include_brackets=True)

        if include_origin:
            self.origin_scatterer.scatter_grid(grid=[mapper.grid.origin])

        if include_pixelization_grid:
            self.pixelization_grid_scatterer.scatter_grid(grid=mapper.pixelization_grid)

        if include_grid:
            self.grid_scatterer.scatter_grid(grid=mapper.grid)

        if include_border:
            sub_border_1d_indexes = mapper.grid.mask.regions._sub_border_1d_indexes
            sub_border_grid = mapper.grid[sub_border_1d_indexes, :]
            self.border_scatterer.scatter_grid(grid=sub_border_grid)

        if image_pixel_indexes is not None:
            self.index_scatterer.scatter_grid_indexes(
                grid=mapper.grid, indexes=image_pixel_indexes
            )

        if source_pixel_indexes is not None:

            indexes = mapper.image_pixel_indexes_from_source_pixel_indexes(
                source_pixel_indexes=source_pixel_indexes
            )

            self.index_scatterer.scatter_grid_indexes(grid=mapper.grid, indexes=indexes)

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

        self.ticks.set_yticks(
            array=None, extent=mapper.pixelization_grid.extent, units=self.units
        )
        self.ticks.set_xticks(
            array=None, extent=mapper.pixelization_grid.extent, units=self.units
        )

        self.voronoi_drawer.draw_voronoi_pixels(
            mapper=mapper, values=source_pixel_values, cmap=self.cmap.cmap, cb=self.cb
        )

        self.labels.set_title()
        self.labels.set_yunits(units=self.units, include_brackets=True)
        self.labels.set_xunits(units=self.units, include_brackets=True)

        if include_origin:
            self.origin_scatterer.scatter_grid(grid=[mapper.grid.origin])

        if include_pixelization_grid:
            self.pixelization_grid_scatterer.scatter_grid(grid=mapper.pixelization_grid)

        if include_grid:
            self.grid_scatterer.scatter_grid(grid=mapper.grid)

        if include_border:
            sub_border_1d_indexes = mapper.grid.mask.regions._sub_border_1d_indexes
            sub_border_grid = mapper.grid[sub_border_1d_indexes, :]
            self.border_scatterer.scatter_grid(grid=sub_border_grid)

        if positions is not None:
            self.positions_scatterer.scatter_coordinates(coordinates=positions)

        if lines is not None:
            self.liner.draw_grid(grid=lines)

        if image_pixel_indexes is not None:
            self.index_scatterer.scatter_grid_indexes(
                grid=mapper.grid, indexes=image_pixel_indexes
            )

        if source_pixel_indexes is not None:

            indexes = mapper.image_pixel_indexes_from_source_pixel_indexes(
                source_pixel_indexes=source_pixel_indexes
            )

            self.index_scatterer.scatter_grid_indexes(grid=mapper.grid, indexes=indexes)

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
        title=None,
        yunits=None,
        xunits=None,
        titlesize=None,
        ysize=None,
        xsize=None,
    ):

        plotter = copy.deepcopy(self)

        plotter.labels.title = title if title is not None else self.labels.title
        plotter.labels._yunits = yunits if yunits is not None else self.labels._yunits
        plotter.labels._xunits = xunits if xunits is not None else self.labels._xunits
        plotter.labels.titlesize = (
            titlesize if titlesize is not None else self.labels.titlesize
        )
        plotter.labels.ysize = ysize if ysize is not None else self.labels.ysize
        plotter.labels.xsize = xsize if xsize is not None else self.labels.xsize

        return plotter

    def plotter_with_new_cmap(
        self,
        cmap=None,
        norm=None,
        norm_max=None,
        norm_min=None,
        linthresh=None,
        linscale=None,
    ):

        plotter = copy.deepcopy(self)

        plotter.cmap.cmap = cmap if cmap is not None else self.cmap.cmap
        plotter.cmap.norm = norm if norm is not None else self.cmap.norm
        plotter.cmap.norm_max = norm_max if norm_max is not None else self.cmap.norm_max
        plotter.cmap.norm_min = norm_min if norm_min is not None else self.cmap.norm_min
        plotter.cmap.linthresh = (
            linthresh if linthresh is not None else self.cmap.linthresh
        )
        plotter.cmap.linscale = linscale if linscale is not None else self.cmap.linscale

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
        cb=None,
        ticks=None,
        labels=None,
        legend=None,
        output=None,
        origin_scatterer=None,
        mask_scatterer=None,
        border_scatterer=None,
        grid_scatterer=None,
        positions_scatterer=None,
        index_scatterer=None,
        pixelization_grid_scatterer=None,
        liner=None,
        array_overlayer=None,
        voronoi_drawer=None,
    ):

        super(Plotter, self).__init__(
            module=module,
            units=units,
            figure=figure,
            cmap=cmap,
            cb=cb,
            legend=legend,
            ticks=ticks,
            labels=labels,
            output=output,
            origin_scatterer=origin_scatterer,
            mask_scatterer=mask_scatterer,
            border_scatterer=border_scatterer,
            grid_scatterer=grid_scatterer,
            positions_scatterer=positions_scatterer,
            index_scatterer=index_scatterer,
            pixelization_grid_scatterer=pixelization_grid_scatterer,
            liner=liner,
            array_overlayer=array_overlayer,
            voronoi_drawer=voronoi_drawer,
        )


class SubPlotter(AbstractPlotter):
    def __init__(
        self,
        module=None,
        units=None,
        figure=None,
        cmap=None,
        cb=None,
        ticks=None,
        labels=None,
        legend=None,
        output=None,
        origin_scatterer=None,
        mask_scatterer=None,
        border_scatterer=None,
        grid_scatterer=None,
        positions_scatterer=None,
        index_scatterer=None,
        pixelization_grid_scatterer=None,
        liner=None,
        array_overlayer=None,
        voronoi_drawer=None,
    ):

        super(SubPlotter, self).__init__(
            module=module,
            units=units,
            figure=figure,
            cmap=cmap,
            cb=cb,
            legend=legend,
            ticks=ticks,
            labels=labels,
            output=output,
            origin_scatterer=origin_scatterer,
            mask_scatterer=mask_scatterer,
            border_scatterer=border_scatterer,
            grid_scatterer=grid_scatterer,
            positions_scatterer=positions_scatterer,
            index_scatterer=index_scatterer,
            pixelization_grid_scatterer=pixelization_grid_scatterer,
            liner=liner,
            array_overlayer=array_overlayer,
            voronoi_drawer=voronoi_drawer,
        )

    def open_subplot_figure(self, number_subplots):
        """Setup a figure for plotting an image.

        Parameters
        -----------
        figsize : (int, int)
            The size of the figure in (rows, columns).
        as_subplot : bool
            If the figure is a subplot, the setup_figure function is omitted to ensure that each subplot does not create a \
            new figure and so that it can be output using the *output.output_figure(structure=None)* function.
        """
        figsize = self.get_subplot_figsize(number_subplots=number_subplots)
        plt.figure(figsize=figsize)

    def setup_subplot(self, number_subplots, subplot_index, aspect=None):
        rows, columns = self.get_subplot_rows_columns(number_subplots=number_subplots)
        if aspect is None:
            plt.subplot(rows, columns, subplot_index)
        else:
            plt.subplot(rows, columns, subplot_index, aspect=float(aspect))

    def get_subplot_rows_columns(self, number_subplots):
        """Get the size of a sub plotters in (rows, columns), based on the number of subplots that are going to be plotted.

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
        """Get the size of a sub plotters in (rows, columns), based on the number of subplots that are going to be plotted.

        Parameters
        -----------
        number_subplots : int
            The number of subplots that are to be plotted in the figure.
        """

        if self.figure.figsize is not None:
            return self.figure.figsize

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


class Include:
    def __init__(
        self,
        origin=None,
        mask=None,
        grid=None,
        border=None,
        inversion_pixelization_grid=None,
        inversion_grid=None,
        inversion_border=None,
        inversion_image_pixelization_grid=None,
    ):

        self.origin = self.load_include(value=origin, name="origin")
        self.mask = self.load_include(value=mask, name="mask")
        self.grid = self.load_include(value=grid, name="grid")
        self.border = self.load_include(value=border, name="border")
        self.inversion_pixelization_grid = self.load_include(
            value=inversion_pixelization_grid, name="inversion_pixelization_grid"
        )
        self.inversion_grid = self.load_include(
            value=inversion_grid, name="inversion_grid"
        )
        self.inversion_border = self.load_include(
            value=inversion_border, name="inversion_border"
        )
        self.inversion_image_pixelization_grid = self.load_include(
            value=inversion_image_pixelization_grid,
            name="inversion_image_pixelization_grid",
        )

    @staticmethod
    def load_include(value, name):

        return (
            conf.instance.visualize_general.get(
                section_name="include", attribute_name=name, attribute_type=bool
            )
            if value is None
            else value
        )

    def grid_from_grid(self, grid):

        if self.grid:
            return grid
        else:
            return None

    def mask_from_grid(self, grid):

        if self.mask:
            return grid.mask
        else:
            return None

    def mask_from_masked_dataset(self, masked_dataset):

        if self.mask:
            return masked_dataset.mask
        else:
            return None

    def mask_from_fit(self, fit):
        """Get the masks of the fit if the masks should be plotted on the fit.

        Parameters
        -----------
        fit : datas.fitting.fitting.AbstractLensHyperFit
            The fit to the datas, which includes a lisrt of every model image, residual_map, chi-squareds, etc.
        include_mask : bool
            If *True*, the masks is plotted on the fit's datas.
        """
        if self.mask:
            return fit.mask
        else:
            return None

    def real_space_mask_from_fit(self, fit):
        """Get the masks of the fit if the masks should be plotted on the fit.

        Parameters
        -----------
        fit : datas.fitting.fitting.AbstractLensHyperFit
            The fit to the datas, which includes a lisrt of every model image, residual_map, chi-squareds, etc.
        include_mask : bool
            If *True*, the masks is plotted on the fit's datas.
        """
        if self.mask:
            return fit.settings_masked_dataset.real_space_mask
        else:
            return None


def include_key_from_dictionary(dictionary):

    include_key = None

    for key, value in dictionary.items():
        if isinstance(value, Include):
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


def kpc_per_arcsec_of_object_from_dictionary(dictionary):

    kpc_per_arcsec = None

    for key, value in dictionary.items():
        if hasattr(value, "kpc_per_arcsec"):
            return value.kpc_per_arcsec

    return kpc_per_arcsec


def set_include_and_plotter(func):
    @wraps(func)
    def wrapper(*args, **kwargs):

        include_key = include_key_from_dictionary(dictionary=kwargs)

        if include_key is not None:
            include = kwargs[include_key]
        else:
            include = Include()
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
            include = Include()
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

        title = plotter.labels.title_from_func(func=func)
        yunits = plotter.labels.yunits_from_func(func=func)
        xunits = plotter.labels.xunits_from_func(func=func)

        plotter = plotter.plotter_with_new_labels(
            title=title, yunits=yunits, xunits=xunits
        )

        filename = plotter.output.filename_from_func(func=func)

        plotter = plotter.plotter_with_new_output(filename=filename)

        kpc_per_arcsec = kpc_per_arcsec_of_object_from_dictionary(dictionary=kwargs)

        plotter = plotter.plotter_with_new_units(conversion_factor=kpc_per_arcsec)

        kwargs[plotter_key] = plotter

        return func(*args, **kwargs)

    return wrapper


def plot_array(
    array,
    mask=None,
    lines=None,
    positions=None,
    grid=None,
    array_overlay=None,
    include=None,
    plotter=None,
):

    if include is None:
        include = Include()

    if plotter is None:
        plotter = Plotter()

    plotter.plot_array(
        array=array,
        mask=mask,
        lines=lines,
        positions=positions,
        grid=grid,
        array_overlay=array_overlay,
        include_origin=include.origin,
        include_border=include.border,
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
        include = Include()

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
        include = Include()

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
