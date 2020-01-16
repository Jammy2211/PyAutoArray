from autoarray import conf
import matplotlib

backend = conf.get_matplotlib_backend()
matplotlib.use(backend)
import matplotlib.pyplot as plt

import numpy as np
from functools import wraps
import copy

from autoarray import exc
from autoarray.plot import mat_objs
import itertools
from autoarray.operators.inversion import mappers


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

class AbstractPlotter(object):
    def __init__(
        self,
        units=mat_objs.Units(),
        figure=mat_objs.Figure(),
        cmap=mat_objs.ColorMap(),
        cb=mat_objs.ColorBar(),
            legend=mat_objs.Legend(),
            ticks=mat_objs.Ticks(),
            labels=mat_objs.Labels(),
            output=mat_objs.Output(),
        origin_scatterer=mat_objs.Scatterer(),
        mask_scatterer=mat_objs.Scatterer(),
        border_scatterer=mat_objs.Scatterer(),
        grid_scatterer=mat_objs.Scatterer(),
        positions_scatterer=mat_objs.Scatterer(),
        liner=mat_objs.Liner(),
    ):

        if not self.is_sub_plotter:
            load_setting_func = load_figure_setting
        else:
            load_setting_func = load_subplot_setting

        self.units = mat_objs.Units.from_instance_and_config(units=units)

        self.figure = mat_objs.Figure.from_instance_and_config(
            figure=figure, load_func=load_setting_func
        )
        self.cmap = mat_objs.ColorMap.from_instance_and_config(
            colormap=cmap, load_func=load_setting_func
        )
        self.cb = mat_objs.ColorBar.from_instance_and_config(
            cb=cb, load_func=load_setting_func
        )

        self.ticks = mat_objs.Ticks.from_instance_and_config(
            ticks=ticks, load_func=load_setting_func, units=self.units,
        )
        self.labels = mat_objs.Labels.from_instance_and_config(
            labels=labels,
            load_func=load_setting_func,
            units=self.units,
        )

        self.legend = mat_objs.Legend.from_instance_and_config(
            legend=legend,
            load_func=load_setting_func,
        )

        self.output = mat_objs.Output.from_instance_and_config(
            output=output,
            load_func=load_setting_func,
            is_sub_plotter=self.is_sub_plotter,
        )

        self.origin_scatterer = mat_objs.Scatterer.from_instance_and_config(
            scatterer=origin_scatterer, section="origin", load_func=load_setting_func)

        self.mask_scatterer = mat_objs.Scatterer.from_instance_and_config(
            scatterer=mask_scatterer, section="mask", load_func=load_setting_func)

        self.border_scatterer = mat_objs.Scatterer.from_instance_and_config(
            scatterer=border_scatterer, section="border", load_func=load_setting_func)

        self.grid_scatterer = mat_objs.Scatterer.from_instance_and_config(
            scatterer=grid_scatterer, section="grid", load_func=load_setting_func)

        self.positions_scatterer = mat_objs.Scatterer.from_instance_and_config(
            scatterer=positions_scatterer, section="positions", load_func=load_setting_func)

        self.liner = mat_objs.Liner.from_instance_and_config(liner=liner, section="liner", load_func=load_setting_func)


    @property
    def is_sub_plotter(self):
        raise NotImplementedError()

    def plot_array(
        self,
        array,
        mask=None,
        lines=None,
        positions=None,
        grid=None,
        include_origin=False,
        include_border=False,
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

        self.ticks.set_yticks(
            array=array,
            extent=extent,
        )
        self.ticks.set_xticks(
            array=array,
            extent=extent,
        )

        self.labels.set_title()
        self.labels.set_yunits(include_brackets=True)
        self.labels.set_xunits(include_brackets=True)

        self.cb.set()
        if include_origin:
            self.origin_scatterer.scatter_grids(grids=[array.origin])

        if mask is not None:
            self.mask_scatterer.scatter_grids(grids=mask.geometry.edge_grid.in_1d_binned)

        if include_border and mask is not None:
            self.border_scatterer.scatter_grids(grids=mask.geometry.border_grid.in_1d_binned)

        if grid is not None:
            self.grid_scatterer.scatter_grids(grids=grid)

        if positions is not None:
            self.positions_scatterer.scatter_grids(grids=positions)

        if lines is not None:
            self.liner.draw_grids(grids=lines)

        self.output.to_figure(structure=array)
        self.figure.close()

    def plot_grid(
        self,
        grid,
        color_array=None,
        axis_limits=None,
        indexes=None,
        lines=None,
        symmetric_around_centre=True,
        bypass_limits=False,
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

            self.grid_scatterer.scatter_grids(grids=grid)

        elif color_array is not None:

            plt.cm.get_cmap(self.cmap.cmap)
            self.grid_scatterer.scatter_colored_grid(grid=grid, color_array=color_array, cmap=self.cmap.cmap)
            self.cb.set()

        self.labels.set_title()
        self.labels.set_yunits(include_brackets=True)
        self.labels.set_xunits(include_brackets=True)

        if not bypass_limits:

            self.set_axis_limits(
                axis_limits=axis_limits,
                grid=grid,
                symmetric_around_centre=symmetric_around_centre,
            )

        self.ticks.set_yticks(
            array=None,
            extent=grid.extent,
            symmetric_around_centre=symmetric_around_centre,
        )
        self.ticks.set_xticks(
            array=None,
            extent=grid.extent,
            symmetric_around_centre=symmetric_around_centre,
        )

        if indexes is not None:
            self.grid_scatterer.scatter_grid_indexes(grid=grid, indexes=indexes)

        if lines is not None:
            self.liner.draw_grids(grids=lines)

        self.output.to_figure(structure=grid)
        self.figure.close()

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
        self.labels.set_title()

        if x is None:
            x = np.arange(len(y))

        self.liner.draw_y_vs_x(y=y, x=x, plot_axis_type=plot_axis_type, label=label)

        self.labels.set_yunits(include_brackets=False)
        self.labels.set_xunits(include_brackets=False)

        self.liner.draw_vertical_lines(
            vertical_lines=vertical_lines, vertical_line_labels=vertical_line_labels
        )

        self.legend.set()

        self.ticks.set_xticks(
            array=None,
            extent=[np.min(x), np.max(x)],
        )

        self.output.to_figure(structure=None)

        self.figure.close()

    def plot_mapper(
        self,
        mapper,
        include_centres=False,
        include_grid=False,
        include_border=False,
        image_pixels=None,
        source_pixels=None,
    ):

        if isinstance(mapper, mappers.MapperRectangular):

            self.plot_rectangular_mapper(
                mapper=mapper,
                include_centres=include_centres,
                include_grid=include_grid,
                include_border=include_border,
                image_pixels=image_pixels,
                source_pixels=source_pixels,
            )

        else:

            self.plot_voronoi_mapper(
                mapper=mapper,
                include_centres=include_centres,
                include_grid=include_grid,
                include_border=include_border,
                image_pixels=image_pixels,
                source_pixels=source_pixels,
            )

    def plot_rectangular_mapper(
        self,
        mapper,
        include_centres=False,
        include_grid=False,
        include_border=False,
        image_pixels=None,
        source_pixels=None,
    ):

        self.figure.open()

        self.set_axis_limits(
            axis_limits=mapper.pixelization_grid.extent,
            grid=None,
            symmetric_around_centre=False,
        )

        self.ticks.set_yticks(
            array=None,
            extent=mapper.pixelization_grid.extent,
        )
        self.ticks.set_xticks(
            array=None,
            extent=mapper.pixelization_grid.extent,
        )

        self.liner.draw_rectangular_grid_lines(extent=mapper.pixelization_grid.extent, shape_2d=mapper.shape_2d)

        self.labels.set_title()
        self.labels.set_yunits(include_brackets=True)
        self.labels.set_xunits(include_brackets=True)

        # self.scatter_centres(mapper=mapper, include_centres=include_centres)
        #
        # self.scatter_grid(include_grid=include_grid, mapper=mapper)
        #
        # self.scatter_border(include_border=include_border, mapper=mapper)

        point_colors = itertools.cycle(["y", "r", "k", "g", "m"])
        self.scatter_source_plane_image_pixels(
            grid=mapper.grid, image_pixels=image_pixels, point_colors=point_colors
        )
        self.scatter_source_plane_source_pixels(
            grid=mapper.grid,
            mapper=mapper,
            source_pixels=source_pixels,
            point_colors=point_colors,
        )

        self.output.to_figure(structure=None)
        self.figure.close()

    def plot_voronoi_mapper(
        self,
        mapper,
        source_pixel_values,
        include_centres=True,
        include_grid=True,
        include_border=False,
        lines=None,
        image_pixels=None,
        source_pixels=None,
    ):

        self.figure.open()

        self.set_axis_limits(
            axis_limits=mapper.pixelization_grid.extent,
            grid=None,
            symmetric_around_centre=False,
        )

        self.ticks.set_yticks(
            array=None,
            extent=mapper.pixelization_grid.extent,
        )
        self.ticks.set_xticks(
            array=None,
            extent=mapper.pixelization_grid.extent,
        )

        regions_SP, vertices_SP = self.voronoi_finite_polygons_2d(
            voronoi=mapper.voronoi
        )

        color_values = source_pixel_values[:] / np.max(source_pixel_values)
        cmap = plt.get_cmap("jet")

        self.cb.set_with_values(cmap=cmap, color_values=source_pixel_values)

        for region, index in zip(regions_SP, range(mapper.pixels)):
            polygon = vertices_SP[region]
            col = cmap(color_values[index])
            plt.fill(*zip(*polygon), alpha=0.7, facecolor=col, lw=0.0)

        self.labels.set_title()
        self.labels.set_yunits(include_brackets=True)
        self.labels.set_xunits(include_brackets=True)

        self.plot_centres(mapper=mapper, include_centres=include_centres)

        self.plot_mapper_grid(include_grid=include_grid, mapper=mapper)

        self.plot_border(include_border=include_border, mapper=mapper)

        if lines is not None:
            self.liner.draw_grids(grids=lines)

        point_colors = itertools.cycle(["y", "r", "k", "g", "m"])
        self.plot_source_plane_image_pixels(
            grid=mapper.grid, image_pixels=image_pixels, point_colors=point_colors
        )
        self.plot_source_plane_source_pixels(
            grid=mapper.grid,
            mapper=mapper,
            source_pixels=source_pixels,
            point_colors=point_colors,
        )

        self.output.to_figure(structure=None)
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

    def voronoi_finite_polygons_2d(self, vor, radius=None):
        """
        Reconstruct infinite voronoi regions in a 2D diagram to finite
        regions.
        Parameters
        ----------
        vor : Voronoi
            Input diagram
        radius : float, optional
            Distance to 'points at infinity'.
        Returns
        -------
        regions : list of tuples
            Indices of vertices in each revised Voronoi regions.
        vertices : list of tuples
            Coordinates for revised Voronoi vertices. Same as coordinates
            of input vertices, with 'points at infinity' appended to the
            end.
        """

        if vor.points.shape[1] != 2:
            raise ValueError("Requires 2D input")

        new_regions = []
        new_vertices = vor.vertices.tolist()

        center = vor.points.mean(axis=0)
        if radius is None:
            radius = vor.points.ptp().max() * 2

        # Construct a map containing all ridges for a given point
        all_ridges = {}
        for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
            all_ridges.setdefault(p1, []).append((p2, v1, v2))
            all_ridges.setdefault(p2, []).append((p1, v1, v2))

        # Reconstruct infinite regions
        for p1, region in enumerate(vor.point_region):
            vertices = vor.regions[region]

            if all(v >= 0 for v in vertices):
                # finite region
                new_regions.append(vertices)
                continue

            # reconstruct a non-finite region
            ridges = all_ridges[p1]
            new_region = [v for v in vertices if v >= 0]

            for p2, v1, v2 in ridges:
                if v2 < 0:
                    v1, v2 = v2, v1
                if v1 >= 0:
                    # finite ridge: already in the region
                    continue

                # Compute the missing endpoint of an infinite ridge

                t = vor.points[p2] - vor.points[p1]  # tangent
                t /= np.linalg.norm(t)
                n = np.array([-t[1], t[0]])  # hyper

                midpoint = vor.points[[p1, p2]].mean(axis=0)
                direction = np.sign(np.dot(midpoint - center, n)) * n
                far_point = vor.vertices[v2] + direction * radius

                new_region.append(len(new_vertices))
                new_vertices.append(far_point.tolist())

            # sort region counterclockwise
            vs = np.asarray([new_vertices[v] for v in new_region])
            c = vs.mean(axis=0)
            angles = np.arctan2(vs[:, 1] - c[1], vs[:, 0] - c[0])
            new_region = np.array(new_region)[np.argsort(angles)]

            # finish
            new_regions.append(new_region.tolist())

        return new_regions, np.asarray(new_vertices)

    def scatter_image_pixels(self, grid, image_pixels, point_colors):

        if image_pixels is not None:

            for image_pixel_set in image_pixels:
                color = next(point_colors)
                plt.scatter(
                    y=np.asarray(grid[image_pixel_set, 0]),
                    x=np.asarray(grid[image_pixel_set, 1]),
                    color=color,
                    s=10.0,
                )

    def scatter_image_plane_source_pixels(self, grid, mapper, source_pixels, point_colors):

        if source_pixels is not None:

            for source_pixel_set in source_pixels:
                color = next(point_colors)
                for source_pixel in source_pixel_set:
                    plt.scatter(
                        y=np.asarray(
                            grid[
                                mapper.all_sub_mask_1d_indexes_for_pixelization_1d_index[
                                    source_pixel
                                ],
                                0,
                            ]
                        ),
                        x=np.asarray(
                            grid[
                                mapper.all_sub_mask_1d_indexes_for_pixelization_1d_index[
                                    source_pixel
                                ],
                                1,
                            ]
                        ),
                        s=8,
                        color=color,
                    )

    def scatter_source_plane_image_pixels(self, grid, image_pixels, point_colors):

        if image_pixels is not None:

            for image_pixel_set in image_pixels:
                color = next(point_colors)
                plt.scatter(
                    y=np.asarray(grid[[image_pixel_set], 0]),
                    x=np.asarray(grid[[image_pixel_set], 1]),
                    s=8,
                    color=color,
                )

    def scatter_source_plane_source_pixels(
            self, grid, mapper, source_pixels, point_colors
    ):

        if source_pixels is not None:

            for source_pixel_set in source_pixels:
                color = next(point_colors)
                for source_pixel in source_pixel_set:
                    plt.scatter(
                        y=np.asarray(
                            grid[
                                mapper.all_sub_mask_1d_indexes_for_pixelization_1d_index[
                                    source_pixel
                                ],
                                0,
                            ]
                        ),
                        x=np.asarray(
                            grid[
                                mapper.all_sub_mask_1d_indexes_for_pixelization_1d_index[
                                    source_pixel
                                ],
                                1,
                            ]
                        ),
                        s=8,
                        color=color,
                    )

    def plotter_with_new_labels(self, labels=mat_objs.Labels()):

        plotter = copy.deepcopy(self)

        plotter.labels.title = (
            labels.title if labels.title is not None else self.labels.title
        )
        plotter.labels._yunits = (
            labels._yunits if labels._yunits is not None else self.labels._yunits
        )
        plotter.labels._xunits = (
            labels._xunits if labels._xunits is not None else self.labels._xunits
        )
        plotter.labels.titlesize = (
            labels.titlesize if labels.titlesize is not None else self.labels.titlesize
        )
        plotter.labels.ysize = (
            labels.ysize if labels.ysize is not None else self.labels.ysize
        )
        plotter.labels.xsize = (
            labels.xsize if labels.xsize is not None else self.labels.xsize
        )

        return plotter

    def plotter_with_new_units(self, units=mat_objs.Units()):

        plotter = copy.deepcopy(self)

        new_units = mat_objs.Units()

        new_units.use_scaled = units.use_scaled = (
            units.use_scaled
            if units.use_scaled is not None
            else self.units.use_scaled
        )

        new_units.in_kpc = units.in_kpc = (
            units.in_kpc
            if units.in_kpc is not None
            else self.units.in_kpc
        )

        new_units.conversion_factor = units.conversion_factor = (
            units.conversion_factor
            if units.conversion_factor is not None
            else self.units.conversion_factor
        )

        plotter.units = new_units

        plotter.ticks.units = new_units

        plotter.labels.units = new_units

        return plotter

    def plotter_with_new_output(self, output=mat_objs.Output()):

        plotter = copy.deepcopy(self)

        plotter.output.path = (
            output.path if output.path is not None else self.output.path
        )

        plotter.output.filename = (
            output.filename if output.filename is not None else self.output.filename
        )

        plotter.output._format = (
            output._format if output._format is not None else self.output._format
        )

        return plotter


class Plotter(AbstractPlotter):
    def __init__(
        self,
        units=mat_objs.Units(),
        figure=mat_objs.Figure(),
        cmap=mat_objs.ColorMap(),
        cb=mat_objs.ColorBar(),
        ticks=mat_objs.Ticks(),
        labels=mat_objs.Labels(),
        legend=mat_objs.Legend(),
        output=mat_objs.Output(),
        origin_scatterer=mat_objs.Scatterer(),
        mask_scatterer=mat_objs.Scatterer(),
        border_scatterer=mat_objs.Scatterer(),
        grid_scatterer=mat_objs.Scatterer(),
        positions_scatterer=mat_objs.Scatterer(),
        liner=mat_objs.Liner(),
    ):

        super(Plotter, self).__init__(
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
            liner=liner,
        )

    @property
    def is_sub_plotter(self):
        return False


class SubPlotter(AbstractPlotter):
    def __init__(
        self,
            units=mat_objs.Units(),
        figure=mat_objs.Figure(),
        cmap=mat_objs.ColorMap(),
        cb=mat_objs.ColorBar(),
            legend=mat_objs.Legend(),
            ticks=mat_objs.Ticks(),
            labels=mat_objs.Labels(),
            output=mat_objs.Output(),
        origin_scatterer=mat_objs.Scatterer(),
        mask_scatterer=mat_objs.Scatterer(),
        border_scatterer=mat_objs.Scatterer(),
            grid_scatterer=mat_objs.Scatterer(),
        positions_scatterer=mat_objs.Scatterer(),
        liner=mat_objs.Liner(),
    ):

        super(SubPlotter, self).__init__(
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
            liner=liner,
        )

    def open_subplot_figure(self, number_subplots):
        """Setup a figure for plotting an image.

        Parameters
        -----------
        figsize : (int, int)
            The size of the figure in (rows, columns).
        as_subplot : bool
            If the figure is a subplot, the setup_figure function is omitted to ensure that each subplot does not create a \
            new figure and so that it can be output using the *output.output_figure(structure=None, is_sub_plotter=False)* function.
        """

        figsize = self.get_subplot_figsize(number_subplots=number_subplots)
        plt.figure(figsize=figsize)

    def setup_subplot(self, number_subplots, subplot_index):
        rows, columns = self.get_subplot_rows_columns(number_subplots=number_subplots)
        plt.subplot(rows, columns, subplot_index)

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

    @property
    def is_sub_plotter(self):
        return True


class Include(object):
    def __init__(
        self,
        origin=None,
        mask=None,
        grid=None,
        border=None,
        inversion_centres=None,
        inversion_grid=None,
        inversion_border=None,
        inversion_image_pixelization_grid=None,
    ):

        self.origin = self.load_include(value=origin, name="origin")
        self.mask = self.load_include(value=mask, name="mask")
        self.grid = self.load_include(value=grid, name="grid")
        self.border = self.load_include(value=border, name="border")
        self.inversion_centres = self.load_include(
            value=inversion_centres, name="inversion_centres"
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
            return fit.masked_dataset.real_space_mask
        else:
            return None


def plotter_key_from_dictionary(dictionary):

    plotter_key = None

    for key, value in dictionary.items():
        if isinstance(value, AbstractPlotter):
            plotter_key = key

    if plotter_key is None:
        raise exc.PlottingException(
            "The plot function called could not locate a Plotter in the kwarg arguments"
            "in order to set the labels."
        )

    return plotter_key


def kpc_per_arcsec_of_object_from_dictionary(dictionary):

    kpc_per_arcsec = None

    for key, value in dictionary.items():
        if hasattr(value, "kpc_per_arcsec"):
            return value.kpc_per_arcsec

    return kpc_per_arcsec


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

        plotter = plotter.plotter_with_new_output(mat_objs.Output(filename=filename))

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
            labels=mat_objs.Labels(title=title, yunits=yunits, xunits=xunits)
        )

        filename = plotter.output.filename_from_func(func=func)

        plotter = plotter.plotter_with_new_output(mat_objs.Output(filename=filename))

        kpc_per_arcsec = kpc_per_arcsec_of_object_from_dictionary(dictionary=kwargs)

        plotter = plotter.plotter_with_new_units(
            units=mat_objs.Units(conversion_factor=kpc_per_arcsec),
        )

        kwargs[plotter_key] = plotter

        return func(*args, **kwargs)

    return wrapper


def plot_array(
        array,
        mask=None,
        lines=None,
        positions=None,
        grid=None,
        include_origin=False,
        include_border=False,
        plotter=Plotter()):

    plotter.plot_array(array=array, mask=mask, lines=lines, positions=positions, grid=grid, include_origin=include_origin, include_border=include_border)