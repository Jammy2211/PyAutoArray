import matplotlib
from autoarray import conf

backend = conf.get_matplotlib_backend()
matplotlib.use(backend)
import matplotlib.pyplot as plt
import itertools
import numpy as np
import matplotlib.cm as cm

from autoarray.plotters import grid_plotters
from autoarray.operators.inversion import mappers


class MapperPlotter(grid_plotters.GridPlotter):
    def __init__(
        self,
        is_sub_plotter=False,
        use_scaled_units=None,
        unit_conversion_factor=None,
        figsize=None,
        aspect=None,
        cmap=None,
        norm=None,
        norm_min=None,
        norm_max=None,
        linthresh=None,
        linscale=None,
        cb_ticksize=None,
        cb_fraction=None,
        cb_pad=None,
        cb_tick_values=None,
        cb_tick_labels=None,
        titlesize=None,
        xlabelsize=None,
        ylabelsize=None,
        xyticksize=None,
        grid_pointsize=5,
        grid_pointcolor="k",
        label_title=None,
        label_yunits=None,
        label_xunits=None,
        label_yticks=None,
        label_xticks=None,
        output_path=None,
        output_format="show",
        output_filename=None,
    ):

        super(MapperPlotter, self).__init__(
            is_sub_plotter=is_sub_plotter,
            use_scaled_units=use_scaled_units,
            unit_conversion_factor=unit_conversion_factor,
            figsize=figsize,
            aspect=aspect,
            cmap=cmap,
            norm=norm,
            norm_min=norm_min,
            norm_max=norm_max,
            linthresh=linthresh,
            linscale=linscale,
            cb_ticksize=cb_ticksize,
            cb_fraction=cb_fraction,
            cb_pad=cb_pad,
            cb_tick_values=cb_tick_values,
            cb_tick_labels=cb_tick_labels,
            titlesize=titlesize,
            xlabelsize=xlabelsize,
            ylabelsize=ylabelsize,
            xyticksize=xyticksize,
            grid_pointsize=grid_pointsize,
            grid_pointcolor=grid_pointcolor,
            label_title=label_title,
            label_yunits=label_yunits,
            label_xunits=label_xunits,
            label_yticks=label_yticks,
            label_xticks=label_xticks,
            output_path=output_path,
            output_format=output_format,
            output_filename=output_filename,
        )

    def plotter_as_sub_plotter(
        self,
    ):

        return MapperPlotter(
            is_sub_plotter=True,
            use_scaled_units=self.use_scaled_units,
            unit_conversion_factor=self.unit_conversion_factor,
            figsize=self.figsize,
            aspect=self.aspect,
            cmap=self.cmap,
            norm=self.norm,
            norm_min=self.norm_min,
            norm_max=self.norm_max,
            linthresh=self.linthresh,
            linscale=self.linscale,
            cb_ticksize=self.cb_ticksize,
            cb_fraction=self.cb_fraction,
            cb_pad=self.cb_pad,
            cb_tick_values=self.cb_tick_values,
            cb_tick_labels=self.cb_tick_labels,
            titlesize=self.titlesize,
            xlabelsize=self.xlabelsize,
            ylabelsize=self.ylabelsize,
            xyticksize=self.xyticksize,
            grid_pointsize=self.grid_pointsize,
            label_title=self.label_title,
            label_yunits=self.label_yunits,
            label_xunits=self.label_xunits,
            label_yticks=self.label_yticks,
            label_xticks=self.label_xticks,
            output_path=self.output_path,
            output_format=self.output_format,
            output_filename=self.output_filename,
        )

    def plotter_with_new_labels_and_filename(
        self,
        label_title=None,
        label_yunits=None,
        label_xunits=None,
        output_filename=None,
    ):

        label_title = self.label_title if label_title is None else label_title
        label_yunits = self.label_yunits if label_yunits is None else label_yunits
        label_xunits = self.label_xunits if label_xunits is None else label_xunits
        output_filename = (
            self.output_filename if output_filename is None else output_filename
        )

        return MapperPlotter(
            is_sub_plotter=self.is_sub_plotter,
            use_scaled_units=self.use_scaled_units,
            unit_conversion_factor=self.unit_conversion_factor,
            figsize=self.figsize,
            aspect=self.aspect,
            cmap=self.cmap,
            norm=self.norm,
            norm_min=self.norm_min,
            norm_max=self.norm_max,
            linthresh=self.linthresh,
            linscale=self.linscale,
            cb_ticksize=self.cb_ticksize,
            cb_fraction=self.cb_fraction,
            cb_pad=self.cb_pad,
            cb_tick_values=self.cb_tick_values,
            cb_tick_labels=self.cb_tick_labels,
            titlesize=self.titlesize,
            xlabelsize=self.xlabelsize,
            ylabelsize=self.ylabelsize,
            xyticksize=self.xyticksize,
            grid_pointsize=self.grid_pointsize,
            label_title=label_title,
            label_yunits=label_yunits,
            label_xunits=label_xunits,
            label_yticks=self.label_yticks,
            label_xticks=self.label_xticks,
            output_path=self.output_path,
            output_format=self.output_format,
            output_filename=output_filename,
        )

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

        self.setup_figure()

        self.set_axis_limits(
            axis_limits=mapper.pixelization_grid.extent,
            grid=None,
            symmetric_around_centre=False,
        )

        self.set_yxticks(
            array=None,
            extent=mapper.pixelization_grid.extent,
        )

        self.plot_rectangular_pixelization_lines(mapper=mapper)

        self.set_title()
        self.set_yx_labels_and_ticksize()

        self.plot_centres(mapper=mapper, include_centres=include_centres)

        self.plot_mapper_grid(
            include_grid=include_grid,
            mapper=mapper,
        )

        self.plot_border(
            include_border=include_border,
            mapper=mapper,
        )

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

        self.output_figure(
            None,
        )
        self.close_figure()


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

        self.setup_figure()

        self.set_axis_limits(
            axis_limits=mapper.pixelization_grid.extent,
            grid=None,
            symmetric_around_centre=False,
        )

        self.set_yxticks(
            array=None,
            extent=mapper.pixelization_grid.extent,
        )

        regions_SP, vertices_SP = self.voronoi_finite_polygons_2d(voronoi=mapper.voronoi)

        color_values = source_pixel_values[:] / np.max(source_pixel_values)
        cmap = plt.get_cmap("jet")

        self.set_colorbar(
            cmap=cmap,
            color_values=source_pixel_values,
        )

        for region, index in zip(regions_SP, range(mapper.pixels)):
            polygon = vertices_SP[region]
            col = cmap(color_values[index])
            plt.fill(*zip(*polygon), alpha=0.7, facecolor=col, lw=0.0)

        self.set_title()
        self.set_yx_labels_and_ticksize(
        )

        self.plot_centres(mapper=mapper, include_centres=include_centres)

        self.plot_mapper_grid(
            include_grid=include_grid,
            mapper=mapper,
        )

        self.plot_border(
            include_border=include_border,
            mapper=mapper,
            as_subplot=True,
        )

        self.plot_lines(line_lists=lines)

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

        self.output_figure(
            None,
        )
        self.close_figure()


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


    def plot_rectangular_pixelization_lines(self, mapper):

        ys = np.linspace(
            mapper.pixelization_grid.scaled_minima[0],
            mapper.pixelization_grid.scaled_maxima[0],
            mapper.pixelization_grid.shape_2d[0] + 1,
        )
        xs = np.linspace(
            mapper.pixelization_grid.scaled_minima[1],
            mapper.pixelization_grid.scaled_maxima[1],
            mapper.pixelization_grid.shape_2d[1] + 1,
        )

        # grid lines
        for x in xs:
            plt.plot([x, x], [ys[0], ys[-1]], color="black", linestyle="-")
        for y in ys:
            plt.plot([xs[0], xs[-1]], [y, y], color="black", linestyle="-")


    def set_colorbar(
        self, color_values,
    ):

        cax = cm.ScalarMappable(cmap=self.cmap)
        cax.set_array(color_values)

        if self.cb_tick_values is None and self.cb_tick_labels is None:
            plt.colorbar(mappable=cax, fraction=self.cb_fraction, pad=self.cb_pad)
        elif self.cb_tick_values is not None and self.cb_tick_labels is not None:
            cb = plt.colorbar(
                mappable=cax, fraction=self.cb_fraction, pad=self.cb_pad, ticks=self.cb_tick_values
            )
            cb.ax.set_yticklabels(self.cb_tick_labels)


    def plot_centres(self, mapper, include_centres):

        if include_centres:

            pixelization_grid = mapper.pixelization_grid

            plt.scatter(y=pixelization_grid[:, 0], x=pixelization_grid[:, 1], s=3, c="r")


    def plot_mapper_grid(
        self,
        mapper,
        include_grid,
    ):

        if include_grid:

            self.plot_grid(
                grid=mapper.grid,
                bypass_limits=True,
            )


    def plot_border(
        self,
        mapper,
        include_border,
    ):

        if include_border:

            border = mapper.grid[mapper.grid.mask.regions._sub_border_1d_indexes]

            self.plot_grid(
                grid=border,
            )


    def plot_image_pixels(self, grid, image_pixels, point_colors):

        if image_pixels is not None:

            for image_pixel_set in image_pixels:
                color = next(point_colors)
                plt.scatter(
                    y=np.asarray(grid[image_pixel_set, 0]),
                    x=np.asarray(grid[image_pixel_set, 1]),
                    color=color,
                    s=10.0,
                )


    def plot_image_plane_source_pixels(self, grid, mapper, source_pixels, point_colors):

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


    def plot_source_plane_image_pixels(self, grid, image_pixels, point_colors):

        if image_pixels is not None:

            for image_pixel_set in image_pixels:
                color = next(point_colors)
                plt.scatter(
                    y=np.asarray(grid[[image_pixel_set], 0]),
                    x=np.asarray(grid[[image_pixel_set], 1]),
                    s=8,
                    color=color,
                )


    def plot_source_plane_source_pixels(self, grid, mapper, source_pixels, point_colors):

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
