import matplotlib
from autoarray import conf

backend = conf.get_matplotlib_backend()
matplotlib.use(backend)
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import itertools
from scipy.spatial import Voronoi

from autoarray.plotters import imaging_plotters, grid_plotters
from autoarray.util import plotter_util
from autoarray.operators.inversion import mappers


def image_and_mapper(
    imaging,
    mapper,
    mask=None,
    positions=None,
    include_centres=False,
    include_grid=False,
    include_border=False,
    image_pixels=None,
    source_pixels=None,
    use_scaled_units=True,
    unit_conversion_factor=None,
    unit_label="scaled",
    output_path=None,
    output_filename="image_and_mapper",
    output_format="show",
):

    rows, columns, figsize = plotter_util.get_subplot_rows_columns_figsize(
        number_subplots=2
    )
    plt.figure(figsize=figsize)
    plt.subplot(rows, columns, 1)

    imaging_plotters.image(
        imaging=imaging,
        mask=mask,
        positions=positions,
        as_subplot=True,
        use_scaled_units=use_scaled_units,
        unit_conversion_factor=unit_conversion_factor,
        unit_label=unit_label,
        xyticksize=16,
        norm="linear",
        norm_min=None,
        norm_max=None,
        linthresh=0.05,
        linscale=0.01,
        figsize=None,
        aspect="square",
        cmap="jet",
        cb_ticksize=10,
        titlesize=10,
        xlabelsize=10,
        ylabelsize=10,
        output_path=output_path,
        output_format=output_format,
    )

    point_colors = itertools.cycle(["y", "r", "k", "g", "m"])
    plot_image_pixels(
        grid=mapper.grid.geometry.unmasked_grid,
        image_pixels=image_pixels,
        point_colors=point_colors,
    )
    plot_image_plane_source_pixels(
        grid=mapper.grid.geometry.unmasked_grid,
        mapper=mapper,
        source_pixels=source_pixels,
        point_colors=point_colors,
    )

    plt.subplot(rows, columns, 2)

    plot_mapper(
        mapper=mapper,
        include_centres=include_centres,
        include_grid=include_grid,
        include_border=include_border,
        image_pixels=image_pixels,
        source_pixels=source_pixels,
        as_subplot=True,
        unit_label=unit_label,
        use_scaled_units=use_scaled_units,
        unit_conversion_factor=unit_conversion_factor,
        figsize=None,
    )

    plotter_util.output_subplot_array(
        output_path=output_path,
        output_filename=output_filename,
        output_format=output_format,
    )
    plt.close()


def plot_mapper(
    mapper,
    include_centres=False,
    include_grid=False,
    include_border=False,
    image_pixels=None,
    source_pixels=None,
    as_subplot=False,
    use_scaled_units=True,
    unit_label="scaled",
    unit_conversion_factor=None,
    xyticksize=16,
    figsize=(7, 7),
    title="Mapper",
    titlesize=16,
    xlabelsize=16,
    ylabelsize=16,
    output_path=None,
    output_filename="mapper",
    output_format="show",
):

    if isinstance(mapper, mappers.MapperRectangular):

        rectangular_mapper(
            mapper=mapper,
            include_centres=include_centres,
            include_grid=include_grid,
            include_border=include_border,
            image_pixels=image_pixels,
            source_pixels=source_pixels,
            as_subplot=as_subplot,
            use_scaled_units=use_scaled_units,
            unit_label=unit_label,
            unit_conversion_factor=unit_conversion_factor,
            xyticksize=xyticksize,
            figsize=figsize,
            title=title,
            titlesize=titlesize,
            xlabelsize=xlabelsize,
            ylabelsize=ylabelsize,
            output_path=output_path,
            output_filename=output_filename,
            output_format=output_format,
        )


def rectangular_mapper(
    mapper,
    include_centres=False,
    include_grid=False,
    include_border=False,
    image_pixels=None,
    source_pixels=None,
    as_subplot=False,
    use_scaled_units=True,
    unit_label="scaled",
    unit_conversion_factor=None,
    xyticksize=16,
    figsize=(7, 7),
    title="Rectangular Mapper",
    titlesize=16,
    xlabelsize=16,
    ylabelsize=16,
    output_path=None,
    output_filename="rectangular_mapper",
    output_format="show",
):

    plotter_util.setup_figure(figsize=figsize, as_subplot=as_subplot)

    grid_plotters.set_axis_limits(
        axis_limits=mapper.pixelization_grid.extent,
        grid=None,
        symmetric_around_centre=False,
    )

    plotter_util.set_yxticks(
        array=None,
        extent=mapper.pixelization_grid.extent,
        use_scaled_units=use_scaled_units,
        unit_conversion_factor=unit_conversion_factor,
        xticks_manual=None,
        yticks_manual=None,
    )

    plot_rectangular_pixelization_lines(mapper=mapper)

    plotter_util.set_title(title=title, titlesize=titlesize)
    plotter_util.set_yx_labels_and_ticksize(
        unit_label_y=unit_label,
        unit_label_x=unit_label,
        xlabelsize=xlabelsize,
        ylabelsize=ylabelsize,
        xyticksize=xyticksize,
    )

    plot_centres(include_centres=include_centres, mapper=mapper)

    plot_mapper_grid(
        include_grid=include_grid,
        mapper=mapper,
        as_subplot=True,
        unit_label=unit_label,
        unit_conversion_factor=unit_conversion_factor,
        pointsize=10,
        xyticksize=xyticksize,
        title=title,
        titlesize=titlesize,
        xlabelsize=xlabelsize,
        ylabelsize=ylabelsize,
    )

    plot_border(
        include_border=include_border,
        mapper=mapper,
        as_subplot=True,
        unit_label=unit_label,
        unit_conversion_factor=unit_conversion_factor,
        pointsize=30,
        xyticksize=xyticksize,
        title=title,
        titlesize=titlesize,
        xlabelsize=xlabelsize,
        ylabelsize=ylabelsize,
    )

    point_colors = itertools.cycle(["y", "r", "k", "g", "m"])
    plot_source_plane_image_pixels(
        grid=mapper.grid, image_pixels=image_pixels, point_colors=point_colors
    )
    plot_source_plane_source_pixels(
        grid=mapper.grid,
        mapper=mapper,
        source_pixels=source_pixels,
        point_colors=point_colors,
    )

    plotter_util.output_figure(
        None,
        as_subplot=as_subplot,
        output_path=output_path,
        output_filename=output_filename,
        output_format=output_format,
    )
    plotter_util.close_figure(as_subplot=as_subplot)


def voronoi_mapper(
    mapper,
    source_pixel_values,
    include_centres=True,
    lines=None,
    include_grid=True,
    include_border=False,
    image_pixels=None,
    source_pixels=None,
    as_subplot=False,
    use_scaled_units=True,
    unit_label="scaled",
    unit_conversion_factor=None,
    xyticksize=16,
    figsize=(7, 7),
    title="Rectangular Mapper",
    titlesize=16,
    xlabelsize=16,
    ylabelsize=16,
    cb_ticksize=10,
    cb_fraction=0.047,
    cb_pad=0.01,
    cb_tick_values=None,
    cb_tick_labels=None,
    output_path=None,
    output_filename="voronoi_mapper",
    output_format="show",
):

    plotter_util.setup_figure(figsize=figsize, as_subplot=as_subplot)

    grid_plotters.set_axis_limits(
        axis_limits=mapper.pixelization_grid.extent,
        grid=None,
        symmetric_around_centre=False,
    )

    plotter_util.set_yxticks(
        array=None,
        extent=mapper.pixelization_grid.extent,
        use_scaled_units=use_scaled_units,
        unit_conversion_factor=unit_conversion_factor,
        xticks_manual=None,
        yticks_manual=None,
    )

    regions_SP, vertices_SP = voronoi_finite_polygons_2d(mapper.voronoi)

    color_values = source_pixel_values[:] / np.max(source_pixel_values)
    cmap = plt.get_cmap("jet")

    set_colorbar(
        cmap=cmap,
        color_values=source_pixel_values,
        cb_fraction=cb_fraction,
        cb_pad=cb_pad,
        cb_tick_values=cb_tick_values,
        cb_tick_labels=cb_tick_labels,
    )

    for region, index in zip(regions_SP, range(mapper.pixels)):
        polygon = vertices_SP[region]
        col = cmap(color_values[index])
        plt.fill(*zip(*polygon), alpha=0.7, facecolor=col, lw=0.0)

    plotter_util.set_title(title=title, titlesize=titlesize)
    plotter_util.set_yx_labels_and_ticksize(
        unit_label_y=unit_label,
        unit_label_x=unit_label,
        xlabelsize=xlabelsize,
        ylabelsize=ylabelsize,
        xyticksize=xyticksize,
    )

    plot_centres(include_centres=include_centres, mapper=mapper)

    plot_mapper_grid(
        include_grid=include_grid,
        mapper=mapper,
        as_subplot=True,
        unit_label=unit_label,
        unit_conversion_factor=unit_conversion_factor,
        pointsize=10,
        xyticksize=xyticksize,
        title=title,
        titlesize=titlesize,
        xlabelsize=xlabelsize,
        ylabelsize=ylabelsize,
    )

    plot_border(
        include_border=include_border,
        mapper=mapper,
        as_subplot=True,
        unit_label=unit_label,
        unit_conversion_factor=unit_conversion_factor,
        pointsize=30,
        xyticksize=xyticksize,
        title=title,
        titlesize=titlesize,
        xlabelsize=xlabelsize,
        ylabelsize=ylabelsize,
    )

    plotter_util.plot_lines(line_lists=lines)

    point_colors = itertools.cycle(["y", "r", "k", "g", "m"])
    plot_source_plane_image_pixels(
        grid=mapper.grid, image_pixels=image_pixels, point_colors=point_colors
    )
    plot_source_plane_source_pixels(
        grid=mapper.grid,
        mapper=mapper,
        source_pixels=source_pixels,
        point_colors=point_colors,
    )

    plotter_util.output_figure(
        None,
        as_subplot=as_subplot,
        output_path=output_path,
        output_filename=output_filename,
        output_format=output_format,
    )
    plotter_util.close_figure(as_subplot=as_subplot)


def voronoi_finite_polygons_2d(vor, radius=None):
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


def plot_rectangular_pixelization_lines(mapper,):

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
    cmap, color_values, cb_fraction, cb_pad, cb_tick_values, cb_tick_labels
):

    cax = cm.ScalarMappable(cmap=cmap)
    cax.set_array(color_values)

    if cb_tick_values is None and cb_tick_labels is None:
        plt.colorbar(mappable=cax, fraction=cb_fraction, pad=cb_pad)
    elif cb_tick_values is not None and cb_tick_labels is not None:
        cb = plt.colorbar(
            mappable=cax, fraction=cb_fraction, pad=cb_pad, ticks=cb_tick_values
        )
        cb.ax.set_yticklabels(cb_tick_labels)


def plot_centres(include_centres, mapper):

    if include_centres:

        pixelization_grid = mapper.pixelization_grid

        plt.scatter(y=pixelization_grid[:, 0], x=pixelization_grid[:, 1], s=3, c="r")


def plot_mapper_grid(
    include_grid,
    mapper,
    as_subplot,
    unit_label,
    unit_conversion_factor,
    pointsize,
    xyticksize,
    title,
    titlesize,
    xlabelsize,
    ylabelsize,
):

    if include_grid:

        grid_plotters.plot_grid(
            grid=mapper.grid,
            as_subplot=as_subplot,
            unit_label_y=unit_label,
            unit_label_x=unit_label,
            unit_conversion_factor=unit_conversion_factor,
            pointsize=pointsize,
            xyticksize=xyticksize,
            title=title,
            titlesize=titlesize,
            xlabelsize=xlabelsize,
            ylabelsize=ylabelsize,
            bypass_limits=True,
        )


def plot_border(
    include_border,
    mapper,
    as_subplot,
    unit_label,
    unit_conversion_factor,
    pointsize,
    xyticksize,
    title,
    titlesize,
    xlabelsize,
    ylabelsize,
):

    if include_border:

        border = mapper.grid[mapper.grid.mask.regions._sub_border_1d_indexes]

        grid_plotters.plot_grid(
            grid=border,
            as_subplot=as_subplot,
            unit_label_y=unit_label,
            unit_label_x=unit_label,
            unit_conversion_factor=unit_conversion_factor,
            pointsize=pointsize,
            pointcolor="y",
            xyticksize=xyticksize,
            title=title,
            titlesize=titlesize,
            xlabelsize=xlabelsize,
            ylabelsize=ylabelsize,
        )


def plot_image_pixels(grid, image_pixels, point_colors):

    if image_pixels is not None:

        for image_pixel_set in image_pixels:
            color = next(point_colors)
            plt.scatter(
                y=np.asarray(grid[image_pixel_set, 0]),
                x=np.asarray(grid[image_pixel_set, 1]),
                color=color,
                s=10.0,
            )


def plot_image_plane_source_pixels(grid, mapper, source_pixels, point_colors):

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


def plot_source_plane_image_pixels(grid, image_pixels, point_colors):

    if image_pixels is not None:

        for image_pixel_set in image_pixels:
            color = next(point_colors)
            plt.scatter(
                y=np.asarray(grid[[image_pixel_set], 0]),
                x=np.asarray(grid[[image_pixel_set], 1]),
                s=8,
                color=color,
            )


def plot_source_plane_source_pixels(grid, mapper, source_pixels, point_colors):

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
