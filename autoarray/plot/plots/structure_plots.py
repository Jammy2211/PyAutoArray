from autoarray.plot.mat_wrap import mat_decorators


@mat_decorators.set_plot_defaults_2d
@mat_decorators.set_labels
def plot_array(
    array, visuals_2d=None, include_2d=None, plotter_2d=None, extent_manual=None
):

    visuals_2d += include_2d.visuals_from_array(array=array)

    plotter_2d._plot_array(
        array=array, visuals_2d=visuals_2d, extent_manual=extent_manual
    )


@mat_decorators.set_plot_defaults_2d
@mat_decorators.set_labels
def plot_frame(frame, plotter_2d=None, visuals_2d=None, include_2d=None):

    visuals_2d += include_2d.visuals_from_frame(frame=frame)

    plotter_2d._plot_frame(frame=frame, visuals_2d=visuals_2d)


@mat_decorators.set_plot_defaults_2d
@mat_decorators.set_labels
def plot_grid(
    grid,
    plotter_2d=None,
    visuals_2d=None,
    include_2d=None,
    color_array=None,
    axis_limits=None,
    indexes=None,
    symmetric_around_centre=True,
):

    visuals_2d += include_2d.visuals_from_grid(grid=grid)

    plotter_2d._plot_grid(
        grid=grid,
        visuals_2d=visuals_2d,
        color_array=color_array,
        axis_limits=axis_limits,
        indexes=indexes,
        symmetric_around_centre=symmetric_around_centre,
    )


@mat_decorators.set_plot_defaults_2d
@mat_decorators.set_labels
def plot_mapper_obj(
    mapper,
    plotter_2d=None,
    visuals_2d=None,
    include_2d=None,
    image_pixel_indexes=None,
    source_pixel_indexes=None,
):

    plotter_2d._plot_mapper(
        mapper=mapper,
        visuals_2d=visuals_2d,
        image_pixel_indexes=image_pixel_indexes,
        source_pixel_indexes=source_pixel_indexes,
    )


@mat_decorators.set_plot_defaults_1d
@mat_decorators.set_labels
def plot_line(
    y,
    x,
    plotter_1d=None,
    visuals_1d=None,
    include_1d=None,
    label=None,
    plot_axis_type="semilogy",
    vertical_lines=None,
    vertical_line_labels=None,
):

    plotter_1d._plot_line(
        y=y,
        x=x,
        label=label,
        plot_axis_type=plot_axis_type,
        vertical_lines=vertical_lines,
        vertical_line_labels=vertical_line_labels,
    )
