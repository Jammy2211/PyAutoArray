from autoarray.plot.plotter import visuals as ps


@ps.set_plot_defaults
def plot_array(array, visuals=None, include=None, plotter=None, extent_manual=None):

    visuals += include.visuals_from_array(array=array)

    plotter._plot_array(array=array, visuals=visuals, extent_manual=extent_manual)


@ps.set_plot_defaults
def plot_frame(frame, plotter=None, visuals=None, include=None):

    visuals += include.visuals_from_frame(frame=frame)

    plotter._plot_frame(frame=frame, visuals=visuals)


@ps.set_plot_defaults
def plot_grid(
    grid,
    plotter=None,
    visuals=None,
    include=None,
    color_array=None,
    axis_limits=None,
    indexes=None,
    symmetric_around_centre=True,
):

    visuals += include.visuals_from_grid(grid=grid)

    plotter._plot_grid(
        grid=grid,
        visuals=visuals,
        color_array=color_array,
        axis_limits=axis_limits,
        indexes=indexes,
        symmetric_around_centre=symmetric_around_centre,
    )


@ps.set_plot_defaults
def plot_mapper_obj(
    mapper,
    plotter=None,
    visuals=None,
    include=None,
    image_pixel_indexes=None,
    source_pixel_indexes=None,
):

    plotter._plot_mapper(
        mapper=mapper,
        visuals=visuals,
        image_pixel_indexes=image_pixel_indexes,
        source_pixel_indexes=source_pixel_indexes,
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

    plotter._plot_line(
        y=y,
        x=x,
        label=label,
        plot_axis_type=plot_axis_type,
        vertical_lines=vertical_lines,
        vertical_line_labels=vertical_line_labels,
    )
