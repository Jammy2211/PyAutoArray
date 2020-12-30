from autoarray.plot.mat_wrap import mat_decorators
from autoarray.plot.mat_wrap import visuals as vis
from autoarray.plot.mat_wrap import include as inc
from autoarray.plot.mat_wrap import plotter as p
from autoarray.structures import arrays, frames, grids
from autoarray.inversion import mappers
import typing


@mat_decorators.set_labels
def plot_array(
    array: arrays.Array,
    visuals_2d: vis.Visuals2D = vis.Visuals2D(),
    include_2d: inc.Include2D = inc.Include2D(),
    plotter_2d: p.Plotter2D = p.Plotter2D(),
    extent_manual=None,
):

    visuals_2d += include_2d.visuals_from_array(array=array)

    plotter_2d._plot_array(
        array=array, visuals_2d=visuals_2d, extent_manual=extent_manual
    )


@mat_decorators.set_labels
def plot_frame(
    frame: frames.Frame,
    plotter_2d: p.Plotter2D = p.Plotter2D(),
    visuals_2d: vis.Visuals2D = vis.Visuals2D(),
    include_2d: inc.Include2D = inc.Include2D(),
):

    visuals_2d += include_2d.visuals_from_frame(frame=frame)

    plotter_2d._plot_frame(frame=frame, visuals_2d=visuals_2d)


@mat_decorators.set_labels
def plot_grid(
    grid: grids.Grid,
    plotter_2d: p.Plotter2D = p.Plotter2D(),
    visuals_2d: vis.Visuals2D = vis.Visuals2D(),
    include_2d: inc.Include2D = inc.Include2D(),
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


@mat_decorators.set_labels
def plot_mapper_obj(
    mapper: mappers.Mapper,
    plotter_2d: p.Plotter2D = p.Plotter2D(),
    visuals_2d: vis.Visuals2D = vis.Visuals2D(),
    include_2d: inc.Include2D = inc.Include2D(),
    source_pixelilzation_values=None,
    full_indexes=None,
    pixelization_indexes=None,
):

    visuals_2d += include_2d.visuals_of_source_from_mapper(mapper=mapper)

    plotter_2d._plot_mapper(
        mapper=mapper,
        visuals_2d=visuals_2d,
        source_pixelilzation_values=source_pixelilzation_values,
        full_indexes=full_indexes,
        pixelization_indexes=pixelization_indexes,
    )


@mat_decorators.set_labels
def plot_line(
    y,
    x,
    plotter_1d: p.Plotter1D = p.Plotter1D(),
    visuals_1d: vis.Visuals1D = vis.Visuals1D(),
    include_1d: inc.Include1D = inc.Include1D(),
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
