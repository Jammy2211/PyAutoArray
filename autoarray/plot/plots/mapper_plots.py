from autoarray.plot.mat_wrap import mat_decorators
from autoarray.plot.mat_wrap import visuals as vis
from autoarray.plot.mat_wrap import include as inc
from autoarray.plot.mat_wrap import plotter as p
from autoarray.plot.plots import structure_plots
import typing


@mat_decorators.set_labels
def subplot_image_and_mapper(
    image,
    mapper,
    visuals_2d: vis.Visuals2D = vis.Visuals2D(),
    include_2d: inc.Include2D = inc.Include2D(),
    plotter_2d: p.Plotter2D = p.Plotter2D(),
    full_indexes=None,
    pixelization_indexes=None,
):

    plotter_2d = plotter_2d.plotter_for_subplot_from(func=subplot_image_and_mapper)

    number_subplots = 2

    plotter_2d.open_subplot_figure(number_subplots=number_subplots)

    plotter_2d.setup_subplot(number_subplots=number_subplots, subplot_index=1)

    structure_plots.plot_array(
        array=image, visuals_2d=visuals_2d, include_2d=include_2d, plotter_2d=plotter_2d
    )

    if full_indexes is not None:

        plotter_2d.index_scatter.scatter_grid_indexes(
            grid=mapper.source_full_grid.geometry.unmasked_grid_sub_1,
            indexes=full_indexes,
        )

    if pixelization_indexes is not None:

        indexes = mapper.full_indexes_from_pixelization_indexes(
            pixelization_indexes=pixelization_indexes
        )

        plotter_2d.index_scatter.scatter_grid_indexes(
            grid=mapper.source_full_grid.geometry.unmasked_grid_sub_1, indexes=indexes
        )

    plotter_2d.setup_subplot(number_subplots=number_subplots, subplot_index=2)

    structure_plots.plot_mapper_obj(
        mapper=mapper,
        visuals_2d=visuals_2d,
        include_2d=include_2d,
        plotter_2d=plotter_2d,
        full_indexes=full_indexes,
        pixelization_indexes=pixelization_indexes,
    )

    plotter_2d.output.subplot_to_figure()
    plotter_2d.figure.close()
