import itertools

from autoarray.plotters import plotters
from autoarray.plots import imaging_plots


@plotters.set_labels
def subplot_image_and_mapper(
    imaging,
    mapper,
    mask=None,
    positions=None,
    image_pixels=None,
    source_pixels=None,
    include=plotters.Include(),
    sub_plotter=plotters.SubPlotter(),
):

    number_subplots = 2

    sub_plotter.setup_subplot_figure(number_subplots=number_subplots)

    sub_plotter.setup_subplot(number_subplots=number_subplots, subplot_index=1)

    imaging_plots.image(
        imaging=imaging,
        mask=mask,
        positions=positions,
        include=include,
        plotter=sub_plotter,
    )

    point_colors = itertools.cycle(["y", "r", "k", "g", "m"])
    sub_plotter.mapper.plot_image_pixels(
        grid=mapper.grid.geometry.unmasked_grid,
        image_pixels=image_pixels,
        point_colors=point_colors,
    )
    sub_plotter.mapper.plot_image_plane_source_pixels(
        mapper=mapper,
        grid=mapper.grid.geometry.unmasked_grid,
        source_pixels=source_pixels,
        point_colors=point_colors,
    )

    sub_plotter.setup_subplot(number_subplots=number_subplots, subplot_index=2)

    mapper_grid(
        mapper=mapper,
        image_pixels=image_pixels,
        source_pixels=source_pixels,
        include=include,
        plotter=sub_plotter,
    )

    sub_plotter.output.subplot_to_figure()
    sub_plotter.close_figure()


@plotters.set_labels
def mapper_grid(
    mapper,
    image_pixels=None,
    source_pixels=None,
    include=plotters.Include(),
    plotter=plotters.Plotter(),
):

    plotter.mapper.plot(
        mapper=mapper,
        image_pixels=image_pixels,
        source_pixels=source_pixels,
        include_centres=include.inversion_centres,
        include_grid=include.inversion_grid,
        include_border=include.inversion_border,
    )
