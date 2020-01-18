import itertools

from autoarray.plot import plotters
from autoarray.plot import imaging_plots


@plotters.set_labels
def subplot_image_and_mapper(
    imaging,
    mapper,
    mask=None,
    positions=None,
    image_pixel_indexes=None,
    source_pixel_indexes=None,
    include=plotters.Include(),
    sub_plotter=plotters.SubPlotter(),
):

    number_subplots = 2

    sub_plotter.open_subplot_figure(number_subplots=number_subplots)

    sub_plotter.setup_subplot(number_subplots=number_subplots, subplot_index=1)

    imaging_plots.image(
        imaging=imaging,
        mask=mask,
        positions=positions,
        include=include,
        plotter=sub_plotter,
    )

    if image_pixel_indexes is not None:

        sub_plotter.index_scatterer.scatter_grid_indexes(
            grid=mapper.grid.geometry.unmasked_grid, indexes=image_pixel_indexes
        )

    if source_pixel_indexes is not None:

        indexes = mapper.image_pixel_indexes_from_source_pixel_indexes(
            source_pixel_indexes=source_pixel_indexes
        )

        sub_plotter.index_scatterer.scatter_grid_indexes(
            grid=mapper.grid.geometry.unmasked_grid, indexes=indexes
        )

    sub_plotter.setup_subplot(number_subplots=number_subplots, subplot_index=2)

    sub_plotter.plot_mapper(
        mapper=mapper,
        image_pixel_indexes=image_pixel_indexes,
        source_pixel_indexes=source_pixel_indexes,
        include_grid=include.inversion_grid,
        include_pixelization_grid=include.inversion_pixelization_grid,
        include_border=include.inversion_border,
    )

    sub_plotter.output.subplot_to_figure()
    sub_plotter.figure.close()
