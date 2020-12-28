from autoarray.plot.plotter import plotter, include as inc


@inc.set_include
@plotter.set_plotter_for_figure
@plotter.set_labels
def subplot_image_and_mapper(
    image,
    mapper,
    image_positions=None,
    source_positions=None,
    image_pixel_indexes=None,
    source_pixel_indexes=None,
    include=None,
    plotter=None,
):

    number_subplots = 2

    plotter.open_subplot_figure(number_subplots=number_subplots)

    plotter.setup_subplot(number_subplots=number_subplots, subplot_index=1)

    plotter._plot_array(
        array=image,
        mask=include.mask_from_grid(grid=mapper.grid),
        positions=image_positions,
        include_origin=include.origin,
    )

    if image_pixel_indexes is not None:

        plotter.index_scatter.scatter_grid_indexes(
            grid=mapper.grid.geometry.unmasked_grid_sub_1, indexes=image_pixel_indexes
        )

    if source_pixel_indexes is not None:

        indexes = mapper.image_pixel_indexes_from_source_pixel_indexes(
            source_pixel_indexes=source_pixel_indexes
        )

        plotter.index_scatter.scatter_grid_indexes(
            grid=mapper.grid.geometry.unmasked_grid_sub_1, indexes=indexes
        )

    plotter.setup_subplot(number_subplots=number_subplots, subplot_index=2)

    plotter._plot_mapper(
        mapper=mapper,
        positions=source_positions,
        image_pixel_indexes=image_pixel_indexes,
        source_pixel_indexes=source_pixel_indexes,
        include_origin=include.origin,
        include_grid=include.inversion_grid,
        include_pixelization_grid=include.inversion_pixelization_grid,
        include_border=include.inversion_border,
    )

    plotter.output.subplot_to_figure()
    plotter.figure.close()
