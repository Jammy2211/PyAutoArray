from autoarray.plot.mat_wrap import mat_decorators


@mat_decorators.set_plot_defaults_2d
@mat_decorators.set_labels
def subplot_image_and_mapper(
    image,
    mapper,
    image_positions=None,
    source_positions=None,
    image_pixel_indexes=None,
    source_pixel_indexes=None,
    include_2d=None,
    plotter_2d=None,
):

    number_subplots = 2

    plotter_2d.open_subplot_figure(number_subplots=number_subplots)

    plotter_2d.setup_subplot(number_subplots=number_subplots, subplot_index=1)

    plotter_2d._plot_array(
        array=image,
        mask=include_2d.mask_from_grid(grid=mapper.grid),
        positions=image_positions,
        include_origin=include_2d.origin,
    )

    if image_pixel_indexes is not None:

        plotter_2d.index_scatter.scatter_grid_indexes(
            grid=mapper.grid.geometry.unmasked_grid_sub_1, indexes=image_pixel_indexes
        )

    if source_pixel_indexes is not None:

        indexes = mapper.image_pixel_indexes_from_source_pixel_indexes(
            source_pixel_indexes=source_pixel_indexes
        )

        plotter_2d.index_scatter.scatter_grid_indexes(
            grid=mapper.grid.geometry.unmasked_grid_sub_1, indexes=indexes
        )

    plotter_2d.setup_subplot(number_subplots=number_subplots, subplot_index=2)

    plotter_2d._plot_mapper(
        mapper=mapper,
        positions=source_positions,
        image_pixel_indexes=image_pixel_indexes,
        source_pixel_indexes=source_pixel_indexes,
        include_origin=include_2d.origin,
        include_grid=include_2d.inversion_grid,
        include_pixelization_grid=include_2d.inversion_pixelization_grid,
        include_border=include_2d.inversion_border,
    )

    plotter_2d.output.subplot_to_figure()
    plotter_2d.figure.close()
