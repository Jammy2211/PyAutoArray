import matplotlib
from autoarray import conf

backend = conf.get_matplotlib_backend()
matplotlib.use(backend)
import matplotlib.pyplot as plt
import itertools

from autoarray.plotters import plotters
from autoarray.plots import imaging_plots


@plotters.set_labels
def image_and_mapper(
    imaging,
    mapper,
    mask=None,
    positions=None,
    image_pixels=None,
    source_pixels=None,
    include=plotters.Include(),
    plotter=plotters.Plotter(),
):



    plotter = plotter.plotter_with_new_output_filename(
        output_filename="image_and_mapper"
    )

    rows, columns, figsize_tool = plotter.get_subplot_rows_columns_figsize(
        number_subplots=2
    )

    plt.figure(figsize=figsize)
    plt.subplot(rows, columns, 1)

    imaging_plots.image(
        imaging=imaging,
        mask=mask,
        positions=positions,
        include=include,
        plotter=plotter,
    )

    point_colors = itertools.cycle(["y", "r", "k", "g", "m"])
    plotter.plot_image_pixels(
        grid=mapper.grid.geometry.unmasked_grid,
        image_pixels=image_pixels,
        point_colors=point_colors,
    )
    plotter.plot_image_plane_source_pixels(
        mapper=mapper,
        grid=mapper.grid.geometry.unmasked_grid,
        source_pixels=source_pixels,
        point_colors=point_colors,
    )

    plt.subplot(rows, columns, 2)

    mapper_grid(
        mapper=mapper,
        image_pixels=image_pixels,
        source_pixels=source_pixels,
        include=include,
        plotter=plotter,
    )

    plotter.output.to_figure(structure=None)
    plt.close()


@plotters.set_labels
def mapper_grid(
    mapper,
    image_pixels=None,
    source_pixels=None,
    include=plotters.Include(),
    plotter=plotters.Plotter(),
):

    plotter.array.plot(
        mapper=mapper,
        image_pixels=image_pixels,
        source_pixels=source_pixels,
        include_centres=include.inversion_centres,
        include_grid=include.inversion_grid,
        include_border=include.inversion_border,
    )
