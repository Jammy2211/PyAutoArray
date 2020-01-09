import matplotlib
from autoarray import conf

backend = conf.get_matplotlib_backend()
matplotlib.use(backend)
import matplotlib.pyplot as plt
import itertools

from autoarray.plotters import plotters, array_plotters, mapper_plotters
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
    array_plotter=array_plotters.ArrayPlotter(),
    mapper_plotter=mapper_plotters.MapperPlotter(),
):

    array_plotter = array_plotter.plotter_as_sub_plotter()
    mapper_plotter = mapper_plotter.plotter_as_sub_plotter()
    mapper_plotter = mapper_plotter.plotter_with_new_output_filename(
        output_filename="image_and_mapper"
    )

    rows, columns, figsize_tool = array_plotter.get_subplot_rows_columns_figsize(
        number_subplots=2
    )

    if array_plotter.figsize is None:
        figsize = figsize_tool
    else:
        figsize = array_plotter.figsize

    plt.figure(figsize=figsize)
    plt.subplot(rows, columns, 1)

    imaging_plots.image(
        imaging=imaging, mask=mask, positions=positions, include=include, array_plotter=array_plotter
    )

    point_colors = itertools.cycle(["y", "r", "k", "g", "m"])
    mapper_plotter.plot_image_pixels(
        grid=mapper.grid.geometry.unmasked_grid,
        image_pixels=image_pixels,
        point_colors=point_colors,
    )
    mapper_plotter.plot_image_plane_source_pixels(
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
        mapper_plotter=mapper_plotter,
    )

    mapper_plotter.output_subplot_array()
    plt.close()



@plotters.set_labels
def mapper_grid(
    mapper,
    image_pixels=None,
    source_pixels=None,
    include=plotters.Include(),
    mapper_plotter=mapper_plotters.MapperPlotter(),
):

    mapper_plotter.plot_mapper(
        mapper=mapper,
        image_pixels=image_pixels,
        source_pixels=source_pixels,
        include_centres=include.inversion_centres,
        include_grid=include.inversion_grid,
        include_border=include.inversion_border,
    )
