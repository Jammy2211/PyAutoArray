from autoarray.plot.plotters import abstract_plotters
from autoarray.plot.mat_wrap import visuals as vis
from autoarray.plot.mat_wrap import include as inc
from autoarray.plot.mat_wrap import mat_plot
from autoarray.structures import arrays, frames, grids
from autoarray.inversion import mappers
import copy


class ArrayPlotter(abstract_plotters.AbstractPlotter):
    def __init__(
        self,
        mat_plot_2d: mat_plot.MatPlot2D = mat_plot.MatPlot2D(),
        visuals_2d: vis.Visuals2D = vis.Visuals2D(),
        include_2d: inc.Include2D = inc.Include2D(),
    ):

        super().__init__(
            visuals_2d=visuals_2d, include_2d=include_2d, mat_plot_2d=mat_plot_2d
        )

    @abstract_plotters.set_labels
    def array(self, array: arrays.Array, extent_manual=None):

        self.mat_plot_2d.plot_array(
            array=array,
            visuals_2d=self.visuals_2d
            + self.include_2d.visuals_from_array(array=array),
            extent_manual=extent_manual,
        )


class FramePlotter(abstract_plotters.AbstractPlotter):
    def __init__(
        self,
        mat_plot_2d: mat_plot.MatPlot2D = mat_plot.MatPlot2D(),
        visuals_2d: vis.Visuals2D = vis.Visuals2D(),
        include_2d: inc.Include2D = inc.Include2D(),
    ):
        super().__init__(
            visuals_2d=visuals_2d, include_2d=include_2d, mat_plot_2d=mat_plot_2d
        )

    @abstract_plotters.set_labels
    def frame(self, frame: frames.Frame):

        self.mat_plot_2d._plot_frame(
            frame=frame,
            visuals_2d=self.visuals_2d
            + self.include_2d.visuals_from_frame(frame=frame),
        )


class GridPlotter(abstract_plotters.AbstractPlotter):
    def __init__(
        self,
        mat_plot_2d: mat_plot.MatPlot2D = mat_plot.MatPlot2D(),
        visuals_2d: vis.Visuals2D = vis.Visuals2D(),
        include_2d: inc.Include2D = inc.Include2D(),
    ):
        super().__init__(
            visuals_2d=visuals_2d, include_2d=include_2d, mat_plot_2d=mat_plot_2d
        )

    @abstract_plotters.set_labels
    def grid(
        self,
        grid: grids.Grid,
        color_array=None,
        axis_limits=None,
        indexes=None,
        symmetric_around_centre=True,
    ):

        self.mat_plot_2d.plot_grid(
            grid=grid,
            visuals_2d=self.visuals_2d + self.include_2d.visuals_from_grid(grid=grid),
            color_array=color_array,
            axis_limits=axis_limits,
            indexes=indexes,
            symmetric_around_centre=symmetric_around_centre,
        )


class MapperPlotter(abstract_plotters.AbstractPlotter):
    def __init__(
        self,
        mat_plot_2d: mat_plot.MatPlot2D = mat_plot.MatPlot2D(),
        visuals_2d: vis.Visuals2D = vis.Visuals2D(),
        include_2d: inc.Include2D = inc.Include2D(),
    ):
        super().__init__(
            visuals_2d=visuals_2d, include_2d=include_2d, mat_plot_2d=mat_plot_2d
        )

    @abstract_plotters.set_labels
    def mapper(
        self,
        mapper: mappers.Mapper,
        source_pixelilzation_values=None,
        full_indexes=None,
        pixelization_indexes=None,
    ):

        self.mat_plot_2d.plot_mapper(
            mapper=mapper,
            visuals_2d=self.visuals_2d
            + self.include_2d.visuals_of_source_from_mapper(mapper=mapper),
            source_pixelilzation_values=source_pixelilzation_values,
            full_indexes=full_indexes,
            pixelization_indexes=pixelization_indexes,
        )

    @abstract_plotters.set_labels
    def subplot_image_and_mapper(
        self, image, mapper, full_indexes=None, pixelization_indexes=None
    ):

        mat_plot_2d = self.mat_plot_2d.plotter_for_subplot_from(
            func=self.subplot_image_and_mapper
        )

        number_subplots = 2

        mat_plot_2d.open_subplot_figure(number_subplots=number_subplots)

        mat_plot_2d.setup_subplot(number_subplots=number_subplots, subplot_index=1)

        self.mat_plot_2d.plot_array(
            array=image,
            visuals_2d=self.visuals_2d
            + self.include_2d.visuals_from_array(array=image),
        )

        if full_indexes is not None:

            mat_plot_2d.index_scatter.scatter_grid_indexes(
                grid=mapper.source_full_grid.geometry.unmasked_grid_sub_1,
                indexes=full_indexes,
            )

        if pixelization_indexes is not None:

            indexes = mapper.full_indexes_from_pixelization_indexes(
                pixelization_indexes=pixelization_indexes
            )

            mat_plot_2d.index_scatter.scatter_grid_indexes(
                grid=mapper.source_full_grid.geometry.unmasked_grid_sub_1,
                indexes=indexes,
            )

        mat_plot_2d.setup_subplot(number_subplots=number_subplots, subplot_index=2)

        self.mapper(
            mapper=mapper,
            full_indexes=full_indexes,
            pixelization_indexes=pixelization_indexes,
        )

        mat_plot_2d.output.subplot_to_figure()
        mat_plot_2d.figure.close()


class LinePlotter:
    def __init__(
        self,
        mat_plot_1d: mat_plot.MatPlot1D = mat_plot.MatPlot1D(),
        visuals_1d: vis.Visuals1D = vis.Visuals1D(),
        include_1d: inc.Include1D = inc.Include1D(),
    ):

        self.mat_plot_1d = mat_plot_1d
        self.visuals_1d = visuals_1d
        self.include_1d = include_1d

    @abstract_plotters.set_labels
    def line(
        self,
        y,
        x,
        label=None,
        plot_axis_type="semilogy",
        vertical_lines=None,
        vertical_line_labels=None,
    ):

        self.mat_plot_1d.plot_line(
            y=y,
            x=x,
            label=label,
            plot_axis_type=plot_axis_type,
            vertical_lines=vertical_lines,
            vertical_line_labels=vertical_line_labels,
        )
