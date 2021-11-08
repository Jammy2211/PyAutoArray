from typing import Union

from autoarray.plot.abstract_plotters import AbstractPlotter
from autoarray.plot.mat_wrap.visuals import Visuals2D
from autoarray.plot.mat_wrap.include import Include2D
from autoarray.plot.mat_wrap.mat_plot import MatPlot2D
from autoarray.plot.mat_wrap.mat_plot import AutoLabels
from autoarray.structures.arrays.two_d.array_2d import Array2D
from autoarray.inversion.mappers.rectangular import MapperRectangular
from autoarray.inversion.mappers.voronoi import MapperVoronoi


class MapperPlotter(AbstractPlotter):
    def __init__(
        self,
        mapper: Union[MapperRectangular, MapperVoronoi],
        mat_plot_2d: MatPlot2D = MatPlot2D(),
        visuals_2d: Visuals2D = Visuals2D(),
        include_2d: Include2D = Include2D(),
    ):
        super().__init__(
            visuals_2d=visuals_2d, include_2d=include_2d, mat_plot_2d=mat_plot_2d
        )

        self.mapper = mapper

    def figure_2d(self, solution_vector: bool = None):

        self.mat_plot_2d.plot_mapper(
            mapper=self.mapper,
            visuals_2d=self.extractor_2d.via_mapper_for_source_from(mapper=self.mapper),
            source_pixelilzation_values=solution_vector,
            auto_labels=AutoLabels(title="Mapper", filename="mapper"),
        )

    def subplot_image_and_mapper(self, image: Array2D):

        self.open_subplot_figure(number_subplots=2)

        self.mat_plot_2d.plot_array(
            array=image,
            visuals_2d=self.extractor_2d.via_mapper_for_data_from(mapper=self.mapper),
            auto_labels=AutoLabels(title="Image"),
        )

        if self.visuals_2d.pixelization_indexes is not None:

            indexes = self.mapper.pixelization_indexes_for_slim_indexes(
                pixelization_indexes=self.visuals_2d.pixelization_indexes
            )

            self.mat_plot_2d.index_scatter.scatter_grid_indexes(
                grid=self.mapper.source_grid_slim.mask.masked_grid, indexes=indexes
            )

        self.figure_2d()

        self.mat_plot_2d.output.subplot_to_figure(
            auto_filename="subplot_image_and_mapper"
        )
        self.close_subplot_figure()

    def plot_source_from(self, source_pixelization_values, auto_labels):

        self.mat_plot_2d.plot_mapper(
            mapper=self.mapper,
            visuals_2d=self.extractor_2d.via_mapper_for_source_from(mapper=self.mapper),
            auto_labels=auto_labels,
            source_pixelilzation_values=self.mapper.reconstruction_from(
                source_pixelization_values
            ),
        )
