from typing import Union

from autoarray.plot.abstract_plotters import AbstractPlotter
from autoarray.plot.mat_wrap.visuals import Visuals2D
from autoarray.plot.mat_wrap.include import Include2D
from autoarray.plot.mat_wrap.mat_plot import MatPlot2D
from autoarray.plot.mat_wrap.mat_plot import AutoLabels
from autoarray.structures.arrays.two_d.array_2d import Array2D
from autoarray.structures.grids.two_d.grid_2d_irregular import Grid2DIrregular
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

    @property
    def visuals_data_with_include_2d(self) -> Visuals2D:
        """
        Extracts from a `Mapper` attributes that can be plotted for figures in its data-plane (e.g. the reconstructed
        data) and return them in a `Visuals` object.

        Only attributes with `True` entries in the `Include` object are extracted for plotting.

        From a `Mapper` the following attributes can be extracted for plotting in the data-plane:

        - origin: the (y,x) origin of the `Array2D`'s coordinate system in the data plane.
        - mask : the `Mask` defined in the data-plane containing the data that is used by the `Mapper`.
        - mapper_data_pixelization_grid: the `Mapper`'s pixelization grid in the data-plane.
        - mapper_border_grid: the border of the `Mapper`'s full grid in the data-plane.

        Parameters
        ----------
        mapper : Mapper
            The mapper whose data-plane attributes are extracted for plotting.

        Returns
        -------
        Visuals2D
            The collection of attributes that can be plotted by a `Plotter2D` object.
        """
        return self.visuals_2d + self.visuals_2d.__class__(
            origin=self.extract_2d(
                "origin",
                Grid2DIrregular(grid=[self.mapper.source_grid_slim.mask.origin]),
            ),
            mask=self.extract_2d("mask", self.mapper.source_grid_slim.mask),
            border=self.extract_2d(
                "border", self.mapper.source_grid_slim.mask.border_grid_sub_1.binned
            ),
            pixelization_grid=self.extract_2d(
                "pixelization_grid",
                self.mapper.data_pixelization_grid,
                "mapper_data_pixelization_grid",
            ),
        )

    @property
    def visuals_source_with_include_2d(self) -> Visuals2D:
        """
        Extracts from a `Mapper` attributes that can be plotted for figures in its source-plane (e.g. the reconstruction
        and return them in a `Visuals` object.

        Only attributes with `True` entries in the `Include` object are extracted for plotting.

        From a `Mapper` the following attributes can be extracted for plotting in the source-plane:

        - origin: the (y,x) origin of the coordinate system in the source plane.
        - mapper_source_pixelization_grid: the `Mapper`'s pixelization grid in the source-plane.
        - mapper_source_grid_slim: the `Mapper`'s full grid in the source-plane.
        - mapper_border_grid: the border of the `Mapper`'s full grid in the data-plane.

        Parameters
        ----------
        mapper : Mapper
            The mapper whose source-plane attributes are extracted for plotting.

        Returns
        -------
        Visuals2D
            The collection of attributes that can be plotted by a `Plotter2D` object.
        """

        return self.visuals_2d + self.visuals_2d.__class__(
            origin=self.extract_2d(
                "origin",
                Grid2DIrregular(grid=[self.mapper.source_pixelization_grid.origin]),
            ),
            grid=self.extract_2d(
                "grid", self.mapper.source_grid_slim, "mapper_source_grid_slim"
            ),
            border=self.extract_2d(
                "border", self.mapper.source_grid_slim.sub_border_grid
            ),
            pixelization_grid=self.extract_2d(
                "pixelization_grid",
                self.mapper.source_pixelization_grid,
                "mapper_source_pixelization_grid",
            ),
        )

    def figure_2d(self, solution_vector: bool = None):

        self.mat_plot_2d.plot_mapper(
            mapper=self.mapper,
            visuals_2d=self.visuals_source_with_include_2d,
            source_pixelilzation_values=solution_vector,
            auto_labels=AutoLabels(title="Mapper", filename="mapper"),
        )

    def subplot_image_and_mapper(self, image: Array2D):

        self.open_subplot_figure(number_subplots=2)

        self.mat_plot_2d.plot_array(
            array=image,
            visuals_2d=self.visuals_data_with_include_2d,
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
            visuals_2d=self.visuals_source_with_include_2d,
            auto_labels=auto_labels,
            source_pixelilzation_values=self.mapper.reconstruction_from(
                source_pixelization_values
            ),
        )
