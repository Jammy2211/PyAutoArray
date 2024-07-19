import numpy as np
from typing import Union

from autoarray.plot.abstract_plotters import Plotter
from autoarray.plot.visuals.two_d import Visuals2D
from autoarray.plot.include.two_d import Include2D
from autoarray.plot.mat_plot.two_d import MatPlot2D
from autoarray.plot.auto_labels import AutoLabels
from autoarray.structures.arrays.uniform_2d import Array2D
from autoarray.inversion.pixelization.mappers.rectangular import (
    MapperRectangularNoInterp,
)

import logging

logger = logging.getLogger(__name__)


class MapperPlotter(Plotter):
    def __init__(
        self,
        mapper: MapperRectangularNoInterp,
        mat_plot_2d: MatPlot2D = MatPlot2D(),
        visuals_2d: Visuals2D = Visuals2D(),
        include_2d: Include2D = Include2D(),
    ):
        """
        Plots the attributes of `Mapper` objects using the matplotlib method `imshow()` and many other matplotlib
        functions which customize the plot's appearance.

        The `mat_plot_2d` attribute wraps matplotlib function calls to make the figure. By default, the settings
        passed to every matplotlib function called are those specified in the `config/visualize/mat_wrap/*.ini` files,
        but a user can manually input values into `MatPlot2d` to customize the figure's appearance.

        Overlaid on the figure are visuals, contained in the `Visuals2D` object. Attributes may be extracted from
        the `Mapper` and plotted via the visuals object, if the corresponding entry is `True` in the `Include2D`
        object or the `config/visualize/include.ini` file.

        Parameters
        ----------
        mapper
            The mapper the plotter plots.
        mat_plot_2d
            Contains objects which wrap the matplotlib function calls that make 2D plots.
        visuals_2d
            Contains 2D visuals that can be overlaid on 2D plots.
        include_2d
            Specifies which attributes of the `Mapper` are extracted and plotted as visuals for 2D plots.
        """
        super().__init__(
            visuals_2d=visuals_2d, include_2d=include_2d, mat_plot_2d=mat_plot_2d
        )

        self.mapper = mapper

    def get_visuals_2d_for_data(self) -> Visuals2D:
        return self.get_2d.via_mapper_for_data_from(mapper=self.mapper)

    def get_visuals_2d_for_source(self) -> Visuals2D:
        return self.get_2d.via_mapper_for_source_from(mapper=self.mapper)

    def figure_2d(
        self, interpolate_to_uniform: bool = True, solution_vector: bool = None
    ):
        """
        Plots the plotter's `Mapper` object in 2D.

        Parameters
        ----------
        interpolate_to_uniform
            By default, the mesh's reconstruction is interpolated to a uniform 2D array for plotting. If the
            reconstruction can be plotted in an alternative format (e.g. using Voronoi pixels for a Voronoi mesh)
            settings `interpolate_to_uniform=False` plots the reconstruction using this.
        solution_vector
            A vector of values which can culor the pixels of the mapper's source pixels.
        """
        self.mat_plot_2d.plot_mapper(
            mapper=self.mapper,
            visuals_2d=self.get_2d.via_mapper_for_source_from(mapper=self.mapper),
            interpolate_to_uniform=interpolate_to_uniform,
            pixel_values=solution_vector,
            auto_labels=AutoLabels(
                title="Pixelization Mesh (Image-Plane)", filename="mapper"
            ),
        )

    def subplot_image_and_mapper(
        self, image: Array2D, interpolate_to_uniform: bool = True
    ):
        """
        Make a subplot of an input image and the `Mapper`'s source-plane reconstruction.

        This function can include colored points that mark the mappings between the image pixels and their
        corresponding locations in the `Mapper` source-plane and reconstruction. This therefore visually illustrates
        the mapping process.

        Parameters
        ----------
        interpolate_to_uniform
            By default, the mesh's reconstruction is interpolated to a uniform 2D array for plotting. If the
            reconstruction can be plotted in an alternative format (e.g. using Voronoi pixels for a Voronoi mesh)
            settings `interpolate_to_uniform=False` plots the reconstruction using this.
        image
            The image which is plotted on the subplot.
        """
        self.open_subplot_figure(number_subplots=2)

        self.mat_plot_2d.plot_array(
            array=image,
            visuals_2d=self.get_visuals_2d_for_data(),
            auto_labels=AutoLabels(title="Image (Image-Plane)"),
        )

        if self.visuals_2d.pix_indexes is not None:
            indexes = self.mapper.pix_indexes_for_slim_indexes(
                pix_indexes=self.visuals_2d.pix_indexes
            )

            self.mat_plot_2d.index_scatter.scatter_grid_indexes(
                grid=self.mapper.over_sampler.over_sampled_grid,
                indexes=indexes,
            )

        self.figure_2d(interpolate_to_uniform=interpolate_to_uniform)

        self.mat_plot_2d.output.subplot_to_figure(
            auto_filename="subplot_image_and_mapper"
        )
        self.close_subplot_figure()

    def plot_source_from(
        self,
        pixel_values: np.ndarray,
        zoom_to_brightest: bool = True,
        interpolate_to_uniform: bool = False,
        auto_labels: AutoLabels = AutoLabels(),
    ):
        """
        Plot the source of the `Mapper` where the coloring is specified by an input set of values.

        Parameters
        ----------
        pixel_values
            The values of the mapper's source pixels used for coloring the figure.
        zoom_to_brightest
            For images not in the image-plane (e.g. the `plane_image`), whether to automatically zoom the plot to
            the brightest regions of the galaxies being plotted as opposed to the full extent of the grid.
        interpolate_to_uniform
            If `True`, the mapper's reconstruction is interpolated to a uniform grid before plotting, for example
            meaning that an irregular Delaunay grid can be plotted as a uniform grid.
        auto_labels
            The labels given to the figure.
        """
        try:
            self.mat_plot_2d.plot_mapper(
                mapper=self.mapper,
                visuals_2d=self.get_visuals_2d_for_source(),
                auto_labels=auto_labels,
                pixel_values=pixel_values,
                zoom_to_brightest=zoom_to_brightest,
                interpolate_to_uniform=interpolate_to_uniform,
            )
        except ValueError:
            logger.info(
                "Could not plot the source-plane via the Mapper because of a ValueError."
            )
