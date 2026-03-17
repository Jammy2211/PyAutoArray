import numpy as np
import logging

from autoarray.plot.abstract_plotters import AbstractPlotter
from autoarray.plot.mat_plot.two_d import MatPlot2D
from autoarray.plot.auto_labels import AutoLabels
from autoarray.plot.plots.inversion import plot_inversion_reconstruction
from autoarray.plot.plots.array import plot_array
from autoarray.structures.arrays.uniform_2d import Array2D
from autoarray.structures.plot.structure_plotters import (
    _auto_mask_edge,
    _numpy_lines,
    _numpy_grid,
    _numpy_positions,
    _output_for_mat_plot,
)

logger = logging.getLogger(__name__)


class MapperPlotter(AbstractPlotter):
    def __init__(
        self,
        mapper,
        mat_plot_2d: MatPlot2D = None,
        mesh_grid=None,
        lines=None,
        grid=None,
        positions=None,
    ):
        super().__init__(mat_plot_2d=mat_plot_2d)
        self.mapper = mapper
        self.mesh_grid = mesh_grid
        self.lines = lines
        self.grid = grid
        self.positions = positions

    def figure_2d(self, solution_vector=None):
        is_sub = self.mat_plot_2d.is_for_subplot
        ax = self.mat_plot_2d.setup_subplot() if is_sub else None
        output_path, filename, fmt = _output_for_mat_plot(self.mat_plot_2d, is_sub, "mapper")

        try:
            plot_inversion_reconstruction(
                pixel_values=solution_vector,
                mapper=self.mapper,
                ax=ax,
                title="Pixelization Mesh (Source-Plane)",
                colormap=self.mat_plot_2d.cmap.cmap,
                use_log10=self.mat_plot_2d.use_log10,
                lines=_numpy_lines(self.lines),
                grid=_numpy_grid(self.mesh_grid),
                output_path=output_path,
                output_filename=filename,
                output_format=fmt,
            )
        except Exception as exc:
            logger.info(f"Could not plot the source-plane via the Mapper: {exc}")

    def figure_2d_image(self, image):
        is_sub = self.mat_plot_2d.is_for_subplot
        ax = self.mat_plot_2d.setup_subplot() if is_sub else None
        output_path, filename, fmt = _output_for_mat_plot(self.mat_plot_2d, is_sub, "mapper_image")

        try:
            arr = image.native.array
            extent = image.geometry.extent
        except AttributeError:
            arr = np.asarray(image)
            extent = None

        plot_array(
            array=arr,
            ax=ax,
            extent=extent,
            mask=_auto_mask_edge(image) if hasattr(image, "mask") else None,
            lines=_numpy_lines(self.lines),
            title="Image (Image-Plane)",
            colormap=self.mat_plot_2d.cmap.cmap,
            use_log10=self.mat_plot_2d.use_log10,
            output_path=output_path,
            output_filename=filename,
            output_format=fmt,
        )

    def subplot_image_and_mapper(self, image: Array2D):
        self.open_subplot_figure(number_subplots=2)
        self.figure_2d_image(image=image)
        self.figure_2d()
        self.mat_plot_2d.output.subplot_to_figure(
            auto_filename="subplot_image_and_mapper"
        )
        self.close_subplot_figure()

    def plot_source_from(
        self,
        pixel_values: np.ndarray,
        zoom_to_brightest: bool = True,
        auto_labels: AutoLabels = AutoLabels(),
    ):
        is_sub = self.mat_plot_2d.is_for_subplot
        ax = self.mat_plot_2d.setup_subplot() if is_sub else None
        output_path, filename, fmt = _output_for_mat_plot(
            self.mat_plot_2d, is_sub, auto_labels.filename or "reconstruction"
        )

        try:
            plot_inversion_reconstruction(
                pixel_values=pixel_values,
                mapper=self.mapper,
                ax=ax,
                title=auto_labels.title or "Source Reconstruction",
                colormap=self.mat_plot_2d.cmap.cmap,
                use_log10=self.mat_plot_2d.use_log10,
                zoom_to_brightest=zoom_to_brightest,
                lines=_numpy_lines(self.lines),
                grid=_numpy_grid(self.mesh_grid),
                output_path=output_path,
                output_filename=filename,
                output_format=fmt,
            )
        except ValueError:
            logger.info(
                "Could not plot the source-plane via the Mapper because of a ValueError."
            )
