import numpy as np
import logging

import matplotlib.pyplot as plt

from autoarray.plot.abstract_plotters import AbstractPlotter
from autoarray.plot.wrap.base.output import Output
from autoarray.plot.wrap.base.cmap import Cmap
from autoarray.plot.auto_labels import AutoLabels
from autoarray.plot.plots.inversion import plot_inversion_reconstruction
from autoarray.plot.plots.array import plot_array
from autoarray.structures.arrays.uniform_2d import Array2D
from autoarray.structures.plot.structure_plotters import (
    _auto_mask_edge,
    _numpy_lines,
    _numpy_grid,
    _numpy_positions,
    _output_for_plotter,
)

logger = logging.getLogger(__name__)


class MapperPlotter(AbstractPlotter):
    def __init__(
        self,
        mapper,
        output: Output = None,
        cmap: Cmap = None,
        use_log10: bool = False,
        mesh_grid=None,
        lines=None,
        grid=None,
        positions=None,
    ):
        super().__init__(output=output, cmap=cmap, use_log10=use_log10)
        self.mapper = mapper
        self.mesh_grid = mesh_grid
        self.lines = lines
        self.grid = grid
        self.positions = positions

    def figure_2d(self, solution_vector=None, ax=None):
        if ax is None:
            output_path, filename, fmt = _output_for_plotter(self.output, "mapper")
        else:
            output_path, filename, fmt = None, "mapper", "png"

        try:
            plot_inversion_reconstruction(
                pixel_values=solution_vector,
                mapper=self.mapper,
                ax=ax,
                title="Pixelization Mesh (Source-Plane)",
                colormap=self.cmap.cmap,
                use_log10=self.use_log10,
                lines=_numpy_lines(self.lines),
                grid=_numpy_grid(self.mesh_grid),
                output_path=output_path,
                output_filename=filename,
                output_format=fmt,
            )
        except Exception as exc:
            logger.info(f"Could not plot the source-plane via the Mapper: {exc}")

    def figure_2d_image(self, image, ax=None):
        if ax is None:
            output_path, filename, fmt = _output_for_plotter(self.output, "mapper_image")
        else:
            output_path, filename, fmt = None, "mapper_image", "png"

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
            colormap=self.cmap.cmap,
            use_log10=self.use_log10,
            output_path=output_path,
            output_filename=filename,
            output_format=fmt,
        )

    def subplot_image_and_mapper(self, image: Array2D):
        fig, axes = plt.subplots(1, 2, figsize=(14, 7))

        self.figure_2d_image(image=image, ax=axes[0])
        self.figure_2d(ax=axes[1])

        plt.tight_layout()
        self.output.subplot_to_figure(auto_filename="subplot_image_and_mapper")
        plt.close()

    def plot_source_from(
        self,
        pixel_values: np.ndarray,
        zoom_to_brightest: bool = True,
        auto_labels: AutoLabels = AutoLabels(),
        ax=None,
    ):
        if ax is None:
            output_path, filename, fmt = _output_for_plotter(
                self.output, auto_labels.filename or "reconstruction"
            )
        else:
            output_path, filename, fmt = None, auto_labels.filename or "reconstruction", "png"

        try:
            plot_inversion_reconstruction(
                pixel_values=pixel_values,
                mapper=self.mapper,
                ax=ax,
                title=auto_labels.title or "Source Reconstruction",
                colormap=self.cmap.cmap,
                use_log10=self.use_log10,
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
