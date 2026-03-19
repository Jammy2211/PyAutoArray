import numpy as np

import matplotlib.pyplot as plt

from autoconf import conf

from autoarray.inversion.mappers.abstract import Mapper
from autoarray.plot.abstract_plotters import AbstractPlotter
from autoarray.plot.wrap.base.output import Output
from autoarray.plot.wrap.base.cmap import Cmap
from autoarray.plot.auto_labels import AutoLabels
from autoarray.plot.plots.array import plot_array
from autoarray.structures.arrays.uniform_2d import Array2D
from autoarray.inversion.inversion.abstract import AbstractInversion
from autoarray.inversion.plot.mapper_plotters import MapperPlotter
from autoarray.structures.plot.structure_plotters import (
    _auto_mask_edge,
    _numpy_lines,
    _numpy_grid,
    _numpy_positions,
    _output_for_plotter,
)


class InversionPlotter(AbstractPlotter):
    def __init__(
        self,
        inversion: AbstractInversion,
        output: Output = None,
        cmap: Cmap = None,
        use_log10: bool = False,
        mesh_grid=None,
        lines=None,
        grid=None,
        positions=None,
        residuals_symmetric_cmap: bool = True,
    ):
        super().__init__(output=output, cmap=cmap, use_log10=use_log10)
        self.inversion = inversion
        self.mesh_grid = mesh_grid
        self.lines = lines
        self.grid = grid
        self.positions = positions
        self.residuals_symmetric_cmap = residuals_symmetric_cmap

    def mapper_plotter_from(self, mapper_index: int, mesh_grid=None) -> MapperPlotter:
        return MapperPlotter(
            mapper=self.inversion.cls_list_from(cls=Mapper)[mapper_index],
            output=self.output,
            cmap=self.cmap,
            use_log10=self.use_log10,
            mesh_grid=mesh_grid if mesh_grid is not None else self.mesh_grid,
            lines=self.lines,
            grid=self.grid,
            positions=self.positions,
        )

    def _plot_array(self, array, auto_filename: str, title: str, ax=None):
        if ax is None:
            output_path, filename, fmt = _output_for_plotter(self.output, auto_filename)
        else:
            output_path, filename, fmt = None, auto_filename, "png"

        try:
            arr = array.native.array
            extent = array.geometry.extent
            mask_overlay = _auto_mask_edge(array)
        except AttributeError:
            arr = np.asarray(array)
            extent = None
            mask_overlay = None

        plot_array(
            array=arr,
            ax=ax,
            extent=extent,
            mask=mask_overlay,
            grid=_numpy_grid(self.grid),
            positions=_numpy_positions(self.positions),
            lines=_numpy_lines(self.lines),
            title=title,
            colormap=self.cmap.cmap,
            use_log10=self.use_log10,
            output_path=output_path,
            output_filename=filename,
            output_format=fmt,
        )

    def figures_2d(self, reconstructed_operated_data: bool = False):
        if reconstructed_operated_data:
            try:
                self._plot_array(
                    array=self.inversion.mapped_reconstructed_operated_data,
                    auto_filename="reconstructed_operated_data",
                    title="Reconstructed Image",
                )
            except AttributeError:
                self._plot_array(
                    array=self.inversion.mapped_reconstructed_data,
                    auto_filename="reconstructed_data",
                    title="Reconstructed Image",
                )

    def figures_2d_of_pixelization(
        self,
        pixelization_index: int = 0,
        data_subtracted: bool = False,
        reconstructed_operated_data: bool = False,
        reconstruction: bool = False,
        reconstruction_noise_map: bool = False,
        signal_to_noise_map: bool = False,
        regularization_weights: bool = False,
        sub_pixels_per_image_pixels: bool = False,
        mesh_pixels_per_image_pixels: bool = False,
        image_pixels_per_mesh_pixel: bool = False,
        magnification_per_mesh_pixel: bool = False,
        zoom_to_brightest: bool = True,
        mesh_grid=None,
        ax=None,
        title_override=None,
    ):
        if not self.inversion.has(cls=Mapper):
            return

        mapper_plotter = self.mapper_plotter_from(
            mapper_index=pixelization_index, mesh_grid=mesh_grid
        )

        if data_subtracted:
            try:
                array = self.inversion.data_subtracted_dict[mapper_plotter.mapper]
                self._plot_array(
                    array=array,
                    auto_filename="data_subtracted",
                    title=title_override or "Data Subtracted",
                    ax=ax,
                )
            except AttributeError:
                pass

        if reconstructed_operated_data:
            array = self.inversion.mapped_reconstructed_operated_data_dict[
                mapper_plotter.mapper
            ]
            from autoarray.structures.visibilities import Visibilities
            if isinstance(array, Visibilities):
                array = self.inversion.mapped_reconstructed_data_dict[mapper_plotter.mapper]
            self._plot_array(
                array=array,
                auto_filename="reconstructed_operated_data",
                title=title_override or "Reconstructed Image",
                ax=ax,
            )

        if reconstruction:
            vmax_custom = False
            if "vmax" in self.cmap.kwargs:
                if self.cmap.kwargs["vmax"] is None:
                    reconstruction_vmax_factor = conf.instance["visualize"]["general"][
                        "inversion"
                    ]["reconstruction_vmax_factor"]
                    self.cmap.kwargs["vmax"] = (
                        reconstruction_vmax_factor * np.max(self.inversion.reconstruction)
                    )
                    vmax_custom = True

            pixel_values = self.inversion.reconstruction_dict[mapper_plotter.mapper]
            mapper_plotter.plot_source_from(
                pixel_values=pixel_values,
                zoom_to_brightest=zoom_to_brightest,
                auto_labels=AutoLabels(
                    title=title_override or "Source Reconstruction",
                    filename="reconstruction",
                ),
                ax=ax,
            )
            if vmax_custom:
                self.cmap.kwargs["vmax"] = None

        if reconstruction_noise_map:
            try:
                mapper_plotter.plot_source_from(
                    pixel_values=self.inversion.reconstruction_noise_map_dict[
                        mapper_plotter.mapper
                    ],
                    auto_labels=AutoLabels(
                        title=title_override or "Noise Map",
                        filename="reconstruction_noise_map",
                    ),
                    ax=ax,
                )
            except TypeError:
                pass

        if signal_to_noise_map:
            try:
                signal_to_noise_values = (
                    self.inversion.reconstruction_dict[mapper_plotter.mapper]
                    / self.inversion.reconstruction_noise_map_dict[mapper_plotter.mapper]
                )
                mapper_plotter.plot_source_from(
                    pixel_values=signal_to_noise_values,
                    auto_labels=AutoLabels(
                        title=title_override or "Signal To Noise Map",
                        filename="signal_to_noise_map",
                    ),
                    ax=ax,
                )
            except TypeError:
                pass

        if regularization_weights:
            try:
                mapper_plotter.plot_source_from(
                    pixel_values=self.inversion.regularization_weights_mapper_dict[
                        mapper_plotter.mapper
                    ],
                    auto_labels=AutoLabels(
                        title=title_override or "Regularization weight_list",
                        filename="regularization_weights",
                    ),
                    ax=ax,
                )
            except (IndexError, ValueError):
                pass

        if sub_pixels_per_image_pixels:
            sub_size = Array2D(
                values=mapper_plotter.mapper.over_sampler.sub_size,
                mask=self.inversion.dataset.mask,
            )
            self._plot_array(
                array=sub_size,
                auto_filename="sub_pixels_per_image_pixels",
                title=title_override or "Sub Pixels Per Image Pixels",
                ax=ax,
            )

        if mesh_pixels_per_image_pixels:
            try:
                mesh_arr = mapper_plotter.mapper.mesh_pixels_per_image_pixels
                self._plot_array(
                    array=mesh_arr,
                    auto_filename="mesh_pixels_per_image_pixels",
                    title=title_override or "Mesh Pixels Per Image Pixels",
                    ax=ax,
                )
            except Exception:
                pass

        if image_pixels_per_mesh_pixel:
            try:
                mapper_plotter.plot_source_from(
                    pixel_values=mapper_plotter.mapper.data_weight_total_for_pix_from(),
                    auto_labels=AutoLabels(
                        title=title_override or "Image Pixels Per Source Pixel",
                        filename="image_pixels_per_mesh_pixel",
                    ),
                    ax=ax,
                )
            except TypeError:
                pass

    def subplot_of_mapper(
        self, mapper_index: int = 0, auto_filename: str = "subplot_inversion"
    ):
        mapper = self.inversion.cls_list_from(cls=Mapper)[mapper_index]

        fig, axes = plt.subplots(3, 4, figsize=(28, 21))
        axes = axes.flatten()

        self.figures_2d_of_pixelization(
            pixelization_index=mapper_index, data_subtracted=True, ax=axes[0]
        )
        self.figures_2d_of_pixelization(
            pixelization_index=mapper_index, reconstructed_operated_data=True, ax=axes[1]
        )

        use_log10_orig = self.use_log10
        self.use_log10 = True
        self.figures_2d_of_pixelization(
            pixelization_index=mapper_index, reconstructed_operated_data=True, ax=axes[2]
        )
        self.use_log10 = use_log10_orig

        self.figures_2d_of_pixelization(
            pixelization_index=mapper_index,
            reconstructed_operated_data=True,
            mesh_grid=mapper.image_plane_mesh_grid,
            ax=axes[3],
            title_override="Mesh Pixel Grid Overlaid",
        )
        self.figures_2d_of_pixelization(
            pixelization_index=mapper_index, reconstruction=True, ax=axes[4]
        )
        self.figures_2d_of_pixelization(
            pixelization_index=mapper_index,
            reconstruction=True,
            zoom_to_brightest=False,
            ax=axes[5],
            title_override="Source Reconstruction (Unzoomed)",
        )
        self.figures_2d_of_pixelization(
            pixelization_index=mapper_index,
            reconstruction_noise_map=True,
            zoom_to_brightest=False,
            ax=axes[6],
            title_override="Noise-Map (Unzoomed)",
        )
        try:
            self.figures_2d_of_pixelization(
                pixelization_index=mapper_index,
                regularization_weights=True,
                zoom_to_brightest=False,
                ax=axes[7],
                title_override="Regularization Weights (Unzoomed)",
            )
        except IndexError:
            pass

        self.figures_2d_of_pixelization(
            pixelization_index=mapper_index, sub_pixels_per_image_pixels=True, ax=axes[8]
        )
        self.figures_2d_of_pixelization(
            pixelization_index=mapper_index,
            mesh_pixels_per_image_pixels=True,
            ax=axes[9],
        )
        self.figures_2d_of_pixelization(
            pixelization_index=mapper_index,
            image_pixels_per_mesh_pixel=True,
            ax=axes[10],
        )

        plt.tight_layout()
        self.output.subplot_to_figure(
            auto_filename=f"{auto_filename}_{mapper_index}"
        )
        plt.close()

    def subplot_mappings(
        self, pixelization_index: int = 0, auto_filename: str = "subplot_mappings"
    ):
        total_pixels = conf.instance["visualize"]["general"]["inversion"][
            "total_mappings_pixels"
        ]

        mapper = self.inversion.cls_list_from(cls=Mapper)[pixelization_index]

        pix_indexes = self.inversion.max_pixel_list_from(
            total_pixels=total_pixels,
            filter_neighbors=True,
            mapper_index=pixelization_index,
        )

        indexes = mapper.slim_indexes_for_pix_indexes(pix_indexes=pix_indexes)

        fig, axes = plt.subplots(2, 2, figsize=(14, 14))
        axes = axes.flatten()

        self.figures_2d_of_pixelization(
            pixelization_index=pixelization_index, data_subtracted=True, ax=axes[0]
        )
        self.figures_2d_of_pixelization(
            pixelization_index=pixelization_index,
            reconstructed_operated_data=True,
            ax=axes[1],
        )
        self.figures_2d_of_pixelization(
            pixelization_index=pixelization_index, reconstruction=True, ax=axes[2]
        )
        self.figures_2d_of_pixelization(
            pixelization_index=pixelization_index,
            reconstruction=True,
            zoom_to_brightest=False,
            ax=axes[3],
            title_override="Source Reconstruction (Unzoomed)",
        )

        plt.tight_layout()
        self.output.subplot_to_figure(
            auto_filename=f"{auto_filename}_{pixelization_index}"
        )
        plt.close()
