import numpy as np

from autoconf import conf

from autoarray.inversion.mappers.abstract import Mapper
from autoarray.plot.abstract_plotters import AbstractPlotter
from autoarray.plot.mat_plot.two_d import MatPlot2D
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
    _output_for_mat_plot,
)


class InversionPlotter(AbstractPlotter):
    def __init__(
        self,
        inversion: AbstractInversion,
        mat_plot_2d: MatPlot2D = None,
        mesh_grid=None,
        lines=None,
        grid=None,
        positions=None,
        residuals_symmetric_cmap: bool = True,
    ):
        super().__init__(mat_plot_2d=mat_plot_2d)
        self.inversion = inversion
        self.mesh_grid = mesh_grid
        self.lines = lines
        self.grid = grid
        self.positions = positions
        self.residuals_symmetric_cmap = residuals_symmetric_cmap

    def mapper_plotter_from(self, mapper_index: int, mesh_grid=None) -> MapperPlotter:
        return MapperPlotter(
            mapper=self.inversion.cls_list_from(cls=Mapper)[mapper_index],
            mat_plot_2d=self.mat_plot_2d,
            mesh_grid=mesh_grid if mesh_grid is not None else self.mesh_grid,
            lines=self.lines,
            grid=self.grid,
            positions=self.positions,
        )

    def _plot_array(self, array, auto_filename: str, title: str):
        is_sub = self.mat_plot_2d.is_for_subplot
        ax = self.mat_plot_2d.setup_subplot() if is_sub else None
        output_path, filename, fmt = _output_for_mat_plot(
            self.mat_plot_2d, is_sub, auto_filename
        )
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
            colormap=self.mat_plot_2d.cmap.cmap,
            use_log10=self.mat_plot_2d.use_log10,
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
                    title="Data Subtracted",
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
                title="Reconstructed Image",
            )

        if reconstruction:
            vmax_custom = False
            if "vmax" in self.mat_plot_2d.cmap.kwargs:
                if self.mat_plot_2d.cmap.kwargs["vmax"] is None:
                    reconstruction_vmax_factor = conf.instance["visualize"]["general"][
                        "inversion"
                    ]["reconstruction_vmax_factor"]
                    self.mat_plot_2d.cmap.kwargs["vmax"] = (
                        reconstruction_vmax_factor * np.max(self.inversion.reconstruction)
                    )
                    vmax_custom = True

            pixel_values = self.inversion.reconstruction_dict[mapper_plotter.mapper]
            mapper_plotter.plot_source_from(
                pixel_values=pixel_values,
                zoom_to_brightest=zoom_to_brightest,
                auto_labels=AutoLabels(
                    title="Source Reconstruction", filename="reconstruction"
                ),
            )
            if vmax_custom:
                self.mat_plot_2d.cmap.kwargs["vmax"] = None

        if reconstruction_noise_map:
            try:
                mapper_plotter.plot_source_from(
                    pixel_values=self.inversion.reconstruction_noise_map_dict[
                        mapper_plotter.mapper
                    ],
                    auto_labels=AutoLabels(
                        title="Noise Map", filename="reconstruction_noise_map"
                    ),
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
                        title="Signal To Noise Map", filename="signal_to_noise_map"
                    ),
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
                        title="Regularization weight_list",
                        filename="regularization_weights",
                    ),
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
                title="Sub Pixels Per Image Pixels",
            )

        if mesh_pixels_per_image_pixels:
            try:
                mesh_arr = mapper_plotter.mapper.mesh_pixels_per_image_pixels
                self._plot_array(
                    array=mesh_arr,
                    auto_filename="mesh_pixels_per_image_pixels",
                    title="Mesh Pixels Per Image Pixels",
                )
            except Exception:
                pass

        if image_pixels_per_mesh_pixel:
            try:
                mapper_plotter.plot_source_from(
                    pixel_values=mapper_plotter.mapper.data_weight_total_for_pix_from(),
                    auto_labels=AutoLabels(
                        title="Image Pixels Per Source Pixel",
                        filename="image_pixels_per_mesh_pixel",
                    ),
                )
            except TypeError:
                pass

    def subplot_of_mapper(
        self, mapper_index: int = 0, auto_filename: str = "subplot_inversion"
    ):
        self.open_subplot_figure(number_subplots=12)

        contour_original = self.mat_plot_2d.contour

        if self.mat_plot_2d.use_log10:
            self.mat_plot_2d.contour = False

        self.figures_2d_of_pixelization(
            pixelization_index=mapper_index, data_subtracted=True
        )
        self.figures_2d_of_pixelization(
            pixelization_index=mapper_index, reconstructed_operated_data=True
        )

        self.mat_plot_2d.use_log10 = True
        self.mat_plot_2d.contour = False
        self.figures_2d_of_pixelization(
            pixelization_index=mapper_index, reconstructed_operated_data=True
        )
        self.mat_plot_2d.use_log10 = False

        mapper = self.inversion.cls_list_from(cls=Mapper)[mapper_index]

        # Pass mesh_grid directly to this specific call instead of mutating state
        self.set_title(label="Mesh Pixel Grid Overlaid")
        self.figures_2d_of_pixelization(
            pixelization_index=mapper_index,
            reconstructed_operated_data=True,
            mesh_grid=mapper.image_plane_mesh_grid,
        )
        self.set_title(label=None)

        self.figures_2d_of_pixelization(
            pixelization_index=mapper_index, reconstruction=True
        )

        self.set_title(label="Source Reconstruction (Unzoomed)")
        self.figures_2d_of_pixelization(
            pixelization_index=mapper_index,
            reconstruction=True,
            zoom_to_brightest=False,
        )
        self.set_title(label=None)

        self.set_title(label="Noise-Map (Unzoomed)")
        self.figures_2d_of_pixelization(
            pixelization_index=mapper_index,
            reconstruction_noise_map=True,
            zoom_to_brightest=False,
        )

        self.set_title(label="Regularization Weights (Unzoomed)")
        try:
            self.figures_2d_of_pixelization(
                pixelization_index=mapper_index,
                regularization_weights=True,
                zoom_to_brightest=False,
            )
        except IndexError:
            pass
        self.set_title(label=None)

        self.figures_2d_of_pixelization(
            pixelization_index=mapper_index, sub_pixels_per_image_pixels=True
        )
        self.figures_2d_of_pixelization(
            pixelization_index=mapper_index, mesh_pixels_per_image_pixels=True
        )
        self.figures_2d_of_pixelization(
            pixelization_index=mapper_index, image_pixels_per_mesh_pixel=True
        )

        self.mat_plot_2d.output.subplot_to_figure(
            auto_filename=f"{auto_filename}_{mapper_index}"
        )
        self.mat_plot_2d.contour = contour_original
        self.close_subplot_figure()

    def subplot_mappings(
        self, pixelization_index: int = 0, auto_filename: str = "subplot_mappings"
    ):
        self.open_subplot_figure(number_subplots=4)

        self.figures_2d_of_pixelization(
            pixelization_index=pixelization_index, data_subtracted=True
        )

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

        # Pass indexes directly to the specific call
        self.figures_2d_of_pixelization(
            pixelization_index=pixelization_index, reconstructed_operated_data=True
        )
        self.figures_2d_of_pixelization(
            pixelization_index=pixelization_index, reconstruction=True
        )

        self.set_title(label="Source Reconstruction (Unzoomed)")
        self.figures_2d_of_pixelization(
            pixelization_index=pixelization_index,
            reconstruction=True,
            zoom_to_brightest=False,
        )
        self.set_title(label=None)

        self.mat_plot_2d.output.subplot_to_figure(
            auto_filename=f"{auto_filename}_{pixelization_index}"
        )
        self.close_subplot_figure()
