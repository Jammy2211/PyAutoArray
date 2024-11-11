import numpy as np

from autoconf import conf

from autoarray.inversion.pixelization.mappers.abstract import AbstractMapper
from autoarray.plot.abstract_plotters import Plotter
from autoarray.plot.visuals.two_d import Visuals2D
from autoarray.plot.include.two_d import Include2D
from autoarray.plot.mat_plot.two_d import MatPlot2D
from autoarray.plot.auto_labels import AutoLabels
from autoarray.structures.arrays.uniform_2d import Array2D
from autoarray.inversion.inversion.abstract import AbstractInversion
from autoarray.inversion.inversion.mapper_valued import MapperValued
from autoarray.inversion.plot.mapper_plotters import MapperPlotter


class InversionPlotter(Plotter):
    def __init__(
        self,
        inversion: AbstractInversion,
        mat_plot_2d: MatPlot2D = MatPlot2D(),
        visuals_2d: Visuals2D = Visuals2D(),
        include_2d: Include2D = Include2D(),
        residuals_symmetric_cmap: bool = True,
    ):
        """
        Plots the attributes of `Inversion` objects using the matplotlib method `imshow()` and many other matplotlib
        functions which customize the plot's appearance.

        The `mat_plot_2d` attribute wraps matplotlib function calls to make the figure. By default, the settings
        passed to every matplotlib function called are those specified in the `config/visualize/mat_wrap/*.ini` files,
        but a user can manually input values into `MatPlot2d` to customize the figure's appearance.

        Overlaid on the figure are visuals, contained in the `Visuals2D` object. Attributes may be extracted from
        the `Inversion` and plotted via the visuals object, if the corresponding entry is `True` in the `Include2D`
        object or the `config/visualize/include.ini` file.

        Parameters
        ----------
        inversion
            The inversion the plotter plots.
        mat_plot_2d
            Contains objects which wrap the matplotlib function calls that make 2D plots.
        visuals_2d
            Contains 2D visuals that can be overlaid on 2D plots.
        include_2d
            Specifies which attributes of the `Inversion` are extracted and plotted as visuals for 2D plots.

        """
        super().__init__(
            mat_plot_2d=mat_plot_2d, include_2d=include_2d, visuals_2d=visuals_2d
        )

        self.inversion = inversion
        self.residuals_symmetric_cmap = residuals_symmetric_cmap

    def get_visuals_2d_for_data(self) -> Visuals2D:
        try:
            mapper = self.inversion.cls_list_from(cls=AbstractMapper)[0]

            visuals = self.get_2d.via_mapper_for_data_from(mapper=mapper)

            if self.visuals_2d.pix_indexes is not None:
                indexes = mapper.pix_indexes_for_slim_indexes(
                    pix_indexes=self.visuals_2d.pix_indexes
                )

                visuals.indexes = indexes

            return visuals

        except (AttributeError, IndexError):
            return self.visuals_2d

    def mapper_plotter_from(self, mapper_index: int) -> MapperPlotter:
        """
        Returns a `MapperPlotter` corresponding to the `Mapper` in the `Inversion`'s `linear_obj_list` given an input
        `mapper_index`.

        Parameters
        ----------
        mapper_index
            The index of the mapper in the inversion which is used to create the `MapperPlotter`.

        Returns
        -------
        MapperPlotter
            An object that plots mappers which is used for plotting attributes of the inversion.
        """
        return MapperPlotter(
            mapper=self.inversion.cls_list_from(cls=AbstractMapper)[mapper_index],
            mat_plot_2d=self.mat_plot_2d,
            visuals_2d=self.visuals_2d,
            include_2d=self.include_2d,
        )

    def figures_2d(self, reconstructed_image: bool = False):
        """
        Plots the individual attributes of the plotter's `Inversion` object in 2D.

        The API is such that every plottable attribute of the `Inversion` object is an input parameter of type bool of
        the function, which if switched to `True` means that it is plotted.

        Parameters
        ----------
        reconstructed_image
            Whether to make a 2D plot (via `imshow`) of the reconstructed image data.
        """
        if reconstructed_image:
            self.mat_plot_2d.plot_array(
                array=self.inversion.mapped_reconstructed_image,
                visuals_2d=self.get_visuals_2d_for_data(),
                auto_labels=AutoLabels(
                    title="Reconstructed Image", filename="reconstructed_image"
                ),
            )

    def figures_2d_of_pixelization(
        self,
        pixelization_index: int = 0,
        data_subtracted: bool = False,
        reconstructed_image: bool = False,
        reconstruction: bool = False,
        errors: bool = False,
        signal_to_noise_map: bool = False,
        regularization_weights: bool = False,
        sub_pixels_per_image_pixels: bool = False,
        mesh_pixels_per_image_pixels: bool = False,
        image_pixels_per_mesh_pixel: bool = False,
        magnification_per_mesh_pixel: bool = False,
        zoom_to_brightest: bool = True,
        interpolate_to_uniform: bool = False,
    ):
        """
        Plots the individual attributes of a specific `Mapper` of the plotter's `Inversion` object in 2D.

        The API is such that every plottable attribute of the `Mapper` and `Inversion` object is an input parameter of
        type bool of the function, which if switched to `True` means that it is plotted.

        Parameters
        ----------
        pixelization_index
            The index of the `Mapper` in the `Inversion`'s `linear_obj_list` that is plotted.
        reconstructed_image
            Whether to make a 2D plot (via `imshow`) of the mapper's reconstructed image data.
        reconstruction
            Whether to make a 2D plot (via `imshow` or `fill`) of the mapper's source-plane reconstruction.
        errors
            Whether to make a 2D plot (via `imshow` or `fill`) of the mapper's source-plane errors.
        signal_to_noise_map
            Whether to make a 2D plot (via `imshow` or `fill`) of the mapper's source-plane signal-to-noise-map.
        sub_pixels_per_image_pixels
            Whether to make a 2D plot (via `imshow`) of the number of sub pixels per image pixels in the 2D
            data's mask.
        mesh_pixels_per_image_pixels
            Whether to make a 2D plot (via `imshow`) of the number of image-mesh pixels per image pixels in the 2D
            data's mask (only valid for pixelizations which use an `image_mesh`, e.g. Hilbert, KMeans).
        image_pixels_per_mesh_pixel
            Whether to make a 2D plot (via `imshow`) of the number of image pixels per source plane pixel, therefore
            indicating how many image pixels map to each source pixel.
        magnification_per_mesh_pixel
            Whether to make a 2D plot (via `imshow`) of the magnification of each mesh pixel, which is the area
            ratio of the image pixel to source pixel.
        zoom_to_brightest
            For images not in the image-plane (e.g. the `plane_image`), whether to automatically zoom the plot to
            the brightest regions of the galaxies being plotted as opposed to the full extent of the grid.
        interpolate_to_uniform
            If `True`, the mapper's reconstruction is interpolated to a uniform grid before plotting, for example
            meaning that an irregular Delaunay grid can be plotted as a uniform grid.
        """

        if not self.inversion.has(cls=AbstractMapper):
            return

        mapper_plotter = self.mapper_plotter_from(mapper_index=pixelization_index)

        if data_subtracted:
            # Attribute error is cause this raises an error for interferometer inversion, because the data is
            # visibilities not an image. Update this to be handled better in future.

            try:
                array = self.inversion.data_subtracted_dict[mapper_plotter.mapper]

                self.mat_plot_2d.plot_array(
                    array=array,
                    visuals_2d=self.get_visuals_2d_for_data(),
                    grid_indexes=mapper_plotter.mapper.over_sampler.over_sampled_grid,
                    auto_labels=AutoLabels(
                        title="Data Subtracted", filename="data_subtracted"
                    ),
                )
            except AttributeError:
                pass

        if reconstructed_image:
            array = self.inversion.mapped_reconstructed_image_dict[
                mapper_plotter.mapper
            ]

            self.mat_plot_2d.plot_array(
                array=array,
                visuals_2d=self.get_visuals_2d_for_data(),
                grid_indexes=mapper_plotter.mapper.over_sampler.over_sampled_grid,
                auto_labels=AutoLabels(
                    title="Reconstructed Image", filename="reconstructed_image"
                ),
            )

        if reconstruction:
            vmax_custom = False

            if "vmax" in self.mat_plot_2d.cmap.kwargs:
                if self.mat_plot_2d.cmap.kwargs["vmax"] is None:
                    reconstruction_vmax_factor = conf.instance["visualize"]["general"][
                        "inversion"
                    ]["reconstruction_vmax_factor"]

                    self.mat_plot_2d.cmap.kwargs["vmax"] = (
                        reconstruction_vmax_factor
                        * np.max(self.inversion.reconstruction)
                    )
                    vmax_custom = True

            pixel_values = self.inversion.reconstruction_dict[mapper_plotter.mapper]

            mapper_plotter.plot_source_from(
                pixel_values=pixel_values,
                zoom_to_brightest=zoom_to_brightest,
                interpolate_to_uniform=interpolate_to_uniform,
                auto_labels=AutoLabels(
                    title="Source Reconstruction", filename="reconstruction"
                ),
            )

            if vmax_custom:
                self.mat_plot_2d.cmap.kwargs["vmax"] = None

        if errors:
            try:
                mapper_plotter.plot_source_from(
                    pixel_values=self.inversion.errors_dict[mapper_plotter.mapper],
                    auto_labels=AutoLabels(title="Errors", filename="errors"),
                )

            except TypeError:
                pass

        if signal_to_noise_map:
            try:
                signal_to_noise_values = (
                    self.inversion.reconstruction_dict[mapper_plotter.mapper]
                    / self.inversion.errors_dict[mapper_plotter.mapper]
                )

                mapper_plotter.plot_source_from(
                    pixel_values=signal_to_noise_values,
                    auto_labels=AutoLabels(
                        title="Signal To Noise Map", filename="signal_to_noise_map"
                    ),
                )

            except TypeError:
                pass

        # TODO : NEed to understand why this raises an error in voronoi_drawer.

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

            self.mat_plot_2d.plot_array(
                array=sub_size,
                visuals_2d=self.get_visuals_2d_for_data(),
                auto_labels=AutoLabels(
                    title="Sub Pixels Per Image Pixels",
                    filename="sub_pixels_per_image_pixels",
                ),
            )

        if mesh_pixels_per_image_pixels:
            try:
                mesh_pixels_per_image_pixels = (
                    mapper_plotter.mapper.mapper_grids.mesh_pixels_per_image_pixels
                )

                self.mat_plot_2d.plot_array(
                    array=mesh_pixels_per_image_pixels,
                    visuals_2d=self.get_visuals_2d_for_data(),
                    auto_labels=AutoLabels(
                        title="Mesh Pixels Per Image Pixels",
                        filename="mesh_pixels_per_image_pixels",
                    ),
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
        """
        Plots the individual attributes of a specific `Mapper` of the plotter's `Inversion` object in 2D on a subplot.

        Parameters
        ----------
        mapper_index
            The index of the `Mapper` in the `Inversion`'s `linear_obj_list` that is plotted.
        auto_filename
            The default filename of the output subplot if written to hard-disk.
        """

        self.open_subplot_figure(number_subplots=12)

        contour_original = self.mat_plot_2d.contour

        if self.mat_plot_2d.use_log10:
            self.mat_plot_2d.contour = False

        mapper_image_plane_mesh_grid = self.include_2d._mapper_image_plane_mesh_grid

        self.include_2d._mapper_image_plane_mesh_grid = False

        self.figures_2d_of_pixelization(
            pixelization_index=mapper_index, data_subtracted=True
        )

        self.figures_2d_of_pixelization(
            pixelization_index=mapper_index, reconstructed_image=True
        )

        self.mat_plot_2d.use_log10 = True
        self.mat_plot_2d.contour = False

        self.figures_2d_of_pixelization(
            pixelization_index=mapper_index, reconstructed_image=True
        )

        self.mat_plot_2d.use_log10 = False

        self.include_2d._mapper_image_plane_mesh_grid = mapper_image_plane_mesh_grid
        self.include_2d._mapper_image_plane_mesh_grid = True
        self.set_title(label="Mesh Pixel Grid Overlaid")
        self.figures_2d_of_pixelization(
            pixelization_index=mapper_index, reconstructed_image=True
        )
        self.set_title(label=None)

        self.include_2d._mapper_image_plane_mesh_grid = False

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

        self.set_title(label="Errors (Unzoomed)")
        self.figures_2d_of_pixelization(
            pixelization_index=mapper_index, errors=True, zoom_to_brightest=False
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

        self.include_2d._mapper_image_plane_mesh_grid = False

        self.figures_2d_of_pixelization(
            pixelization_index=pixelization_index, data_subtracted=True
        )

        total_pixels = conf.instance["visualize"]["general"]["inversion"][
            "total_mappings_pixels"
        ]

        mapper = self.inversion.cls_list_from(cls=AbstractMapper)[pixelization_index]

        mapper_valued = MapperValued(
            values=self.inversion.reconstruction_dict[mapper], mapper=mapper
        )

        pix_indexes = mapper_valued.max_pixel_list_from(
            total_pixels=total_pixels, filter_neighbors=True
        )

        self.visuals_2d.pix_indexes = [
            [index] for index in pix_indexes[pixelization_index]
        ]

        self.figures_2d_of_pixelization(
            pixelization_index=pixelization_index, reconstructed_image=True
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
