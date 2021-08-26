import numpy as np
from typing import Union

from autoconf import conf
from autoarray.plot.mat_wrap.visuals import Visuals2D
from autoarray.plot.mat_wrap.include import Include2D
from autoarray.plot.mat_wrap.mat_plot import MatPlot2D
from autoarray.plot.mat_wrap.mat_plot import AutoLabels
from autoarray.structures.arrays.two_d.array_2d import Array2D
from autoarray.inversion.plot.mapper_plotters import MapperPlotter
from autoarray.inversion.inversion.imaging import InversionImagingWTilde
from autoarray.inversion.inversion.interferometer import InversionInterferometerMapping


class InversionPlotter(MapperPlotter):
    def __init__(
        self,
        inversion: Union[InversionImagingWTilde, InversionInterferometerMapping],
        mat_plot_2d: MatPlot2D = MatPlot2D(),
        visuals_2d: Visuals2D = Visuals2D(),
        include_2d: Include2D = Include2D(),
    ):
        super().__init__(
            mapper=inversion.mapper,
            mat_plot_2d=mat_plot_2d,
            include_2d=include_2d,
            visuals_2d=visuals_2d,
        )

        self.inversion = inversion

    def as_mapper(self, source_pixelization_values) -> Array2D:
        return self.inversion.mapper.reconstruction_from(source_pixelization_values)

    def figures_2d(
        self,
        reconstructed_image: bool = False,
        reconstruction: bool = False,
        errors: bool = False,
        residual_map: bool = False,
        normalized_residual_map: bool = False,
        chi_squared_map: bool = False,
        regularization_weight_list: bool = False,
        interpolated_reconstruction: bool = False,
        interpolated_errors: bool = False,
    ):
        """Plot the model data of an analysis, using the *Fitter* class object.

        The visualization and output type can be fully customized.

        Parameters
        -----------
        fit : autolens.lens.fitting.Fitter
            Class containing fit between the model data and observed lens data (including residual_map, chi_squared_map etc.)
        output_path : str
            The path where the data is output if the output_type is a file format (e.g. png, fits)
        output_format : str
            How the data is output. File formats (e.g. png, fits) output the data to harddisk. 'show' displays the data
            in the python interpreter window.
        """

        if reconstructed_image:

            self.mat_plot_2d.plot_array(
                array=self.inversion.mapped_reconstructed_image,
                visuals_2d=self.visuals_data_with_include_2d,
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

            self.mat_plot_2d.plot_mapper(
                mapper=self.inversion.mapper,
                visuals_2d=self.visuals_source_with_include_2d,
                auto_labels=AutoLabels(
                    title="Source Reconstruction", filename="reconstruction"
                ),
                source_pixelilzation_values=self.as_mapper(
                    self.inversion.reconstruction
                ),
            )

            if vmax_custom:
                self.mat_plot_2d.cmap.kwargs["vmax"] = None

        if errors:

            self.mat_plot_2d.plot_mapper(
                mapper=self.inversion.mapper,
                visuals_2d=self.visuals_source_with_include_2d,
                auto_labels=AutoLabels(title="Errors", filename="errors"),
                source_pixelilzation_values=self.as_mapper(self.inversion.errors),
            )

        if residual_map:

            self.mat_plot_2d.plot_mapper(
                mapper=self.inversion.mapper,
                visuals_2d=self.visuals_source_with_include_2d,
                auto_labels=AutoLabels(title="Residual Map", filename="residual_map"),
                source_pixelilzation_values=self.as_mapper(self.inversion.residual_map),
            )

        if normalized_residual_map:

            self.mat_plot_2d.plot_mapper(
                mapper=self.inversion.mapper,
                visuals_2d=self.visuals_source_with_include_2d,
                auto_labels=AutoLabels(
                    title="Normalized Residual Map", filename="normalized_residual_map"
                ),
                source_pixelilzation_values=self.as_mapper(
                    self.inversion.normalized_residual_map
                ),
            )

        if chi_squared_map:

            self.mat_plot_2d.plot_mapper(
                mapper=self.inversion.mapper,
                visuals_2d=self.visuals_source_with_include_2d,
                auto_labels=AutoLabels(
                    title="Chi-Squared Map", filename="chi_squared_map"
                ),
                source_pixelilzation_values=self.as_mapper(
                    self.inversion.chi_squared_map
                ),
            )

        if regularization_weight_list:

            self.mat_plot_2d.plot_mapper(
                mapper=self.inversion.mapper,
                visuals_2d=self.visuals_source_with_include_2d,
                auto_labels=AutoLabels(
                    title="Regularization weight_list",
                    filename="regularization_weight_list",
                ),
                source_pixelilzation_values=self.as_mapper(
                    self.inversion.regularization_weight_list
                ),
            )

        if interpolated_reconstruction:

            self.mat_plot_2d.plot_array(
                array=self.inversion.interpolated_reconstructed_data_from_shape_native(),
                visuals_2d=self.visuals_data_with_include_2d,
                auto_labels=AutoLabels(
                    title="Interpolated Reconstruction",
                    filename="interpolated_reconstruction",
                ),
            )

        if interpolated_errors:
            self.mat_plot_2d.plot_array(
                array=self.inversion.interpolated_errors_from_shape_native(),
                visuals_2d=self.visuals_data_with_include_2d,
                auto_labels=AutoLabels(
                    title="Interpolated Errors", filename="interpolated_errors"
                ),
            )

    def subplot(
        self,
        reconstructed_image: bool = False,
        reconstruction: bool = False,
        errors: bool = False,
        residual_map: bool = False,
        normalized_residual_map: bool = False,
        chi_squared_map: bool = False,
        regularization_weight_list: bool = False,
        interpolated_reconstruction: bool = False,
        interpolated_errors: bool = False,
        auto_filename: str = "subplot_inversion",
    ):

        self._subplot_custom_plot(
            reconstructed_image=reconstructed_image,
            reconstruction=reconstruction,
            errors=errors,
            residual_map=residual_map,
            normalized_residual_map=normalized_residual_map,
            chi_squared_map=chi_squared_map,
            regularization_weight_list=regularization_weight_list,
            interpolated_reconstruction=interpolated_reconstruction,
            interpolated_errors=interpolated_errors,
            auto_labels=AutoLabels(filename=auto_filename),
        )

    def subplot_inversion(self):
        return self.subplot(
            reconstructed_image=True,
            reconstruction=True,
            errors=True,
            residual_map=True,
            chi_squared_map=True,
            regularization_weight_list=True,
        )
