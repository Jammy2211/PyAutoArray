from autoconf import conf
from autoarray.plot.mat_wrap import visuals as vis
from autoarray.plot.mat_wrap import include as inc
from autoarray.plot.mat_wrap import mat_plot as mp
from autoarray.plot import structure_plotters
from autoarray.inversion.inversion.abstract import AbstractInversion

import numpy as np


class InversionPlotter(structure_plotters.MapperPlotter):
    def __init__(
        self,
        inversion: AbstractInversion,
        mat_plot_2d: mp.MatPlot2D = mp.MatPlot2D(),
        visuals_2d: vis.Visuals2D = vis.Visuals2D(),
        include_2d: inc.Include2D = inc.Include2D(),
    ):
        super().__init__(
            mapper=inversion.mapper,
            mat_plot_2d=mat_plot_2d,
            include_2d=include_2d,
            visuals_2d=visuals_2d,
        )

        self.inversion = inversion

    def as_mapper(self, source_pixelization_values):
        return self.inversion.mapper.reconstruction_from(source_pixelization_values)

    def figures_2d(
        self,
        reconstructed_image=False,
        reconstruction=False,
        errors=False,
        residual_map=False,
        normalized_residual_map=False,
        chi_squared_map=False,
        regularization_weight_list=False,
        interpolated_reconstruction=False,
        interpolated_errors=False,
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
                auto_labels=mp.AutoLabels(
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
                auto_labels=mp.AutoLabels(
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
                auto_labels=mp.AutoLabels(title="Errors", filename="errors"),
                source_pixelilzation_values=self.as_mapper(self.inversion.errors),
            )

        if residual_map:

            self.mat_plot_2d.plot_mapper(
                mapper=self.inversion.mapper,
                visuals_2d=self.visuals_source_with_include_2d,
                auto_labels=mp.AutoLabels(
                    title="Residual Map", filename="residual_map"
                ),
                source_pixelilzation_values=self.as_mapper(self.inversion.residual_map),
            )

        if normalized_residual_map:

            self.mat_plot_2d.plot_mapper(
                mapper=self.inversion.mapper,
                visuals_2d=self.visuals_source_with_include_2d,
                auto_labels=mp.AutoLabels(
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
                auto_labels=mp.AutoLabels(
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
                auto_labels=mp.AutoLabels(
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
                auto_labels=mp.AutoLabels(
                    title="Interpolated Reconstruction",
                    filename="interpolated_reconstruction",
                ),
            )

        if interpolated_errors:
            self.mat_plot_2d.plot_array(
                array=self.inversion.interpolated_errors_from_shape_native(),
                visuals_2d=self.visuals_data_with_include_2d,
                auto_labels=mp.AutoLabels(
                    title="Interpolated Errors", filename="interpolated_errors"
                ),
            )

    def subplot(
        self,
        reconstructed_image=False,
        reconstruction=False,
        errors=False,
        residual_map=False,
        normalized_residual_map=False,
        chi_squared_map=False,
        regularization_weight_list=False,
        interpolated_reconstruction=False,
        interpolated_errors=False,
        auto_filename="subplot_inversion",
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
            auto_labels=mp.AutoLabels(filename=auto_filename),
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
