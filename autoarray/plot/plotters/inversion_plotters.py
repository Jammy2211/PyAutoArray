from autoarray.plot.plotters import abstract_plotters
from autoarray.plot.mat_wrap import visuals as vis
from autoarray.plot.mat_wrap import include as inc
from autoarray.plot.mat_wrap import mat_plot as mp
from autoarray.plot.plotters import structure_plotters
from autoarray.inversion import inversions as inv


class InversionPlotter(structure_plotters.MapperPlotter):
    def __init__(
        self,
        inversion: inv.AbstractInversion,
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

    def figures(
        self,
        reconstructed_image=False,
        reconstruction=False,
        errors=False,
        residual_map=False,
        normalized_residual_map=False,
        chi_squared_map=False,
        regularization_weights=False,
        interpolated_reconstruction=False,
        interpolated_errors=False,
    ):
        """Plot the model datas_ of an analysis, using the *Fitter* class object.

        The visualization and output type can be fully customized.

        Parameters
        -----------
        fit : autolens.lens.fitting.Fitter
            Class containing fit between the model datas_ and observed lens datas_ (including residual_map, chi_squared_map etc.)
        output_path : str
            The path where the datas_ is output if the output_type is a file format (e.g. png, fits)
        output_format : str
            How the datas_ is output. File formats (e.g. png, fits) output the datas_ to harddisk. 'show' displays the datas_ \
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

        if regularization_weights:

            self.mat_plot_2d.plot_mapper(
                mapper=self.inversion.mapper,
                visuals_2d=self.visuals_source_with_include_2d,
                auto_labels=mp.AutoLabels(
                    title="Regularization Weights", filename="regularization_weights"
                ),
                source_pixelilzation_values=self.as_mapper(
                    self.inversion.regularization_weights
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
        regularization_weights=False,
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
            regularization_weights=regularization_weights,
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
            regularization_weights=True,
        )
