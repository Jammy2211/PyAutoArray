from autoarray.plot.plotters import abstract_plotters
from autoarray.plot.mat_wrap import visuals as vis
from autoarray.plot.mat_wrap import include as inc
from autoarray.plot.mat_wrap import mat_plot


class InversionPlotter(abstract_plotters.AbstractPlotter):
    def __init__(
        self,
        mat_plot_2d: mat_plot.MatPlot2D = mat_plot.MatPlot2D(),
        visuals_2d: vis.Visuals2D = vis.Visuals2D(),
        include_2d: inc.Include2D = inc.Include2D(),
    ):
        super().__init__(
            mat_plot_2d=mat_plot_2d, include_2d=include_2d, visuals_2d=visuals_2d
        )

    def subplot_inversion(
        self, inversion, full_indexes=None, pixelization_indexes=None
    ):

        mat_plot_2d = self.mat_plot_2d.plotter_for_subplot_from(
            func=self.subplot_inversion
        )

        number_subplots = 6

        aspect_inv = mat_plot_2d.figure.aspect_for_subplot_from_grid(
            grid=inversion.mapper.source_full_grid
        )

        mat_plot_2d.open_subplot_figure(number_subplots=number_subplots)

        mat_plot_2d.setup_subplot(number_subplots=number_subplots, subplot_index=1)

        self.reconstructed_image(inversion=inversion)

        mat_plot_2d.setup_subplot(
            number_subplots=number_subplots, subplot_index=2, aspect=aspect_inv
        )

        self.reconstruction(
            inversion=inversion,
            full_indexes=full_indexes,
            pixelization_indexes=pixelization_indexes,
        )

        mat_plot_2d.setup_subplot(
            number_subplots=number_subplots, subplot_index=3, aspect=aspect_inv
        )

        self.errors(
            inversion=inversion,
            full_indexes=full_indexes,
            pixelization_indexes=pixelization_indexes,
        )

        mat_plot_2d.setup_subplot(
            number_subplots=number_subplots, subplot_index=4, aspect=aspect_inv
        )

        self.residual_map(
            inversion=inversion,
            full_indexes=full_indexes,
            pixelization_indexes=pixelization_indexes,
        )

        mat_plot_2d.setup_subplot(
            number_subplots=number_subplots, subplot_index=5, aspect=aspect_inv
        )

        self.chi_squared_map(
            inversion=inversion,
            full_indexes=full_indexes,
            pixelization_indexes=pixelization_indexes,
        )

        mat_plot_2d.setup_subplot(
            number_subplots=number_subplots, subplot_index=6, aspect=aspect_inv
        )

        self.regularization_weights(
            inversion=inversion,
            full_indexes=full_indexes,
            pixelization_indexes=pixelization_indexes,
        )

        mat_plot_2d.output.subplot_to_figure()

        mat_plot_2d.figure.close()

    def individuals(
        self,
        inversion,
        plot_reconstructed_image=False,
        plot_reconstruction=False,
        plot_errors=False,
        plot_residual_map=False,
        plot_normalized_residual_map=False,
        plot_chi_squared_map=False,
        plot_regularization_weight_map=False,
        plot_interpolated_reconstruction=False,
        plot_interpolated_errors=False,
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

        if plot_reconstructed_image:
            self.reconstructed_image(inversion=inversion)
        if plot_reconstruction:
            self.reconstruction(inversion=inversion)
        if plot_errors:
            self.errors(inversion=inversion)
        if plot_residual_map:
            self.residual_map(inversion=inversion)
        if plot_normalized_residual_map:
            self.normalized_residual_map(inversion=inversion)
        if plot_chi_squared_map:
            self.chi_squared_map(inversion=inversion)
        if plot_regularization_weight_map:
            self.regularization_weights(inversion=inversion)
        if plot_interpolated_reconstruction:
            self.interpolated_reconstruction(inversion=inversion)
        if plot_interpolated_errors:
            self.interpolated_errors(inversion=inversion)

    @abstract_plotters.set_labels
    def reconstructed_image(self, inversion):
        self.mat_plot_2d.plot_array(
            array=inversion.mapped_reconstructed_image,
            visuals_2d=self.visuals_2d
            + self.include_2d.visuals_of_data_from_mapper(mapper=inversion.mapper),
        )

    @abstract_plotters.set_labels
    def reconstruction(self, inversion, full_indexes=None, pixelization_indexes=None):

        source_pixelilzation_values = inversion.mapper.reconstructed_source_pixelization_from_solution_vector(
            solution_vector=inversion.reconstruction
        )

        self.mat_plot_2d.plot_mapper(
            mapper=inversion.mapper,
            visuals_2d=self.visuals_2d
            + self.include_2d.visuals_of_data_from_mapper(mapper=inversion.mapper),
            source_pixelilzation_values=source_pixelilzation_values,
            full_indexes=full_indexes,
            pixelization_indexes=pixelization_indexes,
        )

    @abstract_plotters.set_labels
    def errors(self, inversion, full_indexes=None, pixelization_indexes=None):

        source_pixelilzation_values = inversion.mapper.reconstructed_source_pixelization_from_solution_vector(
            solution_vector=inversion.errors
        )

        self.mat_plot_2d.plot_mapper(
            mapper=inversion.mapper,
            visuals_2d=self.visuals_2d
            + self.include_2d.visuals_of_data_from_mapper(mapper=inversion.mapper),
            source_pixelilzation_values=source_pixelilzation_values,
            full_indexes=full_indexes,
            pixelization_indexes=pixelization_indexes,
        )

    @abstract_plotters.set_labels
    def residual_map(self, inversion, full_indexes=None, pixelization_indexes=None):

        source_pixelilzation_values = inversion.mapper.reconstructed_source_pixelization_from_solution_vector(
            solution_vector=inversion.residual_map
        )

        self.mat_plot_2d.plot_mapper(
            mapper=inversion.mapper,
            visuals_2d=self.visuals_2d
            + self.include_2d.visuals_of_data_from_mapper(mapper=inversion.mapper),
            source_pixelilzation_values=source_pixelilzation_values,
            full_indexes=full_indexes,
            pixelization_indexes=pixelization_indexes,
        )

    @abstract_plotters.set_labels
    def normalized_residual_map(
        self, inversion, full_indexes=None, pixelization_indexes=None
    ):

        source_pixelilzation_values = inversion.mapper.reconstructed_source_pixelization_from_solution_vector(
            solution_vector=inversion.normalized_residual_map
        )

        self.mat_plot_2d.plot_mapper(
            mapper=inversion.mapper,
            visuals_2d=self.visuals_2d
            + self.include_2d.visuals_of_data_from_mapper(mapper=inversion.mapper),
            source_pixelilzation_values=source_pixelilzation_values,
            full_indexes=full_indexes,
            pixelization_indexes=pixelization_indexes,
        )

    @abstract_plotters.set_labels
    def chi_squared_map(self, inversion, full_indexes=None, pixelization_indexes=None):

        source_pixelilzation_values = inversion.mapper.reconstructed_source_pixelization_from_solution_vector(
            solution_vector=inversion.chi_squared_map
        )

        self.mat_plot_2d.plot_mapper(
            mapper=inversion.mapper,
            visuals_2d=self.visuals_2d
            + self.include_2d.visuals_of_data_from_mapper(mapper=inversion.mapper),
            source_pixelilzation_values=source_pixelilzation_values,
            full_indexes=full_indexes,
            pixelization_indexes=pixelization_indexes,
        )

    @abstract_plotters.set_labels
    def regularization_weights(
        self, inversion, full_indexes=None, pixelization_indexes=None
    ):

        regularization_weights = inversion.regularization.regularization_weights_from_mapper(
            mapper=inversion.mapper
        )

        self.mat_plot_2d.plot_mapper(
            mapper=inversion.mapper,
            visuals_2d=self.visuals_2d
            + self.include_2d.visuals_of_data_from_mapper(mapper=inversion.mapper),
            source_pixelilzation_values=regularization_weights,
            full_indexes=full_indexes,
            pixelization_indexes=pixelization_indexes,
        )

    @abstract_plotters.set_labels
    def interpolated_reconstruction(self, inversion):

        self.mat_plot_2d.plot_array(
            array=inversion.interpolated_reconstructed_data_from_shape_2d(),
            visuals_2d=self.visuals_2d
            + self.include_2d.visuals_of_data_from_mapper(mapper=inversion.mapper),
        )

    @abstract_plotters.set_labels
    def interpolated_errors(self, inversion):

        self.mat_plot_2d.plot_array(
            array=inversion.interpolated_errors_from_shape_2d(),
            visuals_2d=self.visuals_2d
            + self.include_2d.visuals_of_data_from_mapper(mapper=inversion.mapper),
        )
