from autoarray.plot.plotters import abstract_plotters
from autoarray.plot.mat_wrap import visuals as vis
from autoarray.plot.mat_wrap import include as inc
from autoarray.plot.mat_wrap import mat_plot
from autoarray.plot.plotters import structure_plotters
from autoarray.inversion import inversions as inv


class InversionPlotter(structure_plotters.MapperPlotter):
    def __init__(
        self,
        inversion: inv.AbstractInversion,
        mat_plot_2d: mat_plot.MatPlot2D = mat_plot.MatPlot2D(),
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

    def subplot_inversion(self, full_indexes=None, pixelization_indexes=None):

        mat_plot_2d = self.mat_plot_2d.plotter_for_subplot_from(
            func=self.subplot_inversion
        )

        number_subplots = 6

        aspect_inv = mat_plot_2d.figure.aspect_for_subplot_from_grid(
            grid=self.inversion.mapper.source_full_grid
        )

        mat_plot_2d.open_subplot_figure(number_subplots=number_subplots)

        mat_plot_2d.setup_subplot(number_subplots=number_subplots, subplot_index=1)

        self.figure_reconstructed_image()

        mat_plot_2d.setup_subplot(
            number_subplots=number_subplots, subplot_index=2, aspect=aspect_inv
        )

        self.figure_reconstruction(
            full_indexes=full_indexes, pixelization_indexes=pixelization_indexes
        )

        mat_plot_2d.setup_subplot(
            number_subplots=number_subplots, subplot_index=3, aspect=aspect_inv
        )

        self.figure_errors(
            full_indexes=full_indexes, pixelization_indexes=pixelization_indexes
        )

        mat_plot_2d.setup_subplot(
            number_subplots=number_subplots, subplot_index=4, aspect=aspect_inv
        )

        self.figure_residual_map(
            full_indexes=full_indexes, pixelization_indexes=pixelization_indexes
        )

        mat_plot_2d.setup_subplot(
            number_subplots=number_subplots, subplot_index=5, aspect=aspect_inv
        )

        self.figure_chi_squared_map(
            full_indexes=full_indexes, pixelization_indexes=pixelization_indexes
        )

        mat_plot_2d.setup_subplot(
            number_subplots=number_subplots, subplot_index=6, aspect=aspect_inv
        )

        self.figure_regularization_weights(
            full_indexes=full_indexes, pixelization_indexes=pixelization_indexes
        )

        mat_plot_2d.output.subplot_to_figure()

        mat_plot_2d.figure.close()

    def individuals(
        self,
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
            self.figure_reconstructed_image()
        if plot_reconstruction:
            self.figure_reconstruction()
        if plot_errors:
            self.figure_errors()
        if plot_residual_map:
            self.figure_residual_map()
        if plot_normalized_residual_map:
            self.figure_normalized_residual_map()
        if plot_chi_squared_map:
            self.figure_chi_squared_map()
        if plot_regularization_weight_map:
            self.figure_regularization_weights()
        if plot_interpolated_reconstruction:
            self.figure_interpolated_reconstruction()
        if plot_interpolated_errors:
            self.figure_interpolated_errors()

    @abstract_plotters.set_labels
    def figure_reconstructed_image(self):
        self.mat_plot_2d.plot_array(
            array=self.inversion.mapped_reconstructed_image, visuals_2d=self.visuals_2d
        )

    @abstract_plotters.set_labels
    def figure_reconstruction(self, full_indexes=None, pixelization_indexes=None):

        source_pixelilzation_values = self.inversion.mapper.reconstructed_source_pixelization_from_solution_vector(
            solution_vector=self.inversion.reconstruction
        )

        self.mat_plot_2d.plot_mapper(
            mapper=self.inversion.mapper,
            source_pixelilzation_values=source_pixelilzation_values,
            full_indexes=full_indexes,
            pixelization_indexes=pixelization_indexes,
        )

    @abstract_plotters.set_labels
    def figure_errors(self, full_indexes=None, pixelization_indexes=None):

        source_pixelilzation_values = self.inversion.mapper.reconstructed_source_pixelization_from_solution_vector(
            solution_vector=self.inversion.errors
        )

        self.mat_plot_2d.plot_mapper(
            mapper=self.inversion.mapper,
            visuals_2d=self.visuals_2d,
            source_pixelilzation_values=source_pixelilzation_values,
            full_indexes=full_indexes,
            pixelization_indexes=pixelization_indexes,
        )

    @abstract_plotters.set_labels
    def figure_residual_map(self, full_indexes=None, pixelization_indexes=None):

        source_pixelilzation_values = self.inversion.mapper.reconstructed_source_pixelization_from_solution_vector(
            solution_vector=self.inversion.residual_map
        )

        self.mat_plot_2d.plot_mapper(
            mapper=self.inversion.mapper,
            visuals_2d=self.visuals_2d,
            source_pixelilzation_values=source_pixelilzation_values,
            full_indexes=full_indexes,
            pixelization_indexes=pixelization_indexes,
        )

    @abstract_plotters.set_labels
    def figure_normalized_residual_map(
        self, full_indexes=None, pixelization_indexes=None
    ):

        source_pixelilzation_values = self.inversion.mapper.reconstructed_source_pixelization_from_solution_vector(
            solution_vector=self.inversion.normalized_residual_map
        )

        self.mat_plot_2d.plot_mapper(
            mapper=self.inversion.mapper,
            visuals_2d=self.visuals_2d,
            source_pixelilzation_values=source_pixelilzation_values,
            full_indexes=full_indexes,
            pixelization_indexes=pixelization_indexes,
        )

    @abstract_plotters.set_labels
    def figure_chi_squared_map(self, full_indexes=None, pixelization_indexes=None):

        source_pixelilzation_values = self.inversion.mapper.reconstructed_source_pixelization_from_solution_vector(
            solution_vector=self.inversion.chi_squared_map
        )

        self.mat_plot_2d.plot_mapper(
            mapper=self.inversion.mapper,
            visuals_2d=self.visuals_2d,
            source_pixelilzation_values=source_pixelilzation_values,
            full_indexes=full_indexes,
            pixelization_indexes=pixelization_indexes,
        )

    @abstract_plotters.set_labels
    def figure_regularization_weights(
        self, full_indexes=None, pixelization_indexes=None
    ):

        regularization_weights = self.inversion.regularization.regularization_weights_from_mapper(
            mapper=self.inversion.mapper
        )

        self.mat_plot_2d.plot_mapper(
            mapper=self.inversion.mapper,
            visuals_2d=self.visuals_2d,
            source_pixelilzation_values=regularization_weights,
            full_indexes=full_indexes,
            pixelization_indexes=pixelization_indexes,
        )

    @abstract_plotters.set_labels
    def figure_interpolated_reconstruction(self):

        self.mat_plot_2d.plot_array(
            array=self.inversion.interpolated_reconstructed_data_from_shape_2d(),
            visuals_2d=self.visuals_2d,
        )

    @abstract_plotters.set_labels
    def figure_interpolated_errors(self):

        self.mat_plot_2d.plot_array(
            array=self.inversion.interpolated_errors_from_shape_2d(),
            visuals_2d=self.visuals_2d,
        )
