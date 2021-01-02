from autoarray.plot.plotters import abstract_plotters
from autoarray.plot.mat_wrap import visuals as vis
from autoarray.plot.mat_wrap import include as inc
from autoarray.plot.mat_wrap import mat_plot
from autoarray.fit import fit as f
import typing


class FitImagingPlotter(abstract_plotters.AbstractPlotter):
    def __init__(
        self,
        fit: f.FitImaging,
        mat_plot_2d: mat_plot.MatPlot2D = mat_plot.MatPlot2D(),
        visuals_2d: vis.Visuals2D = vis.Visuals2D(),
        include_2d: inc.Include2D = inc.Include2D(),
    ):

        super().__init__(
            mat_plot_2d=mat_plot_2d, include_2d=include_2d, visuals_2d=visuals_2d
        )

        self.fit = fit

        self.visuals_2d = self.visuals_from_fit_imaging(fit=fit)

    def visuals_from_fit_imaging(self, fit: f.FitImaging):

        return self.visuals_from_structure(structure=fit.image)

    def subplot_fit_imaging(self):

        mat_plot_2d = self.mat_plot_2d.plotter_for_subplot_from(
            func=self.subplot_fit_imaging
        )

        number_subplots = 6

        mat_plot_2d.open_subplot_figure(number_subplots=number_subplots)

        mat_plot_2d.setup_subplot(number_subplots=number_subplots, subplot_index=1)

        self.figure_image()

        mat_plot_2d.setup_subplot(number_subplots=number_subplots, subplot_index=2)

        self.figure_signal_to_noise_map()

        mat_plot_2d.setup_subplot(number_subplots=number_subplots, subplot_index=3)

        self.figure_model_image()

        mat_plot_2d.setup_subplot(number_subplots=number_subplots, subplot_index=4)

        self.figure_residual_map()

        mat_plot_2d.setup_subplot(number_subplots=number_subplots, subplot_index=5)

        self.figure_normalized_residual_map()

        mat_plot_2d.setup_subplot(number_subplots=number_subplots, subplot_index=6)

        self.figure_chi_squared_map()

        mat_plot_2d.output.subplot_to_figure()

        mat_plot_2d.figure.close()

    def individuals(
        self,
        plot_image=False,
        plot_noise_map=False,
        plot_signal_to_noise_map=False,
        plot_model_image=False,
        plot_residual_map=False,
        plot_normalized_residual_map=False,
        plot_chi_squared_map=False,
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

        if plot_image:
            self.figure_image()
        if plot_noise_map:
            self.figure_noise_map()
        if plot_signal_to_noise_map:
            self.figure_signal_to_noise_map()
        if plot_model_image:
            self.figure_model_image()
        if plot_residual_map:
            self.figure_residual_map()
        if plot_normalized_residual_map:
            self.figure_normalized_residual_map()
        if plot_chi_squared_map:
            self.figure_chi_squared_map()

    @abstract_plotters.set_labels
    def figure_image(self):
        """Plot the image of a lens fit.
    
        Set *autolens.datas.array.mat_plot_2d.mat_plot_2d* for a description of all input parameters not described below.
    
        Parameters
        -----------
        image : datas.imaging.datas.Imaging
            The datas-datas, which include_2d the observed datas, noise_map, PSF, signal-to-noise_map, etc.
        origin : True
            If true, the origin of the datas's coordinate system is plotted as a 'x'.
        """
        self.mat_plot_2d.plot_array(array=self.fit.data, visuals_2d=self.visuals_2d)

    @abstract_plotters.set_labels
    def figure_noise_map(self):
        """Plot the noise-map of a lens fit.
    
        Set *autolens.datas.array.mat_plot_2d.mat_plot_2d* for a description of all input parameters not described below.
    
        Parameters
        -----------
        image : datas.imaging.datas.Imaging
            The datas-datas, which include_2d the observed datas, noise_map, PSF, signal-to-noise_map, etc.
        origin : True
            If true, the origin of the datas's coordinate system is plotted as a 'x'.
        """
        self.mat_plot_2d.plot_array(
            array=self.fit.noise_map, visuals_2d=self.visuals_2d
        )

    @abstract_plotters.set_labels
    def figure_signal_to_noise_map(self):
        """Plot the noise-map of a lens fit.
    
        Set *autolens.datas.array.mat_plot_2d.mat_plot_2d* for a description of all input parameters not described below.
    
        Parameters
        -----------
        image : datas.imaging.datas.Imaging
        The datas-datas, which include_2d the observed datas, signal_to_noise_map, PSF, signal-to-signal_to_noise_map, etc.
        origin : True
        If true, the origin of the datas's coordinate system is plotted as a 'x'.
        """
        self.mat_plot_2d.plot_array(
            array=self.fit.signal_to_noise_map, visuals_2d=self.visuals_2d
        )

    @abstract_plotters.set_labels
    def figure_model_image(self):
        """Plot the model image of a fit.
    
        Set *autolens.datas.array.mat_plot_2d.mat_plot_2d* for a description of all input parameters not described below.
    
        Parameters
        -----------
        fit : datas.fitting.fitting.AbstractFitter
            The fit to the datas, which include_2d a list of every model image, residual_map, chi-squareds, etc.
        image_index : int
            The index of the datas in the datas-set of which the model image is plotted.
        """
        self.mat_plot_2d.plot_array(
            array=self.fit.model_data, visuals_2d=self.visuals_2d
        )

    @abstract_plotters.set_labels
    def figure_residual_map(self):
        """Plot the residual-map of a lens fit.
    
        Set *autolens.datas.array.mat_plot_2d.mat_plot_2d* for a description of all input parameters not described below.
    
        Parameters
        -----------
        fit : datas.fitting.fitting.AbstractFitter
            The fit to the datas, which include_2d a list of every model image, residual_map, chi-squareds, etc.
        image_index : int
            The index of the datas in the datas-set of which the residual_map are plotted.
        """
        self.mat_plot_2d.plot_array(
            array=self.fit.residual_map, visuals_2d=self.visuals_2d
        )

    @abstract_plotters.set_labels
    def figure_normalized_residual_map(self):
        """Plot the residual-map of a lens fit.
    
        Set *autolens.datas.array.mat_plot_2d.mat_plot_2d* for a description of all input parameters not described below.
    
        Parameters
        -----------
        fit : datas.fitting.fitting.AbstractFitter
            The fit to the datas, which include_2d a list of every model image, normalized_residual_map, chi-squareds, etc.
        image_index : int
            The index of the datas in the datas-set of which the normalized_residual_map are plotted.
        """
        self.mat_plot_2d.plot_array(
            array=self.fit.normalized_residual_map, visuals_2d=self.visuals_2d
        )

    @abstract_plotters.set_labels
    def figure_chi_squared_map(self):
        """Plot the chi-squared-map of a lens fit.
    
        Set *autolens.datas.array.mat_plot_2d.mat_plot_2d* for a description of all input parameters not described below.
    
        Parameters
        -----------
        fit : datas.fitting.fitting.AbstractFitter
            The fit to the datas, which include_2d a list of every model image, residual_map, chi-squareds, etc.
        image_index : int
            The index of the datas in the datas-set of which the chi-squareds are plotted.
        """
        self.mat_plot_2d.plot_array(
            array=self.fit.chi_squared_map, visuals_2d=self.visuals_2d
        )
