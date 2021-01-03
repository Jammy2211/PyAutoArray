from autoarray.plot.plotters import abstract_plotters
from autoarray.plot.mat_wrap import visuals as vis
from autoarray.plot.mat_wrap import include as inc
from autoarray.plot.mat_wrap import mat_plot
from autoarray.fit import fit as f
import numpy as np

import copy


class FitInterferometerPlotter(abstract_plotters.AbstractPlotter):
    def __init__(
        self,
        fit: f.FitInterferometer,
        mat_plot_1d: mat_plot.MatPlot1D = mat_plot.MatPlot1D(),
        visuals_1d: vis.Visuals1D = vis.Visuals1D(),
        include_1d: inc.Include1D = inc.Include1D(),
        mat_plot_2d: mat_plot.MatPlot2D = mat_plot.MatPlot2D(),
        visuals_2d: vis.Visuals2D = vis.Visuals2D(),
        include_2d: inc.Include2D = inc.Include2D(),
    ):

        super().__init__(
            mat_plot_1d=mat_plot_1d,
            include_1d=include_1d,
            visuals_1d=visuals_1d,
            mat_plot_2d=mat_plot_2d,
            include_2d=include_2d,
            visuals_2d=visuals_2d,
        )

        self.fit = fit

    @property
    def visuals_with_include_2d(self):

        return copy.deepcopy(self.visuals_2d)

    @abstract_plotters.set_labels
    def figure_visibilities(self):
        """Plot the visibilities of a lens fit.
    
        Set *autolens.datas.grid.mat_plot_2d.mat_plot_2d* for a description of all input parameters not described below.
    
        Parameters
        -----------
        visibilities : datas.imaging.datas.Imaging
            The datas-datas, which include_2d the observed datas, noise_map, PSF, signal-to-noise_map, etc.
        origin : True
            If true, the origin of the datas's coordinate system is plotted as a 'x'.
        """
        self.mat_plot_2d.plot_grid(
            grid=self.fit.visibilities.in_grid,
            visuals_2d=self.visuals_2d,
            color_array=np.real(self.fit.noise_map),
        )

    @abstract_plotters.set_labels
    def figure_noise_map(self):
        """Plot the noise-map of a lens fit.
    
        Set *autolens.datas.grid.mat_plot_2d.mat_plot_2d* for a description of all input parameters not described below.
    
        Parameters
        -----------
        visibilities : datas.imaging.datas.Imaging
            The datas-datas, which include_2d the observed datas, noise_map, PSF, signal-to-noise_map, etc.
        origin : True
            If true, the origin of the datas's coordinate system is plotted as a 'x'.
        """
        self.mat_plot_2d.plot_grid(
            grid=self.fit.visibilities.in_grid,
            visuals_2d=self.visuals_2d,
            color_array=np.real(self.fit.noise_map),
        )

    @abstract_plotters.set_labels
    def figure_signal_to_noise_map(self):
        """Plot the noise-map of a lens fit.
    
        Set *autolens.datas.grid.mat_plot_2d.mat_plot_2d* for a description of all input parameters not described below.
    
        Parameters
        -----------
        visibilities : datas.imaging.datas.Imaging
        The datas-datas, which include_2d the observed datas, signal_to_noise_map, PSF, signal-to-signal_to_noise_map, etc.
        origin : True
        If true, the origin of the datas's coordinate system is plotted as a 'x'.
        """
        self.mat_plot_2d.plot_grid(
            grid=self.fit.visibilities.in_grid,
            visuals_2d=self.visuals_2d,
            color_array=np.real(self.fit.signal_to_noise_map),
        )

    @abstract_plotters.set_labels
    def figure_model_visibilities(self):
        """Plot the model visibilities of a fit.
    
        Set *autolens.datas.grid.mat_plot_2d.mat_plot_2d* for a description of all input parameters not described below.
    
        Parameters
        -----------
        fit : datas.fitting.fitting.AbstractFitter
            The fit to the datas, which include_2d a list of every model visibilities, residual_map, chi-squareds, etc.
        visibilities_index : int
            The index of the datas in the datas-set of which the model visibilities is plotted.
        """
        self.mat_plot_2d.plot_grid(
            grid=self.fit.visibilities.in_grid,
            visuals_2d=self.visuals_2d,
            color_array=np.real(self.fit.model_data),
        )

    @abstract_plotters.set_labels
    def figure_residual_map_vs_uv_distances(
        self,
        plot_real=True,
        label_yunits="V$_{R,data}$ - V$_{R,model}$",
        label_xunits=r"UV$_{distance}$ (k$\lambda$)",
    ):
        """Plot the residual-map of a lens fit.
    
        Set *autolens.datas.grid.mat_plot_1d.mat_plot_1d* for a description of all input parameters not described below.
    
        Parameters
        -----------
        fit : datas.fitting.fitting.AbstractFitter
            The fit to the datas, which include_2d a list of every model visibilities, residual_map, chi-squareds, etc.
        visibilities_index : int
            The index of the datas in the datas-set of which the residual_map are plotted.
        """

        if plot_real:
            y = np.real(self.fit.residual_map)
            mat_plot_1d = self.mat_plot_1d.mat_plot_with_new_labels(
                title_label=f"{self.mat_plot_1d.title.kwargs['label']} Real"
            )
            mat_plot_1d = mat_plot_1d.mat_plot_with_new_output(
                filename=mat_plot_1d.output.filename + "_real"
            )
        else:
            y = np.imag(self.fit.residual_map)
            mat_plot_1d = self.mat_plot_1d.mat_plot_with_new_labels(
                title_label=f"{self.mat_plot_1d.title.kwargs['label']} Imag"
            )
            mat_plot_1d = mat_plot_1d.mat_plot_with_new_output(
                filename=mat_plot_1d.output.filename + "_imag"
            )

        mat_plot_1d.plot_line(
            y=y,
            x=self.fit.masked_interferometer.interferometer.uv_distances / 10 ** 3.0,
            plot_axis_type="scatter",
        )

    @abstract_plotters.set_labels
    def figure_normalized_residual_map_vs_uv_distances(
        self,
        plot_real=True,
        label_yunits="V$_{R,data}$ - V$_{R,model}$",
        label_xunits=r"UV$_{distance}$ (k$\lambda$)",
    ):
        """Plot the residual-map of a lens fit.
    
        Set *autolens.datas.grid.mat_plot_1d.mat_plot_1d* for a description of all input parameters not described below.
    
        Parameters
        -----------
        fit : datas.fitting.fitting.AbstractFitter
            The fit to the datas, which include_2d a list of every model visibilities, residual_map, chi-squareds, etc.
        visibilities_index : int
            The index of the datas in the datas-set of which the residual_map are plotted.
        """

        if plot_real:
            y = np.real(self.fit.residual_map)
            mat_plot_1d = self.mat_plot_1d.mat_plot_with_new_labels(
                title_label=f"{self.mat_plot_1d.title.kwargs['label']} Real"
            )
            mat_plot_1d = mat_plot_1d.mat_plot_with_new_output(
                filename=mat_plot_1d.output.filename + "_real"
            )
        else:
            y = np.imag(self.fit.residual_map)
            mat_plot_1d = self.mat_plot_1d.mat_plot_with_new_labels(
                title_label=f"{self.mat_plot_1d.title.kwargs['label']} Imag"
            )
            mat_plot_1d = mat_plot_1d.mat_plot_with_new_output(
                filename=mat_plot_1d.output.filename + "_imag"
            )

        mat_plot_1d.plot_line(
            y=y,
            x=self.fit.masked_interferometer.interferometer.uv_distances / 10 ** 3.0,
            plot_axis_type="scatter",
        )

    @abstract_plotters.set_labels
    def figure_chi_squared_map_vs_uv_distances(
        self,
        plot_real=True,
        label_yunits="V$_{R,data}$ - V$_{R,model}$",
        label_xunits=r"UV$_{distance}$ (k$\lambda$)",
    ):
        """Plot the residual-map of a lens fit.
    
        Set *autolens.datas.grid.mat_plot_1d.mat_plot_1d* for a description of all input parameters not described below.
    
        Parameters
        -----------
        fit : datas.fitting.fitting.AbstractFitter
            The fit to the datas, which include_2d a list of every model visibilities, residual_map, chi-squareds, etc.
        visibilities_index : int
            The index of the datas in the datas-set of which the residual_map are plotted.
        """

        if plot_real:
            y = np.real(self.fit.residual_map)
            mat_plot_1d = self.mat_plot_1d.mat_plot_with_new_labels(
                title_label=f"{self.mat_plot_1d.title.kwargs['label']} Real"
            )
            mat_plot_1d = mat_plot_1d.mat_plot_with_new_output(
                filename=mat_plot_1d.output.filename + "_real"
            )
        else:
            y = np.imag(self.fit.residual_map)
            mat_plot_1d = self.mat_plot_1d.mat_plot_with_new_labels(
                title_label=f"{self.mat_plot_1d.title.kwargs['label']} Imag"
            )
            mat_plot_1d = mat_plot_1d.mat_plot_with_new_output(
                filename=mat_plot_1d.output.filename + "_imag"
            )

        mat_plot_1d.plot_line(
            y=y,
            x=self.fit.masked_interferometer.interferometer.uv_distances / 10 ** 3.0,
            plot_axis_type="scatter",
        )

    def figure_individuals(
        self,
        plot_visibilities=False,
        plot_noise_map=False,
        plot_signal_to_noise_map=False,
        plot_model_visibilities=False,
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

        if plot_visibilities:
            self.figure_visibilities()
        if plot_noise_map:
            self.figure_noise_map()
        if plot_signal_to_noise_map:
            self.figure_signal_to_noise_map()
        if plot_model_visibilities:
            self.figure_model_visibilities()

        if plot_residual_map:
            self.figure_residual_map_vs_uv_distances(plot_real=True)
            self.figure_residual_map_vs_uv_distances(plot_real=False)

        if plot_normalized_residual_map:
            self.figure_normalized_residual_map_vs_uv_distances(plot_real=True)
            self.figure_normalized_residual_map_vs_uv_distances(plot_real=False)

        if plot_chi_squared_map:
            self.figure_chi_squared_map_vs_uv_distances(plot_real=True)
            self.figure_chi_squared_map_vs_uv_distances(plot_real=False)

    def subplot_fit_interferometer(self):

        mat_plot_1d = self.mat_plot_1d.mat_plot_for_subplot_from(
            func=self.subplot_fit_interferometer
        )

        number_subplots = 6

        mat_plot_1d.open_subplot_figure(number_subplots=number_subplots)

        mat_plot_1d.setup_subplot(number_subplots=number_subplots, subplot_index=1)

        self.figure_residual_map_vs_uv_distances()

        mat_plot_1d.setup_subplot(number_subplots=number_subplots, subplot_index=2)

        self.figure_normalized_residual_map_vs_uv_distances()

        mat_plot_1d.setup_subplot(number_subplots=number_subplots, subplot_index=3)

        self.figure_chi_squared_map_vs_uv_distances()

        mat_plot_1d.setup_subplot(number_subplots=number_subplots, subplot_index=4)

        self.figure_residual_map_vs_uv_distances(plot_real=False)

        mat_plot_1d.setup_subplot(number_subplots=number_subplots, subplot_index=5)

        self.figure_normalized_residual_map_vs_uv_distances(plot_real=False)

        mat_plot_1d.setup_subplot(number_subplots=number_subplots, subplot_index=6)

        self.figure_chi_squared_map_vs_uv_distances(plot_real=False)

        mat_plot_1d.output.subplot_to_figure()

        mat_plot_1d.figure.close()
