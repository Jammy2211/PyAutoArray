from autoarray.plot.plotters import abstract_plotters
from autoarray.plot.mat_wrap import visuals as vis
from autoarray.plot.mat_wrap import include as inc
from autoarray.plot.mat_wrap import mat_plot as mp
from autoarray.fit import fit as f
import numpy as np


class AbstractFitInterferometerPlotter(abstract_plotters.AbstractPlotter):
    def __init__(
        self,
        fit: f.FitInterferometer,
        mat_plot_1d,
        visuals_1d,
        include_1d,
        mat_plot_2d,
        visuals_2d,
        include_2d,
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
        return self.visuals_2d + self.visuals_2d.__class__()

    def figures(
        self,
        visibilities=False,
        noise_map=False,
        signal_to_noise_map=False,
        model_visibilities=False,
        residual_map_real=False,
        residual_map_imag=False,
        normalized_residual_map_real=False,
        normalized_residual_map_imag=False,
        chi_squared_map_real=False,
        chi_squared_map_imag=False,
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

        if visibilities:
            self.plot_grid(
                grid=self.fit.visibilities.in_grid,
                visuals_2d=self.visuals_2d,
                auto_labels=mp.AutoLabels(
                    title="Visibilities", filename="visibilities"
                ),
                color_array=np.real(self.fit.noise_map),
            )

        if noise_map:
            self.plot_grid(
                grid=self.fit.visibilities.in_grid,
                visuals_2d=self.visuals_2d,
                auto_labels=mp.AutoLabels(title="Noise-Map", filename="noise_map"),
                color_array=np.real(self.fit.noise_map),
            )

        if signal_to_noise_map:
            self.plot_grid(
                grid=self.fit.visibilities.in_grid,
                visuals_2d=self.visuals_2d,
                auto_labels=mp.AutoLabels(
                    title="Signal-To-Noise Map", filename="signal_to_noise_map"
                ),
                color_array=np.real(self.fit.signal_to_noise_map),
            )
        if model_visibilities:
            self.plot_grid(
                grid=self.fit.visibilities.in_grid,
                visuals_2d=self.visuals_2d,
                auto_labels=mp.AutoLabels(
                    title="Model Visibilities", filename="model_visibilities"
                ),
                color_array=np.real(self.fit.model_data),
            )

        if residual_map_real:
            self.plot_line(
                y=np.real(self.fit.residual_map),
                x=self.fit.masked_interferometer.interferometer.uv_distances
                / 10 ** 3.0,
                visuals_1d=self.visuals_1d,
                auto_labels=mp.AutoLabels(
                    title="Residual Map vs UV-Distance (real)",
                    filename="real_residual_map_vs_uv_distances",
                    ylabel="V$_{R,data}$ - V$_{R,model}$",
                    xlabel=r"UV$_{distance}$ (k$\lambda$)",
                ),
                plot_axis_type="scatter",
            )
        if residual_map_imag:
            self.plot_line(
                y=np.imag(self.fit.residual_map),
                x=self.fit.masked_interferometer.interferometer.uv_distances
                / 10 ** 3.0,
                visuals_1d=self.visuals_1d,
                auto_labels=mp.AutoLabels(
                    title="Residual Map vs UV-Distance (imag)",
                    filename="imag_residual_map_vs_uv_distances",
                    ylabel="V$_{R,data}$ - V$_{R,model}$",
                    xlabel=r"UV$_{distance}$ (k$\lambda$)",
                ),
                plot_axis_type="scatter",
            )

        if normalized_residual_map_real:

            self.plot_line(
                y=np.real(self.fit.residual_map),
                x=self.fit.masked_interferometer.interferometer.uv_distances
                / 10 ** 3.0,
                visuals_1d=self.visuals_1d,
                auto_labels=mp.AutoLabels(
                    title="Normalized Residual Map vs UV-Distance (real)",
                    filename="real_normalized_residual_map_vs_uv_distances",
                    ylabel="V$_{R,data}$ - V$_{R,model}$",
                    xlabel=r"UV$_{distance}$ (k$\lambda$)",
                ),
                plot_axis_type="scatter",
            )
        if normalized_residual_map_imag:
            self.plot_line(
                y=np.imag(self.fit.residual_map),
                x=self.fit.masked_interferometer.interferometer.uv_distances
                / 10 ** 3.0,
                visuals_1d=self.visuals_1d,
                auto_labels=mp.AutoLabels(
                    title="Normalized Residual Map vs UV-Distance (imag)",
                    filename="imag_normalized_residual_map_vs_uv_distances",
                    ylabel="V$_{R,data}$ - V$_{R,model}$",
                    xlabel=r"UV$_{distance}$ (k$\lambda$)",
                ),
                plot_axis_type="scatter",
            )

        if chi_squared_map_real:

            self.plot_line(
                y=np.real(self.fit.residual_map),
                x=self.fit.masked_interferometer.interferometer.uv_distances
                / 10 ** 3.0,
                visuals_1d=self.visuals_1d,
                auto_labels=mp.AutoLabels(
                    title="Chi-Squared Map vs UV-Distance (real)",
                    filename="real_chi_squared_map_vs_uv_distances",
                    ylabel="V$_{R,data}$ - V$_{R,model}$",
                    xlabel=r"UV$_{distance}$ (k$\lambda$)",
                ),
                plot_axis_type="scatter",
            )
        if chi_squared_map_imag:
            self.plot_line(
                y=np.imag(self.fit.residual_map),
                x=self.fit.masked_interferometer.interferometer.uv_distances
                / 10 ** 3.0,
                visuals_1d=self.visuals_1d,
                auto_labels=mp.AutoLabels(
                    title="Chi-Squared Map vs UV-Distance (imag)",
                    filename="imag_chi_squared_map_vs_uv_distances",
                    ylabel="V$_{R,data}$ - V$_{R,model}$",
                    xlabel=r"UV$_{distance}$ (k$\lambda$)",
                ),
                plot_axis_type="scatter",
            )

    def subplot(
        self,
        visibilities=False,
        noise_map=False,
        signal_to_noise_map=False,
        model_visibilities=False,
        residual_map_real=False,
        residual_map_imag=False,
        normalized_residual_map_real=False,
        normalized_residual_map_imag=False,
        chi_squared_map_real=False,
        chi_squared_map_imag=False,
        auto_filename="subplot_fit_interferometer",
    ):

        self._subplot_custom_plot(
            visibilities=visibilities,
            noise_map=noise_map,
            signal_to_noise_map=signal_to_noise_map,
            model_visibilities=model_visibilities,
            residual_map_real=residual_map_real,
            residual_map_imag=residual_map_imag,
            normalized_residual_map_real=normalized_residual_map_real,
            normalized_residual_map_imag=normalized_residual_map_imag,
            chi_squared_map_real=chi_squared_map_real,
            chi_squared_map_imag=chi_squared_map_imag,
            auto_labels=mp.AutoLabels(filename=auto_filename),
        )

    def subplot_fit_interferometer(self):
        return self.subplot(
            residual_map_real=True,
            normalized_residual_map_real=True,
            chi_squared_map_real=True,
            residual_map_imag=True,
            normalized_residual_map_imag=True,
            chi_squared_map_imag=True,
        )


class FitInterferometerPlotter(AbstractFitInterferometerPlotter):
    def __init__(
        self,
        fit: f.FitInterferometer,
        mat_plot_1d: mp.MatPlot1D = mp.MatPlot1D(),
        visuals_1d: vis.Visuals1D = vis.Visuals1D(),
        include_1d: inc.Include1D = inc.Include1D(),
        mat_plot_2d: mp.MatPlot2D = mp.MatPlot2D(),
        visuals_2d: vis.Visuals2D = vis.Visuals2D(),
        include_2d: inc.Include2D = inc.Include2D(),
    ):

        super().__init__(
            fit=fit,
            mat_plot_1d=mat_plot_1d,
            include_1d=include_1d,
            visuals_1d=visuals_1d,
            mat_plot_2d=mat_plot_2d,
            include_2d=include_2d,
            visuals_2d=visuals_2d,
        )
