import numpy as np

from autoarray.plot.abstract_plotters import AbstractPlotter
from autoarray.plot.mat_wrap.visuals import Visuals1D
from autoarray.plot.mat_wrap.visuals import Visuals2D
from autoarray.plot.mat_wrap.include import Include1D
from autoarray.plot.mat_wrap.include import Include2D
from autoarray.plot.mat_wrap.mat_plot import MatPlot1D
from autoarray.plot.mat_wrap.mat_plot import MatPlot2D
from autoarray.plot.mat_wrap.mat_plot import AutoLabels
from autoarray.fit.fit_dataset import FitInterferometer


class AbstractFitInterferometerPlotter(AbstractPlotter):
    def __init__(
        self,
        fit: FitInterferometer,
        mat_plot_1d: MatPlot1D,
        visuals_1d: Visuals1D,
        include_1d: Include1D,
        mat_plot_2d: MatPlot2D,
        visuals_2d: Visuals2D,
        include_2d: Include2D,
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

    def figures_2d(
        self,
        visibilities: bool = False,
        noise_map: bool = False,
        signal_to_noise_map: bool = False,
        model_visibilities: bool = False,
        residual_map_real: bool = False,
        residual_map_imag: bool = False,
        normalized_residual_map_real: bool = False,
        normalized_residual_map_imag: bool = False,
        chi_squared_map_real: bool = False,
        chi_squared_map_imag: bool = False,
        dirty_image: bool = False,
        dirty_noise_map: bool = False,
        dirty_signal_to_noise_map: bool = False,
        dirty_model_image: bool = False,
        dirty_residual_map: bool = False,
        dirty_normalized_residual_map: bool = False,
        dirty_chi_squared_map: bool = False,
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
            How the data is output. File formats (e.g. png, fits) output the data to harddisk. 'show' displays the data \
            in the python interpreter window.
        """

        if visibilities:
            self.mat_plot_2d.plot_grid(
                grid=self.fit.visibilities.in_grid,
                visuals_2d=self.visuals_2d,
                auto_labels=AutoLabels(title="Visibilities", filename="visibilities"),
                color_array=np.real(self.fit.noise_map),
            )

        if noise_map:
            self.mat_plot_2d.plot_grid(
                grid=self.fit.visibilities.in_grid,
                visuals_2d=self.visuals_2d,
                auto_labels=AutoLabels(title="Noise-Map", filename="noise_map"),
                color_array=np.real(self.fit.noise_map),
            )

        if signal_to_noise_map:
            self.mat_plot_2d.plot_grid(
                grid=self.fit.visibilities.in_grid,
                visuals_2d=self.visuals_2d,
                auto_labels=AutoLabels(
                    title="Signal-To-Noise Map", filename="signal_to_noise_map"
                ),
                color_array=np.real(self.fit.signal_to_noise_map),
            )
        if model_visibilities:
            self.mat_plot_2d.plot_grid(
                grid=self.fit.visibilities.in_grid,
                visuals_2d=self.visuals_2d,
                auto_labels=AutoLabels(
                    title="Model Visibilities", filename="model_visibilities"
                ),
                color_array=np.real(self.fit.model_data),
            )

        if residual_map_real:
            self.mat_plot_1d.plot_yx(
                y=np.real(self.fit.residual_map),
                x=self.fit.interferometer.uv_distances / 10 ** 3.0,
                visuals_1d=self.visuals_1d,
                auto_labels=AutoLabels(
                    title="Residual Map vs UV-Distance (real)",
                    filename="real_residual_map_vs_uv_distances",
                    ylabel="V$_{R,data}$ - V$_{R,model}$",
                    xlabel=r"UV$_{distance}$ (k$\lambda$)",
                ),
                plot_axis_type_override="scatter",
            )
        if residual_map_imag:
            self.mat_plot_1d.plot_yx(
                y=np.imag(self.fit.residual_map),
                x=self.fit.interferometer.uv_distances / 10 ** 3.0,
                visuals_1d=self.visuals_1d,
                auto_labels=AutoLabels(
                    title="Residual Map vs UV-Distance (imag)",
                    filename="imag_residual_map_vs_uv_distances",
                    ylabel="V$_{R,data}$ - V$_{R,model}$",
                    xlabel=r"UV$_{distance}$ (k$\lambda$)",
                ),
                plot_axis_type_override="scatter",
            )

        if normalized_residual_map_real:

            self.mat_plot_1d.plot_yx(
                y=np.real(self.fit.residual_map),
                x=self.fit.interferometer.uv_distances / 10 ** 3.0,
                visuals_1d=self.visuals_1d,
                auto_labels=AutoLabels(
                    title="Normalized Residual Map vs UV-Distance (real)",
                    filename="real_normalized_residual_map_vs_uv_distances",
                    ylabel="V$_{R,data}$ - V$_{R,model}$",
                    xlabel=r"UV$_{distance}$ (k$\lambda$)",
                ),
                plot_axis_type_override="scatter",
            )
        if normalized_residual_map_imag:
            self.mat_plot_1d.plot_yx(
                y=np.imag(self.fit.residual_map),
                x=self.fit.interferometer.uv_distances / 10 ** 3.0,
                visuals_1d=self.visuals_1d,
                auto_labels=AutoLabels(
                    title="Normalized Residual Map vs UV-Distance (imag)",
                    filename="imag_normalized_residual_map_vs_uv_distances",
                    ylabel="V$_{R,data}$ - V$_{R,model}$",
                    xlabel=r"UV$_{distance}$ (k$\lambda$)",
                ),
                plot_axis_type_override="scatter",
            )

        if chi_squared_map_real:

            self.mat_plot_1d.plot_yx(
                y=np.real(self.fit.residual_map),
                x=self.fit.interferometer.uv_distances / 10 ** 3.0,
                visuals_1d=self.visuals_1d,
                auto_labels=AutoLabels(
                    title="Chi-Squared Map vs UV-Distance (real)",
                    filename="real_chi_squared_map_vs_uv_distances",
                    ylabel="V$_{R,data}$ - V$_{R,model}$",
                    xlabel=r"UV$_{distance}$ (k$\lambda$)",
                ),
                plot_axis_type_override="scatter",
            )
        if chi_squared_map_imag:
            self.mat_plot_1d.plot_yx(
                y=np.imag(self.fit.residual_map),
                x=self.fit.interferometer.uv_distances / 10 ** 3.0,
                visuals_1d=self.visuals_1d,
                auto_labels=AutoLabels(
                    title="Chi-Squared Map vs UV-Distance (imag)",
                    filename="imag_chi_squared_map_vs_uv_distances",
                    ylabel="V$_{R,data}$ - V$_{R,model}$",
                    xlabel=r"UV$_{distance}$ (k$\lambda$)",
                ),
                plot_axis_type_override="scatter",
            )

        if dirty_image:

            self.mat_plot_2d.plot_array(
                array=self.fit.dirty_image,
                visuals_2d=self.extractor_2d.via_mask_from(
                    mask=self.fit.interferometer.real_space_mask
                ),
                auto_labels=AutoLabels(title="Dirty Image", filename="dirty_image_2d"),
            )

        if dirty_noise_map:

            self.mat_plot_2d.plot_array(
                array=self.fit.dirty_noise_map,
                visuals_2d=self.extractor_2d.via_mask_from(
                    mask=self.fit.interferometer.real_space_mask
                ),
                auto_labels=AutoLabels(
                    title="Dirty Noise Map", filename="dirty_noise_map_2d"
                ),
            )

        if dirty_signal_to_noise_map:

            self.mat_plot_2d.plot_array(
                array=self.fit.dirty_signal_to_noise_map,
                visuals_2d=self.extractor_2d.via_mask_from(
                    mask=self.fit.interferometer.real_space_mask
                ),
                auto_labels=AutoLabels(
                    title="Dirty Signal-To-Noise Map",
                    filename="dirty_signal_to_noise_map_2d",
                ),
            )

        if dirty_model_image:

            self.mat_plot_2d.plot_array(
                array=self.fit.dirty_model_image,
                visuals_2d=self.extractor_2d.via_mask_from(
                    mask=self.fit.interferometer.real_space_mask
                ),
                auto_labels=AutoLabels(
                    title="Dirty Model Image", filename="dirty_model_image_2d"
                ),
            )

        if dirty_residual_map:

            self.mat_plot_2d.plot_array(
                array=self.fit.dirty_residual_map,
                visuals_2d=self.extractor_2d.via_mask_from(
                    mask=self.fit.interferometer.real_space_mask
                ),
                auto_labels=AutoLabels(
                    title="Dirty Residual Map", filename="dirty_residual_map_2d"
                ),
            )

        if dirty_normalized_residual_map:

            self.mat_plot_2d.plot_array(
                array=self.fit.dirty_normalized_residual_map,
                visuals_2d=self.extractor_2d.via_mask_from(
                    mask=self.fit.interferometer.real_space_mask
                ),
                auto_labels=AutoLabels(
                    title="Dirty Normalized Residual Map",
                    filename="dirty_normalized_residual_map_2d",
                ),
            )

        if dirty_chi_squared_map:

            self.mat_plot_2d.plot_array(
                array=self.fit.dirty_chi_squared_map,
                visuals_2d=self.extractor_2d.via_mask_from(
                    mask=self.fit.interferometer.real_space_mask
                ),
                auto_labels=AutoLabels(
                    title="Dirty Chi-Squared Map", filename="dirty_chi_squared_map_2d"
                ),
            )

    def subplot(
        self,
        visibilities: bool = False,
        noise_map: bool = False,
        signal_to_noise_map: bool = False,
        model_visibilities: bool = False,
        residual_map_real: bool = False,
        residual_map_imag: bool = False,
        normalized_residual_map_real: bool = False,
        normalized_residual_map_imag: bool = False,
        chi_squared_map_real: bool = False,
        chi_squared_map_imag: bool = False,
        dirty_image: bool = False,
        dirty_noise_map: bool = False,
        dirty_signal_to_noise_map: bool = False,
        dirty_model_image: bool = False,
        dirty_residual_map: bool = False,
        dirty_normalized_residual_map: bool = False,
        dirty_chi_squared_map: bool = False,
        auto_filename: str = "subplot_fit_interferometer",
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
            dirty_image=dirty_image,
            dirty_noise_map=dirty_noise_map,
            dirty_signal_to_noise_map=dirty_signal_to_noise_map,
            dirty_model_image=dirty_model_image,
            dirty_residual_map=dirty_residual_map,
            dirty_normalized_residual_map=dirty_normalized_residual_map,
            dirty_chi_squared_map=dirty_chi_squared_map,
            auto_labels=AutoLabels(filename=auto_filename),
        )

    def subplot_fit_interferometer(self):
        return self.subplot(
            residual_map_real=True,
            normalized_residual_map_real=True,
            chi_squared_map_real=True,
            residual_map_imag=True,
            normalized_residual_map_imag=True,
            chi_squared_map_imag=True,
            auto_filename="subplot_fit_interferometer",
        )

    def subplot_fit_dirty_images(self):

        return self.subplot(
            dirty_image=True,
            dirty_signal_to_noise_map=True,
            dirty_model_image=True,
            dirty_residual_map=True,
            dirty_normalized_residual_map=True,
            dirty_chi_squared_map=True,
            auto_filename="subplot_fit_dirty_images",
        )


class FitInterferometerPlotter(AbstractFitInterferometerPlotter):
    def __init__(
        self,
        fit: FitInterferometer,
        mat_plot_1d: MatPlot1D = MatPlot1D(),
        visuals_1d: Visuals1D = Visuals1D(),
        include_1d: Include1D = Include1D(),
        mat_plot_2d: MatPlot2D = MatPlot2D(),
        visuals_2d: Visuals2D = Visuals2D(),
        include_2d: Include2D = Include2D(),
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
