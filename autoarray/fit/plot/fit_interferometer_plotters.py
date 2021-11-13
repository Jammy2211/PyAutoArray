import numpy as np
from typing import Callable

from autoarray.plot.abstract_plotters import Plotter
from autoarray.plot.mat_wrap.visuals import Visuals1D
from autoarray.plot.mat_wrap.visuals import Visuals2D
from autoarray.plot.mat_wrap.include import Include1D
from autoarray.plot.mat_wrap.include import Include2D
from autoarray.plot.mat_wrap.mat_plot import MatPlot1D
from autoarray.plot.mat_wrap.mat_plot import MatPlot2D
from autoarray.plot.mat_wrap.mat_plot import AutoLabels
from autoarray.fit.fit_dataset import FitInterferometer


class FitInterferometerPlotterMeta(Plotter):
    def __init__(
        self,
        fit,
        get_visuals_2d_real_space: Callable,
        mat_plot_1d: MatPlot1D,
        visuals_1d: Visuals1D,
        include_1d: Include1D,
        mat_plot_2d: MatPlot2D = MatPlot2D(),
        visuals_2d: Visuals2D = Visuals2D(),
        include_2d: Include2D = Include2D(),
    ):
        """
        Plots the attributes of `FitInterferometer` objects using the matplotlib method `imshow()` and many other
        matplotlib functions which customize the plot's appearance.

        The `mat_plot_2d` attribute wraps matplotlib function calls to make the figure. By default, the settings
        passed to every matplotlib function called are those specified in the `config/visualize/mat_wrap/*.ini` files,
        but a user can manually input values into `MatPlot2d` to customize the figure's appearance.

        Overlaid on the figure are visuals, contained in the `Visuals2D` object. Attributes may be extracted from
        the `Array2D` and plotted via the visuals object, if the corresponding entry is `True` in the `Include2D`
        object or the `config/visualize/include.ini` file.

        Parameters
        ----------
        fit
            The fit to an interferometer dataset the plotter plots.
        mat_plot_2d
            Contains objects which wrap the matplotlib function calls that make the plot.
        visuals_2d
            Contains visuals that can be overlaid on the plot.
        include_2d
            Specifies which attributes of the `Array2D` are extracted and plotted as visuals.
        """
        super().__init__(
            mat_plot_1d=mat_plot_1d,
            include_1d=include_1d,
            visuals_1d=visuals_1d,
            mat_plot_2d=mat_plot_2d,
            include_2d=include_2d,
            visuals_2d=visuals_2d,
        )

        self.fit = fit
        self.get_visuals_2d_real_space = get_visuals_2d_real_space

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
        """
        Plots the individual attributes of the plotter's `FitInterferometer` object in 1D and 2D.

        The API is such that every plottable attribute of the `Interferometer` object is an input parameter of type 
        bool of the function, which if switched to `True` means that it is plotted.

        Parameters
        ----------
        visibilities
            Whether or not to make a 2D plot (via `scatter`) of the visibility data.
        noise_map
            Whether or not to make a 2D plot (via `scatter`) of the noise-map.
        signal_to_noise_map
            Whether or not to make a 2D plot (via `scatter`) of the signal-to-noise-map.
        model_visibilities
            Whether or not to make a 2D plot (via `scatter`) of the model visibility data.
        residual_map_real
            Whether or not to make a 1D plot (via `plot`) of the real component of the residual map.
        residual_map_imag
            Whether or not to make a 1D plot (via `plot`) of the imaginary component of the residual map.
        normalized_residual_map_real
            Whether or not to make a 1D plot (via `plot`) of the real component of the normalized residual map.
        normalized_residual_map_imag
            Whether or not to make a 1D plot (via `plot`) of the imaginary component of the normalized residual map.
        chi_squared_map_real
            Whether or not to make a 1D plot (via `plot`) of the real component of the chi-squared map.
        chi_squared_map_imag
            Whether or not to make a 1D plot (via `plot`) of the imaginary component of the chi-squared map.
        dirty_image
            Whether or not to make a 2D plot (via `imshow`) of the dirty image.
        dirty_noise_map
            Whether or not to make a 2D plot (via `imshow`) of the dirty noise map.
        dirty_model_image
            Whether or not to make a 2D plot (via `imshow`) of the dirty model image.
        dirty_residual_map
            Whether or not to make a 2D plot (via `imshow`) of the dirty residual map.
        dirty_normalized_residual_map
            Whether or not to make a 2D plot (via `imshow`) of the dirty normalized residual map.
        dirty_chi_squared_map
            Whether or not to make a 2D plot (via `imshow`) of the dirty chi-squared map.
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
                visuals_2d=self.get_visuals_2d_real_space(),
                auto_labels=AutoLabels(title="Dirty Image", filename="dirty_image_2d"),
            )

        if dirty_noise_map:

            self.mat_plot_2d.plot_array(
                array=self.fit.dirty_noise_map,
                visuals_2d=self.get_visuals_2d_real_space(),
                auto_labels=AutoLabels(
                    title="Dirty Noise Map", filename="dirty_noise_map_2d"
                ),
            )

        if dirty_signal_to_noise_map:

            self.mat_plot_2d.plot_array(
                array=self.fit.dirty_signal_to_noise_map,
                visuals_2d=self.get_visuals_2d_real_space(),
                auto_labels=AutoLabels(
                    title="Dirty Signal-To-Noise Map",
                    filename="dirty_signal_to_noise_map_2d",
                ),
            )

        if dirty_model_image:

            self.mat_plot_2d.plot_array(
                array=self.fit.dirty_model_image,
                visuals_2d=self.get_visuals_2d_real_space(),
                auto_labels=AutoLabels(
                    title="Dirty Model Image", filename="dirty_model_image_2d"
                ),
            )

        if dirty_residual_map:

            self.mat_plot_2d.plot_array(
                array=self.fit.dirty_residual_map,
                visuals_2d=self.get_visuals_2d_real_space(),
                auto_labels=AutoLabels(
                    title="Dirty Residual Map", filename="dirty_residual_map_2d"
                ),
            )

        if dirty_normalized_residual_map:

            self.mat_plot_2d.plot_array(
                array=self.fit.dirty_normalized_residual_map,
                visuals_2d=self.get_visuals_2d_real_space(),
                auto_labels=AutoLabels(
                    title="Dirty Normalized Residual Map",
                    filename="dirty_normalized_residual_map_2d",
                ),
            )

        if dirty_chi_squared_map:

            self.mat_plot_2d.plot_array(
                array=self.fit.dirty_chi_squared_map,
                visuals_2d=self.get_visuals_2d_real_space(),
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


class FitInterferometerPlotter(Plotter):
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
        """
        Plots the attributes of `FitInterferometer` objects using the matplotlib method `imshow()` and many other
        matplotlib functions which customize the plot's appearance.

        The `mat_plot_2d` attribute wraps matplotlib function calls to make the figure. By default, the settings
        passed to every matplotlib function called are those specified in the `config/visualize/mat_wrap/*.ini` files,
        but a user can manually input values into `MatPlot2d` to customize the figure's appearance.

        Overlaid on the figure are visuals, contained in the `Visuals2D` object. Attributes may be extracted from
        the `Array2D` and plotted via the visuals object, if the corresponding entry is `True` in the `Include2D`
        object or the `config/visualize/include.ini` file.

        Parameters
        ----------
        fit
            The fit to an interferometer dataset the plotter plots.
        mat_plot_2d
            Contains objects which wrap the matplotlib function calls that make the plot.
        visuals_2d
            Contains visuals that can be overlaid on the plot.
        include_2d
            Specifies which attributes of the `Array2D` are extracted and plotted as visuals.
        """
        super().__init__(
            mat_plot_1d=mat_plot_1d,
            include_1d=include_1d,
            visuals_1d=visuals_1d,
            mat_plot_2d=mat_plot_2d,
            include_2d=include_2d,
            visuals_2d=visuals_2d,
        )

        self.fit = fit

        self._fit_interferometer_meta_plotter = FitInterferometerPlotterMeta(
            fit=self.fit,
            get_visuals_2d_real_space=self.get_visuals_2d_real_space,
            mat_plot_1d=self.mat_plot_1d,
            include_1d=self.include_1d,
            visuals_1d=self.visuals_1d,
            mat_plot_2d=self.mat_plot_2d,
            include_2d=self.include_2d,
            visuals_2d=self.visuals_2d,
        )

        self.figures_2d = self._fit_interferometer_meta_plotter.figures_2d
        self.subplot = self._fit_interferometer_meta_plotter.subplot
        self.subplot_fit_interferometer = (
            self._fit_interferometer_meta_plotter.subplot_fit_interferometer
        )
        self.subplot_fit_dirty_images = (
            self._fit_interferometer_meta_plotter.subplot_fit_dirty_images
        )

    def get_visuals_2d_real_space(self) -> Visuals2D:
        return self.get_2d.via_mask_from(mask=self.fit.interferometer.real_space_mask)
