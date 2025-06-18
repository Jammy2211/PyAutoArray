import numpy as np
from typing import Callable

from autoarray.plot.abstract_plotters import Plotter
from autoarray.plot.visuals.one_d import Visuals1D
from autoarray.plot.visuals.two_d import Visuals2D
from autoarray.plot.include.one_d import Include1D
from autoarray.plot.include.two_d import Include2D
from autoarray.plot.mat_plot.one_d import MatPlot1D
from autoarray.plot.mat_plot.two_d import MatPlot2D
from autoarray.plot.auto_labels import AutoLabels
from autoarray.fit.fit_interferometer import FitInterferometer


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
        residuals_symmetric_cmap: bool = True,
    ):
        """
        Plots the attributes of `FitInterferometer` objects using the matplotlib method `imshow()` and many
        other matplotlib functions which customize the plot's appearance.

        The `mat_plot_1d` and `mat_plot_2d` attributes wrap matplotlib function calls to make the figure. By default,
        the settings passed to every matplotlib function called are those specified in
        the `config/visualize/mat_wrap/*.ini` files, but a user can manually input values into `MatPlot2d` to
        customize the figure's appearance.

        Overlaid on the figure are visuals, contained in the `Visuals1D` and `Visuals2D` objects. Attributes may be
        extracted from the `FitInterferometer` and plotted via the visuals object, if the corresponding entry is `True` in
        the `Include1D` or `Include2D` object or the `config/visualize/include.ini` file.

        Parameters
        ----------
        fit
            The fit to an interferometer dataset the plotter plots.
        get_visuals_2d
            A function which extracts from the `FitInterferometer` the 2D visuals which are plotted on figures.
        mat_plot_1d
            Contains objects which wrap the matplotlib function calls that make 1D plots.
        visuals_1d
            Contains 1D visuals that can be overlaid on 1D plots.
        include_1d
            Specifies which attributes of the `FitInterferometer` are extracted and plotted as visuals for 1D plots.
        mat_plot_2d
            Contains objects which wrap the matplotlib function calls that make 2D plots.
        visuals_2d
            Contains 2D visuals that can be overlaid on 2D plots.
        include_2d
            Specifies which attributes of the `FitInterferometer` are extracted and plotted as visuals for 2D plots.
        residuals_symmetric_cmap
            If true, the `residual_map` and `normalized_residual_map` are plotted with a symmetric color map such
            that `abs(vmin) = abs(vmax)`.
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
        self.residuals_symmetric_cmap = residuals_symmetric_cmap

    def figures_2d(
        self,
        data: bool = False,
        noise_map: bool = False,
        signal_to_noise_map: bool = False,
        amplitudes_vs_uv_distances: bool = False,
        model_data: bool = False,
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
        data
            Whether to make a 2D plot (via `scatter`) of the visibility data.
        noise_map
            Whether to make a 2D plot (via `scatter`) of the noise-map.
        signal_to_noise_map
            Whether to make a 2D plot (via `scatter`) of the signal-to-noise-map.
        model_data
            Whether to make a 2D plot (via `scatter`) of the model visibility data.
        residual_map_real
            Whether to make a 1D plot (via `plot`) of the real component of the residual map.
        residual_map_imag
            Whether to make a 1D plot (via `plot`) of the imaginary component of the residual map.
        normalized_residual_map_real
            Whether to make a 1D plot (via `plot`) of the real component of the normalized residual map.
        normalized_residual_map_imag
            Whether to make a 1D plot (via `plot`) of the imaginary component of the normalized residual map.
        chi_squared_map_real
            Whether to make a 1D plot (via `plot`) of the real component of the chi-squared map.
        chi_squared_map_imag
            Whether to make a 1D plot (via `plot`) of the imaginary component of the chi-squared map.
        dirty_image
            Whether to make a 2D plot (via `imshow`) of the dirty image.
        dirty_noise_map
            Whether to make a 2D plot (via `imshow`) of the dirty noise map.
        dirty_model_image
            Whether to make a 2D plot (via `imshow`) of the dirty model image.
        dirty_residual_map
            Whether to make a 2D plot (via `imshow`) of the dirty residual map.
        dirty_normalized_residual_map
            Whether to make a 2D plot (via `imshow`) of the dirty normalized residual map.
        dirty_chi_squared_map
            Whether to make a 2D plot (via `imshow`) of the dirty chi-squared map.
        """

        if data:
            self.mat_plot_2d.plot_grid(
                grid=self.fit.data.in_grid,
                visuals_2d=self.visuals_2d,
                auto_labels=AutoLabels(title="Visibilities", filename="data"),
                color_array=np.real(self.fit.noise_map),
            )

        if noise_map:
            self.mat_plot_2d.plot_grid(
                grid=self.fit.data.in_grid,
                visuals_2d=self.visuals_2d,
                auto_labels=AutoLabels(title="Noise-Map", filename="noise_map"),
                color_array=np.real(self.fit.noise_map),
            )

        if signal_to_noise_map:
            self.mat_plot_2d.plot_grid(
                grid=self.fit.data.in_grid,
                visuals_2d=self.visuals_2d,
                auto_labels=AutoLabels(
                    title="Signal-To-Noise Map", filename="signal_to_noise_map"
                ),
                color_array=np.real(self.fit.signal_to_noise_map),
            )

        if amplitudes_vs_uv_distances:
            self.mat_plot_1d.plot_yx(
                y=self.fit.dataset.amplitudes,
                x=self.fit.dataset.uv_distances / 10**3.0,
                visuals_1d=self.visuals_1d,
                auto_labels=AutoLabels(
                    title="Amplitudes vs UV-distances",
                    filename="amplitudes_vs_uv_distances",
                    yunit="Jy",
                    xunit="k$\lambda$",
                ),
                plot_axis_type_override="scatter",
            )

        if model_data:
            self.mat_plot_2d.plot_grid(
                grid=self.fit.data.in_grid,
                visuals_2d=self.visuals_2d,
                auto_labels=AutoLabels(
                    title="Model Visibilities", filename="model_data"
                ),
                color_array=np.real(self.fit.model_data.array),
            )

        if residual_map_real:
            self.mat_plot_1d.plot_yx(
                y=np.real(self.fit.residual_map),
                x=self.fit.dataset.uv_distances / 10**3.0,
                visuals_1d=self.visuals_1d,
                auto_labels=AutoLabels(
                    title="Residual vs UV-Distance (real)",
                    filename="real_residual_map_vs_uv_distances",
                    xunit="k$\lambda$",
                ),
                plot_axis_type_override="scatter",
            )
        if residual_map_imag:
            self.mat_plot_1d.plot_yx(
                y=np.imag(self.fit.residual_map),
                x=self.fit.dataset.uv_distances / 10**3.0,
                visuals_1d=self.visuals_1d,
                auto_labels=AutoLabels(
                    title="Residual vs UV-Distance (imag)",
                    filename="imag_residual_map_vs_uv_distances",
                    xunit="k$\lambda$",
                ),
                plot_axis_type_override="scatter",
            )

        if normalized_residual_map_real:
            self.mat_plot_1d.plot_yx(
                y=np.real(self.fit.normalized_residual_map),
                x=self.fit.dataset.uv_distances / 10**3.0,
                visuals_1d=self.visuals_1d,
                auto_labels=AutoLabels(
                    title="Norm Residual vs UV-Distance (real)",
                    filename="real_normalized_residual_map_vs_uv_distances",
                    yunit="$\sigma$",
                    xunit="k$\lambda$",
                ),
                plot_axis_type_override="scatter",
            )
        if normalized_residual_map_imag:
            self.mat_plot_1d.plot_yx(
                y=np.imag(self.fit.normalized_residual_map),
                x=self.fit.dataset.uv_distances / 10**3.0,
                visuals_1d=self.visuals_1d,
                auto_labels=AutoLabels(
                    title="Norm Residual vs UV-Distance (imag)",
                    filename="imag_normalized_residual_map_vs_uv_distances",
                    yunit="$\sigma$",
                    xunit="k$\lambda$",
                ),
                plot_axis_type_override="scatter",
            )

        if chi_squared_map_real:
            self.mat_plot_1d.plot_yx(
                y=np.real(self.fit.chi_squared_map),
                x=self.fit.dataset.uv_distances / 10**3.0,
                visuals_1d=self.visuals_1d,
                auto_labels=AutoLabels(
                    title="Chi-Squared vs UV-Distance (real)",
                    filename="real_chi_squared_map_vs_uv_distances",
                    yunit="$\chi^2$",
                    xunit="k$\lambda$",
                ),
                plot_axis_type_override="scatter",
            )
        if chi_squared_map_imag:
            self.mat_plot_1d.plot_yx(
                y=np.imag(self.fit.chi_squared_map),
                x=self.fit.dataset.uv_distances / 10**3.0,
                visuals_1d=self.visuals_1d,
                auto_labels=AutoLabels(
                    title="Chi-Squared vs UV-Distance (imag)",
                    filename="imag_chi_squared_map_vs_uv_distances",
                    yunit="$\chi^2$",
                    xunit="k$\lambda$",
                ),
                plot_axis_type_override="scatter",
            )

        if dirty_image:
            self.mat_plot_2d.plot_array(
                array=self.fit.dirty_image,
                visuals_2d=self.get_visuals_2d_real_space(),
                auto_labels=AutoLabels(title="Dirty Image", filename="dirty_image"),
            )

        if dirty_noise_map:
            self.mat_plot_2d.plot_array(
                array=self.fit.dirty_noise_map,
                visuals_2d=self.get_visuals_2d_real_space(),
                auto_labels=AutoLabels(
                    title="Dirty Noise Map", filename="dirty_noise_map"
                ),
            )

        if dirty_signal_to_noise_map:
            self.mat_plot_2d.plot_array(
                array=self.fit.dirty_signal_to_noise_map,
                visuals_2d=self.get_visuals_2d_real_space(),
                auto_labels=AutoLabels(
                    title="Dirty Signal-To-Noise Map",
                    filename="dirty_signal_to_noise_map",
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

        cmap_original = self.mat_plot_2d.cmap

        if self.residuals_symmetric_cmap:
            self.mat_plot_2d.cmap = self.mat_plot_2d.cmap.symmetric_cmap_from()

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

        if self.residuals_symmetric_cmap:
            self.mat_plot_2d.cmap = cmap_original

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
        data: bool = False,
        noise_map: bool = False,
        signal_to_noise_map: bool = False,
        model_data: bool = False,
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
        auto_filename: str = "subplot_fit",
    ):
        """
        Plots the individual attributes of the plotter's `FitInterferometer` object in 1D and 2D on a subplot.

        The API is such that every plottable attribute of the `Interferometer` object is an input parameter of type
        bool of the function, which if switched to `True` means that it is included on the subplot.

        Parameters
        ----------
        data
            Whether to make a 2D plot (via `scatter`) of the visibility data.
        noise_map
            Whether to make a 2D plot (via `scatter`) of the noise-map.
        signal_to_noise_map
            Whether to make a 2D plot (via `scatter`) of the signal-to-noise-map.
        model_data
            Whether to make a 2D plot (via `scatter`) of the model visibility data.
        residual_map_real
            Whether to make a 1D plot (via `plot`) of the real component of the residual map.
        residual_map_imag
            Whether to make a 1D plot (via `plot`) of the imaginary component of the residual map.
        normalized_residual_map_real
            Whether to make a 1D plot (via `plot`) of the real component of the normalized residual map.
        normalized_residual_map_imag
            Whether to make a 1D plot (via `plot`) of the imaginary component of the normalized residual map.
        chi_squared_map_real
            Whether to make a 1D plot (via `plot`) of the real component of the chi-squared map.
        chi_squared_map_imag
            Whether to make a 1D plot (via `plot`) of the imaginary component of the chi-squared map.
        dirty_image
            Whether to make a 2D plot (via `imshow`) of the dirty image.
        dirty_noise_map
            Whether to make a 2D plot (via `imshow`) of the dirty noise map.
        dirty_model_image
            Whether to make a 2D plot (via `imshow`) of the dirty model image.
        dirty_residual_map
            Whether to make a 2D plot (via `imshow`) of the dirty residual map.
        dirty_normalized_residual_map
            Whether to make a 2D plot (via `imshow`) of the dirty normalized residual map.
        dirty_chi_squared_map
            Whether to make a 2D plot (via `imshow`) of the dirty chi-squared map.
        auto_filename
            The default filename of the output subplot if written to hard-disk.
        """

        self._subplot_custom_plot(
            visibilities=data,
            noise_map=noise_map,
            signal_to_noise_map=signal_to_noise_map,
            model_data=model_data,
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

    def subplot_fit(self):
        """
        Standard subplot of the attributes of the plotter's `FitInterferometer` object.
        """
        return self.subplot(
            residual_map_real=True,
            normalized_residual_map_real=True,
            chi_squared_map_real=True,
            residual_map_imag=True,
            normalized_residual_map_imag=True,
            chi_squared_map_imag=True,
            auto_filename="subplot_fit",
        )

    def subplot_fit_dirty_images(self):
        """
        Standard subplot of the dirty attributes of the plotter's `FitInterferometer` object.
        """
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
        the `FitInterferometer` and plotted via the visuals object, if the corresponding entry is `True` in the `Include2D`
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
        self.subplot_fit = self._fit_interferometer_meta_plotter.subplot_fit
        self.subplot_fit_dirty_images = (
            self._fit_interferometer_meta_plotter.subplot_fit_dirty_images
        )

    def get_visuals_2d_real_space(self) -> Visuals2D:
        return self.get_2d.via_mask_from(mask=self.fit.dataset.real_space_mask)
