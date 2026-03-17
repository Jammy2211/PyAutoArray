import numpy as np

from autoarray.plot.abstract_plotters import AbstractPlotter
from autoarray.plot.mat_plot.one_d import MatPlot1D
from autoarray.plot.mat_plot.two_d import MatPlot2D
from autoarray.plot.auto_labels import AutoLabels
from autoarray.plot.plots.array import plot_array
from autoarray.plot.plots.grid import plot_grid
from autoarray.plot.plots.yx import plot_yx
from autoarray.fit.fit_interferometer import FitInterferometer
from autoarray.structures.plot.structure_plotters import (
    _auto_mask_edge,
    _output_for_mat_plot,
    _zoom_array,
)


class FitInterferometerPlotterMeta(AbstractPlotter):
    def __init__(
        self,
        fit,
        mat_plot_1d: MatPlot1D = None,
        mat_plot_2d: MatPlot2D = None,
        residuals_symmetric_cmap: bool = True,
    ):
        super().__init__(mat_plot_1d=mat_plot_1d, mat_plot_2d=mat_plot_2d)
        self.fit = fit
        self.residuals_symmetric_cmap = residuals_symmetric_cmap

    def _plot_array(self, array, auto_filename: str, title: str):
        is_sub = self.mat_plot_2d.is_for_subplot
        ax = self.mat_plot_2d.setup_subplot() if is_sub else None
        output_path, filename, fmt = _output_for_mat_plot(
            self.mat_plot_2d, is_sub, auto_filename
        )
        array = _zoom_array(array)
        try:
            arr = array.native.array
            extent = array.geometry.extent
        except AttributeError:
            arr = np.asarray(array)
            extent = None
        plot_array(
            array=arr,
            ax=ax,
            extent=extent,
            mask=_auto_mask_edge(array) if hasattr(array, "mask") else None,
            title=title,
            colormap=self.mat_plot_2d.cmap.cmap,
            use_log10=self.mat_plot_2d.use_log10,
            output_path=output_path,
            output_filename=filename,
            output_format=fmt,
            structure=array,
        )

    def _plot_grid(self, grid, auto_filename: str, title: str, color_array=None):
        is_sub = self.mat_plot_2d.is_for_subplot
        ax = self.mat_plot_2d.setup_subplot() if is_sub else None
        output_path, filename, fmt = _output_for_mat_plot(
            self.mat_plot_2d, is_sub, auto_filename
        )
        plot_grid(
            grid=np.array(grid.array),
            ax=ax,
            color_array=color_array,
            title=title,
            output_path=output_path,
            output_filename=filename,
            output_format=fmt,
        )

    def _plot_yx(self, y, x, auto_filename: str, title: str, ylabel: str = "",
                 xlabel: str = "", plot_axis_type: str = "linear"):
        is_sub = self.mat_plot_1d.is_for_subplot
        ax = self.mat_plot_1d.setup_subplot() if is_sub else None
        output_path, filename, fmt = _output_for_mat_plot(
            self.mat_plot_1d, is_sub, auto_filename
        )
        plot_yx(
            y=np.asarray(y),
            x=np.asarray(x) if x is not None else None,
            ax=ax,
            title=title,
            ylabel=ylabel,
            xlabel=xlabel,
            plot_axis_type=plot_axis_type,
            output_path=output_path,
            output_filename=filename,
            output_format=fmt,
        )

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
        if data:
            self._plot_grid(
                grid=self.fit.data.in_grid,
                auto_filename="data",
                title="Visibilities",
                color_array=np.real(self.fit.noise_map),
            )
        if noise_map:
            self._plot_grid(
                grid=self.fit.data.in_grid,
                auto_filename="noise_map",
                title="Noise-Map",
                color_array=np.real(self.fit.noise_map),
            )
        if signal_to_noise_map:
            self._plot_grid(
                grid=self.fit.data.in_grid,
                auto_filename="signal_to_noise_map",
                title="Signal-To-Noise Map",
                color_array=np.real(self.fit.signal_to_noise_map),
            )
        if amplitudes_vs_uv_distances:
            self._plot_yx(
                y=self.fit.dataset.amplitudes,
                x=self.fit.dataset.uv_distances / 10**3.0,
                auto_filename="amplitudes_vs_uv_distances",
                title="Amplitudes vs UV-distances",
                ylabel="Jy",
                xlabel="k$\\lambda$",
                plot_axis_type="scatter",
            )
        if model_data:
            self._plot_grid(
                grid=self.fit.data.in_grid,
                auto_filename="model_data",
                title="Model Visibilities",
                color_array=np.real(self.fit.model_data.array),
            )
        if residual_map_real:
            self._plot_yx(
                y=np.real(self.fit.residual_map),
                x=self.fit.dataset.uv_distances / 10**3.0,
                auto_filename="real_residual_map_vs_uv_distances",
                title="Residual vs UV-Distance (real)",
                xlabel="k$\\lambda$",
                plot_axis_type="scatter",
            )
        if residual_map_imag:
            self._plot_yx(
                y=np.imag(self.fit.residual_map),
                x=self.fit.dataset.uv_distances / 10**3.0,
                auto_filename="imag_residual_map_vs_uv_distances",
                title="Residual vs UV-Distance (imag)",
                xlabel="k$\\lambda$",
                plot_axis_type="scatter",
            )
        if normalized_residual_map_real:
            self._plot_yx(
                y=np.real(self.fit.normalized_residual_map),
                x=self.fit.dataset.uv_distances / 10**3.0,
                auto_filename="real_normalized_residual_map_vs_uv_distances",
                title="Norm Residual vs UV-Distance (real)",
                ylabel="$\\sigma$",
                xlabel="k$\\lambda$",
                plot_axis_type="scatter",
            )
        if normalized_residual_map_imag:
            self._plot_yx(
                y=np.imag(self.fit.normalized_residual_map),
                x=self.fit.dataset.uv_distances / 10**3.0,
                auto_filename="imag_normalized_residual_map_vs_uv_distances",
                title="Norm Residual vs UV-Distance (imag)",
                ylabel="$\\sigma$",
                xlabel="k$\\lambda$",
                plot_axis_type="scatter",
            )
        if chi_squared_map_real:
            self._plot_yx(
                y=np.real(self.fit.chi_squared_map),
                x=self.fit.dataset.uv_distances / 10**3.0,
                auto_filename="real_chi_squared_map_vs_uv_distances",
                title="Chi-Squared vs UV-Distance (real)",
                ylabel="$\\chi^2$",
                xlabel="k$\\lambda$",
                plot_axis_type="scatter",
            )
        if chi_squared_map_imag:
            self._plot_yx(
                y=np.imag(self.fit.chi_squared_map),
                x=self.fit.dataset.uv_distances / 10**3.0,
                auto_filename="imag_chi_squared_map_vs_uv_distances",
                title="Chi-Squared vs UV-Distance (imag)",
                ylabel="$\\chi^2$",
                xlabel="k$\\lambda$",
                plot_axis_type="scatter",
            )
        if dirty_image:
            self._plot_array(
                array=self.fit.dirty_image,
                auto_filename="dirty_image",
                title="Dirty Image",
            )
        if dirty_noise_map:
            self._plot_array(
                array=self.fit.dirty_noise_map,
                auto_filename="dirty_noise_map",
                title="Dirty Noise Map",
            )
        if dirty_signal_to_noise_map:
            self._plot_array(
                array=self.fit.dirty_signal_to_noise_map,
                auto_filename="dirty_signal_to_noise_map",
                title="Dirty Signal-To-Noise Map",
            )
        if dirty_model_image:
            self._plot_array(
                array=self.fit.dirty_model_image,
                auto_filename="dirty_model_image_2d",
                title="Dirty Model Image",
            )

        cmap_original = self.mat_plot_2d.cmap
        if self.residuals_symmetric_cmap:
            self.mat_plot_2d.cmap = self.mat_plot_2d.cmap.symmetric_cmap_from()

        if dirty_residual_map:
            self._plot_array(
                array=self.fit.dirty_residual_map,
                auto_filename="dirty_residual_map_2d",
                title="Dirty Residual Map",
            )
        if dirty_normalized_residual_map:
            self._plot_array(
                array=self.fit.dirty_normalized_residual_map,
                auto_filename="dirty_normalized_residual_map_2d",
                title="Dirty Normalized Residual Map",
            )

        if self.residuals_symmetric_cmap:
            self.mat_plot_2d.cmap = cmap_original

        if dirty_chi_squared_map:
            self._plot_array(
                array=self.fit.dirty_chi_squared_map,
                auto_filename="dirty_chi_squared_map_2d",
                title="Dirty Chi-Squared Map",
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
        return self.subplot(
            dirty_image=True,
            dirty_signal_to_noise_map=True,
            dirty_model_image=True,
            dirty_residual_map=True,
            dirty_normalized_residual_map=True,
            dirty_chi_squared_map=True,
            auto_filename="subplot_fit_dirty_images",
        )


class FitInterferometerPlotter(AbstractPlotter):
    def __init__(
        self,
        fit: FitInterferometer,
        mat_plot_1d: MatPlot1D = None,
        mat_plot_2d: MatPlot2D = None,
    ):
        super().__init__(mat_plot_1d=mat_plot_1d, mat_plot_2d=mat_plot_2d)
        self.fit = fit

        self._fit_interferometer_meta_plotter = FitInterferometerPlotterMeta(
            fit=self.fit,
            mat_plot_1d=self.mat_plot_1d,
            mat_plot_2d=self.mat_plot_2d,
        )

        self.figures_2d = self._fit_interferometer_meta_plotter.figures_2d
        self.subplot = self._fit_interferometer_meta_plotter.subplot
        self.subplot_fit = self._fit_interferometer_meta_plotter.subplot_fit
        self.subplot_fit_dirty_images = (
            self._fit_interferometer_meta_plotter.subplot_fit_dirty_images
        )
