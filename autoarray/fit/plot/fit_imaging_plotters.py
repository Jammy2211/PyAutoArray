import numpy as np
from typing import Callable

from autoarray.plot.abstract_plotters import AbstractPlotter
from autoarray.plot.mat_plot.two_d import MatPlot2D
from autoarray.plot.auto_labels import AutoLabels
from autoarray.plot.plots.array import plot_array
from autoarray.fit.fit_imaging import FitImaging
from autoarray.structures.plot.structure_plotters import (
    _auto_mask_edge,
    _numpy_lines,
    _numpy_grid,
    _numpy_positions,
    _output_for_mat_plot,
    _zoom_array,
)


class FitImagingPlotterMeta(AbstractPlotter):
    def __init__(
        self,
        fit,
        mat_plot_2d: MatPlot2D = None,
        grid=None,
        positions=None,
        lines=None,
        residuals_symmetric_cmap: bool = True,
    ):
        super().__init__(mat_plot_2d=mat_plot_2d)
        self.fit = fit
        self.grid = grid
        self.positions = positions
        self.lines = lines
        self.residuals_symmetric_cmap = residuals_symmetric_cmap

    def _plot_array(self, array, auto_labels):
        if array is None:
            return

        is_sub = self.mat_plot_2d.is_for_subplot
        ax = self.mat_plot_2d.setup_subplot() if is_sub else None
        output_path, filename, fmt = _output_for_mat_plot(
            self.mat_plot_2d, is_sub,
            auto_labels.filename if auto_labels else "array",
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
            grid=_numpy_grid(self.grid),
            positions=_numpy_positions(self.positions),
            lines=_numpy_lines(self.lines),
            title=auto_labels.title if auto_labels else "",
            colormap=self.mat_plot_2d.cmap.cmap,
            use_log10=self.mat_plot_2d.use_log10,
            output_path=output_path,
            output_filename=filename,
            output_format=fmt,
            structure=array,
        )

    def figures_2d(
        self,
        data: bool = False,
        noise_map: bool = False,
        signal_to_noise_map: bool = False,
        model_image: bool = False,
        residual_map: bool = False,
        normalized_residual_map: bool = False,
        chi_squared_map: bool = False,
        residual_flux_fraction_map: bool = False,
        suffix: str = "",
    ):
        if data:
            self._plot_array(
                array=self.fit.data,
                auto_labels=AutoLabels(title="Data", filename=f"data{suffix}"),
            )
        if noise_map:
            self._plot_array(
                array=self.fit.noise_map,
                auto_labels=AutoLabels(title="Noise-Map", filename=f"noise_map{suffix}"),
            )
        if signal_to_noise_map:
            self._plot_array(
                array=self.fit.signal_to_noise_map,
                auto_labels=AutoLabels(
                    title="Signal-To-Noise Map", filename=f"signal_to_noise_map{suffix}"
                ),
            )
        if model_image:
            self._plot_array(
                array=self.fit.model_data,
                auto_labels=AutoLabels(title="Model Image", filename=f"model_image{suffix}"),
            )

        cmap_original = self.mat_plot_2d.cmap
        if self.residuals_symmetric_cmap:
            self.mat_plot_2d.cmap = self.mat_plot_2d.cmap.symmetric_cmap_from()

        if residual_map:
            self._plot_array(
                array=self.fit.residual_map,
                auto_labels=AutoLabels(
                    title="Residual Map", filename=f"residual_map{suffix}"
                ),
            )
        if normalized_residual_map:
            self._plot_array(
                array=self.fit.normalized_residual_map,
                auto_labels=AutoLabels(
                    title="Normalized Residual Map",
                    filename=f"normalized_residual_map{suffix}",
                ),
            )

        self.mat_plot_2d.cmap = cmap_original

        if chi_squared_map:
            self._plot_array(
                array=self.fit.chi_squared_map,
                auto_labels=AutoLabels(
                    title="Chi-Squared Map", filename=f"chi_squared_map{suffix}"
                ),
            )
        if residual_flux_fraction_map:
            self._plot_array(
                array=self.fit.residual_map,
                auto_labels=AutoLabels(
                    title="Residual Flux Fraction Map",
                    filename=f"residual_flux_fraction_map{suffix}",
                ),
            )

    def subplot(
        self,
        data: bool = False,
        noise_map: bool = False,
        signal_to_noise_map: bool = False,
        model_image: bool = False,
        residual_map: bool = False,
        normalized_residual_map: bool = False,
        chi_squared_map: bool = False,
        residual_flux_fraction_map: bool = False,
        auto_filename: str = "subplot_fit",
    ):
        self._subplot_custom_plot(
            data=data,
            noise_map=noise_map,
            signal_to_noise_map=signal_to_noise_map,
            model_image=model_image,
            residual_map=residual_map,
            normalized_residual_map=normalized_residual_map,
            chi_squared_map=chi_squared_map,
            residual_flux_fraction_map=residual_flux_fraction_map,
            auto_labels=AutoLabels(filename=auto_filename),
        )

    def subplot_fit(self):
        return self.subplot(
            data=True,
            signal_to_noise_map=True,
            model_image=True,
            residual_map=True,
            normalized_residual_map=True,
            chi_squared_map=True,
        )


class FitImagingPlotter(AbstractPlotter):
    def __init__(
        self,
        fit: FitImaging,
        mat_plot_2d: MatPlot2D = None,
        grid=None,
        positions=None,
        lines=None,
    ):
        super().__init__(mat_plot_2d=mat_plot_2d)
        self.fit = fit

        self._fit_imaging_meta_plotter = FitImagingPlotterMeta(
            fit=self.fit,
            mat_plot_2d=self.mat_plot_2d,
            grid=grid,
            positions=positions,
            lines=lines,
        )

        self.figures_2d = self._fit_imaging_meta_plotter.figures_2d
        self.subplot = self._fit_imaging_meta_plotter.subplot
        self.subplot_fit = self._fit_imaging_meta_plotter.subplot_fit
