from typing import Callable

from autoarray.plot.abstract_plotters import AbstractPlotter
from autoarray.plot.mat_plot.two_d import MatPlot2D
from autoarray.plot.auto_labels import AutoLabels
from autoarray.fit.fit_imaging import FitImaging
from autoarray.fit.plot.fit_imaging_plotters import FitImagingPlotterMeta


class FitVectorYXPlotterMeta(FitImagingPlotterMeta):
    """
    Plots FitImaging attributes — delegates entirely to FitImagingPlotterMeta
    which already uses the standalone plot_array function.
    """

    def figures_2d(
        self,
        image: bool = False,
        noise_map: bool = False,
        signal_to_noise_map: bool = False,
        model_image: bool = False,
        residual_map: bool = False,
        normalized_residual_map: bool = False,
        chi_squared_map: bool = False,
    ):
        super().figures_2d(
            data=image,
            noise_map=noise_map,
            signal_to_noise_map=signal_to_noise_map,
            model_image=model_image,
            residual_map=residual_map,
            normalized_residual_map=normalized_residual_map,
            chi_squared_map=chi_squared_map,
        )

    def subplot(
        self,
        image: bool = False,
        noise_map: bool = False,
        signal_to_noise_map: bool = False,
        model_image: bool = False,
        residual_map: bool = False,
        normalized_residual_map: bool = False,
        chi_squared_map: bool = False,
        auto_filename: str = "subplot_fit",
    ):
        self._subplot_custom_plot(
            image=image,
            noise_map=noise_map,
            signal_to_noise_map=signal_to_noise_map,
            model_image=model_image,
            residual_map=residual_map,
            normalized_residual_map=normalized_residual_map,
            chi_squared_map=chi_squared_map,
            auto_labels=AutoLabels(filename=auto_filename),
        )

    def subplot_fit(self):
        return self.subplot(
            image=True,
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

        self._fit_imaging_meta_plotter = FitVectorYXPlotterMeta(
            fit=self.fit,
            mat_plot_2d=self.mat_plot_2d,
            grid=grid,
            positions=positions,
            lines=lines,
        )

        self.figures_2d = self._fit_imaging_meta_plotter.figures_2d
        self.subplot = self._fit_imaging_meta_plotter.subplot
        self.subplot_fit = self._fit_imaging_meta_plotter.subplot_fit
