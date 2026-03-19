import matplotlib.pyplot as plt

from autoarray.plot.abstract_plotters import AbstractPlotter
from autoarray.plot.wrap.base.output import Output
from autoarray.plot.wrap.base.cmap import Cmap
from autoarray.fit.fit_imaging import FitImaging
from autoarray.fit.plot.fit_imaging_plotters import FitImagingPlotterMeta


class FitVectorYXPlotterMeta(FitImagingPlotterMeta):
    """
    Plots FitImaging attributes for vector YX data — delegates to FitImagingPlotterMeta
    with remapped parameter names (image → data).
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

    def subplot_fit(self):
        fig, axes = plt.subplots(2, 3, figsize=(21, 14))
        axes = axes.flatten()

        self._plot_array(self.fit.data, "data", "Image", ax=axes[0])
        self._plot_array(
            self.fit.signal_to_noise_map,
            "signal_to_noise_map",
            "Signal-To-Noise Map",
            ax=axes[1],
        )
        self._plot_array(self.fit.model_data, "model_image", "Model Image", ax=axes[2])

        cmap_orig = self.cmap
        if self.residuals_symmetric_cmap:
            self.cmap = self.cmap.symmetric_cmap_from()

        self._plot_array(
            self.fit.residual_map, "residual_map", "Residual Map", ax=axes[3]
        )
        self._plot_array(
            self.fit.normalized_residual_map,
            "normalized_residual_map",
            "Normalized Residual Map",
            ax=axes[4],
        )

        self.cmap = cmap_orig

        self._plot_array(
            self.fit.chi_squared_map, "chi_squared_map", "Chi-Squared Map", ax=axes[5]
        )

        plt.tight_layout()
        self.output.subplot_to_figure(auto_filename="subplot_fit")
        plt.close()


class FitImagingPlotter(AbstractPlotter):
    def __init__(
        self,
        fit: FitImaging,
        output: Output = None,
        cmap: Cmap = None,
        use_log10: bool = False,
        grid=None,
        positions=None,
        lines=None,
    ):
        super().__init__(output=output, cmap=cmap, use_log10=use_log10)
        self.fit = fit

        self._fit_imaging_meta_plotter = FitVectorYXPlotterMeta(
            fit=self.fit,
            output=self.output,
            cmap=self.cmap,
            use_log10=self.use_log10,
            grid=grid,
            positions=positions,
            lines=lines,
        )

        self.figures_2d = self._fit_imaging_meta_plotter.figures_2d
        self.subplot_fit = self._fit_imaging_meta_plotter.subplot_fit
