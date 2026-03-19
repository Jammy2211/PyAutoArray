import numpy as np

import matplotlib.pyplot as plt

from autoarray.plot.abstract_plotters import AbstractPlotter
from autoarray.plot.wrap.base.output import Output
from autoarray.plot.wrap.base.cmap import Cmap
from autoarray.plot.plots.array import plot_array
from autoarray.fit.fit_imaging import FitImaging
from autoarray.structures.plot.structure_plotters import (
    _auto_mask_edge,
    _numpy_lines,
    _numpy_grid,
    _numpy_positions,
    _output_for_plotter,
    _zoom_array,
)


class FitImagingPlotterMeta(AbstractPlotter):
    def __init__(
        self,
        fit,
        output: Output = None,
        cmap: Cmap = None,
        use_log10: bool = False,
        grid=None,
        positions=None,
        lines=None,
        residuals_symmetric_cmap: bool = True,
    ):
        super().__init__(output=output, cmap=cmap, use_log10=use_log10)
        self.fit = fit
        self.grid = grid
        self.positions = positions
        self.lines = lines
        self.residuals_symmetric_cmap = residuals_symmetric_cmap

    def _plot_array(self, array, auto_filename: str, title: str, ax=None):
        if array is None:
            return

        if ax is None:
            output_path, filename, fmt = _output_for_plotter(self.output, auto_filename)
        else:
            output_path, filename, fmt = None, auto_filename, "png"

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
            title=title,
            colormap=self.cmap.cmap,
            use_log10=self.use_log10,
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
                auto_filename=f"data{suffix}",
                title="Data",
            )
        if noise_map:
            self._plot_array(
                array=self.fit.noise_map,
                auto_filename=f"noise_map{suffix}",
                title="Noise-Map",
            )
        if signal_to_noise_map:
            self._plot_array(
                array=self.fit.signal_to_noise_map,
                auto_filename=f"signal_to_noise_map{suffix}",
                title="Signal-To-Noise Map",
            )
        if model_image:
            self._plot_array(
                array=self.fit.model_data,
                auto_filename=f"model_image{suffix}",
                title="Model Image",
            )

        cmap_original = self.cmap
        if self.residuals_symmetric_cmap:
            self.cmap = self.cmap.symmetric_cmap_from()

        if residual_map:
            self._plot_array(
                array=self.fit.residual_map,
                auto_filename=f"residual_map{suffix}",
                title="Residual Map",
            )
        if normalized_residual_map:
            self._plot_array(
                array=self.fit.normalized_residual_map,
                auto_filename=f"normalized_residual_map{suffix}",
                title="Normalized Residual Map",
            )

        self.cmap = cmap_original

        if chi_squared_map:
            self._plot_array(
                array=self.fit.chi_squared_map,
                auto_filename=f"chi_squared_map{suffix}",
                title="Chi-Squared Map",
            )
        if residual_flux_fraction_map:
            self._plot_array(
                array=self.fit.residual_map,
                auto_filename=f"residual_flux_fraction_map{suffix}",
                title="Residual Flux Fraction Map",
            )

    def subplot_fit(self):
        fig, axes = plt.subplots(2, 3, figsize=(21, 14))
        axes = axes.flatten()

        self._plot_array(self.fit.data, "data", "Data", ax=axes[0])
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

        self._plot_array(self.fit.residual_map, "residual_map", "Residual Map", ax=axes[3])
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

        self._fit_imaging_meta_plotter = FitImagingPlotterMeta(
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
