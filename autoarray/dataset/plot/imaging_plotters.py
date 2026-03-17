import copy
import numpy as np
from typing import Callable, Optional

from autoarray.plot.visuals.two_d import Visuals2D
from autoarray.plot.mat_plot.two_d import MatPlot2D
from autoarray.plot.auto_labels import AutoLabels
from autoarray.plot.abstract_plotters import AbstractPlotter
from autoarray.plot.plots.array import plot_array
from autoarray.structures.plot.structure_plotters import (
    _lines_from_visuals,
    _positions_from_visuals,
    _mask_edge_from,
    _grid_from_visuals,
    _output_for_mat_plot,
    _zoom_array,
)
from autoarray.dataset.imaging.dataset import Imaging


class ImagingPlotterMeta(AbstractPlotter):
    def __init__(
        self,
        dataset: Imaging,
        mat_plot_2d: MatPlot2D = None,
        visuals_2d: Visuals2D = None,
    ):
        super().__init__(mat_plot_2d=mat_plot_2d, visuals_2d=visuals_2d)
        self.dataset = dataset

    @property
    def imaging(self):
        return self.dataset

    def _plot_array(self, array, auto_filename: str, title: str, ax=None):
        """Internal helper: plot an Array2D via plot_array()."""
        if array is None:
            return

        is_sub = self.mat_plot_2d.is_for_subplot
        if ax is None:
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
            mask=_mask_edge_from(array if hasattr(array, "mask") else None, self.visuals_2d),
            grid=_grid_from_visuals(self.visuals_2d),
            positions=_positions_from_visuals(self.visuals_2d),
            lines=_lines_from_visuals(self.visuals_2d),
            title=title,
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
        psf: bool = False,
        signal_to_noise_map: bool = False,
        over_sample_size_lp: bool = False,
        over_sample_size_pixelization: bool = False,
        title_str: Optional[str] = None,
    ):
        if data:
            self._plot_array(
                array=self.dataset.data,
                auto_filename="data",
                title=title_str or "Data",
            )

        if noise_map:
            self._plot_array(
                array=self.dataset.noise_map,
                auto_filename="noise_map",
                title=title_str or "Noise-Map",
            )

        if psf:
            if self.dataset.psf is not None:
                self._plot_array(
                    array=self.dataset.psf.kernel,
                    auto_filename="psf",
                    title=title_str or "Point Spread Function",
                )

        if signal_to_noise_map:
            self._plot_array(
                array=self.dataset.signal_to_noise_map,
                auto_filename="signal_to_noise_map",
                title=title_str or "Signal-To-Noise Map",
            )

        if over_sample_size_lp:
            self._plot_array(
                array=self.dataset.grids.over_sample_size_lp,
                auto_filename="over_sample_size_lp",
                title=title_str or "Over Sample Size (Light Profiles)",
            )

        if over_sample_size_pixelization:
            self._plot_array(
                array=self.dataset.grids.over_sample_size_pixelization,
                auto_filename="over_sample_size_pixelization",
                title=title_str or "Over Sample Size (Pixelization)",
            )

    def subplot(
        self,
        data: bool = False,
        noise_map: bool = False,
        psf: bool = False,
        signal_to_noise_map: bool = False,
        over_sampling: bool = False,
        over_sampling_pixelization: bool = False,
        auto_filename: str = "subplot_dataset",
    ):
        self._subplot_custom_plot(
            data=data,
            noise_map=noise_map,
            psf=psf,
            signal_to_noise_map=signal_to_noise_map,
            over_sampling=over_sampling,
            over_sampling_pixelization=over_sampling_pixelization,
            auto_labels=AutoLabels(filename=auto_filename),
        )

    def subplot_dataset(self):
        use_log10_original = self.mat_plot_2d.use_log10

        self.open_subplot_figure(number_subplots=9)

        self.figures_2d(data=True)

        contour_original = copy.copy(self.mat_plot_2d.contour)

        self.mat_plot_2d.use_log10 = True
        self.mat_plot_2d.contour = False
        self.figures_2d(data=True)
        self.mat_plot_2d.use_log10 = False
        self.mat_plot_2d.contour = contour_original

        self.figures_2d(noise_map=True)
        self.figures_2d(psf=True)

        self.mat_plot_2d.use_log10 = True
        self.figures_2d(psf=True)
        self.mat_plot_2d.use_log10 = False

        self.figures_2d(signal_to_noise_map=True)
        self.figures_2d(over_sample_size_lp=True)
        self.figures_2d(over_sample_size_pixelization=True)

        self.mat_plot_2d.output.subplot_to_figure(auto_filename="subplot_dataset")
        self.close_subplot_figure()

        self.mat_plot_2d.use_log10 = use_log10_original


class ImagingPlotter(AbstractPlotter):
    def __init__(
        self,
        dataset: Imaging,
        mat_plot_2d: MatPlot2D = None,
        visuals_2d: Visuals2D = None,
    ):
        super().__init__(mat_plot_2d=mat_plot_2d, visuals_2d=visuals_2d)

        self.dataset = dataset

        self._imaging_meta_plotter = ImagingPlotterMeta(
            dataset=self.dataset,
            mat_plot_2d=self.mat_plot_2d,
            visuals_2d=self.visuals_2d,
        )

        self.figures_2d = self._imaging_meta_plotter.figures_2d
        self.subplot = self._imaging_meta_plotter.subplot
        self.subplot_dataset = self._imaging_meta_plotter.subplot_dataset
