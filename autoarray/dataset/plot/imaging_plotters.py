import numpy as np
from typing import Optional

import matplotlib.pyplot as plt

from autoarray.plot.abstract_plotters import AbstractPlotter
from autoarray.plot.wrap.base.output import Output
from autoarray.plot.wrap.base.cmap import Cmap
from autoarray.plot.plots.array import plot_array
from autoarray.structures.plot.structure_plotters import (
    _auto_mask_edge,
    _numpy_lines,
    _numpy_grid,
    _numpy_positions,
    _output_for_plotter,
    _zoom_array,
)
from autoarray.dataset.imaging.dataset import Imaging


class ImagingPlotterMeta(AbstractPlotter):
    def __init__(
        self,
        dataset: Imaging,
        output: Output = None,
        cmap: Cmap = None,
        use_log10: bool = False,
        grid=None,
        positions=None,
        lines=None,
    ):
        super().__init__(output=output, cmap=cmap, use_log10=use_log10)
        self.dataset = dataset
        self.grid = grid
        self.positions = positions
        self.lines = lines

    @property
    def imaging(self):
        return self.dataset

    def _plot_array(self, array, auto_filename: str, title: str, ax=None):
        if array is None:
            return

        array = _zoom_array(array)

        if ax is None:
            output_path, filename, fmt = _output_for_plotter(self.output, auto_filename)
        else:
            output_path, filename, fmt = None, auto_filename, "png"

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

    def subplot_dataset(self):
        use_log10_orig = self.use_log10

        fig, axes = plt.subplots(3, 3, figsize=(21, 21))
        axes = axes.flatten()

        self._plot_array(self.dataset.data, "data", "Data", ax=axes[0])

        self.use_log10 = True
        self._plot_array(self.dataset.data, "data_log10", "Data (log10)", ax=axes[1])
        self.use_log10 = use_log10_orig

        self._plot_array(self.dataset.noise_map, "noise_map", "Noise-Map", ax=axes[2])

        if self.dataset.psf is not None:
            self._plot_array(
                self.dataset.psf.kernel, "psf", "Point Spread Function", ax=axes[3]
            )
            self.use_log10 = True
            self._plot_array(
                self.dataset.psf.kernel, "psf_log10", "PSF (log10)", ax=axes[4]
            )
            self.use_log10 = use_log10_orig

        self._plot_array(
            self.dataset.signal_to_noise_map,
            "signal_to_noise_map",
            "Signal-To-Noise Map",
            ax=axes[5],
        )
        self._plot_array(
            self.dataset.grids.over_sample_size_lp,
            "over_sample_size_lp",
            "Over Sample Size (Light Profiles)",
            ax=axes[6],
        )
        self._plot_array(
            self.dataset.grids.over_sample_size_pixelization,
            "over_sample_size_pixelization",
            "Over Sample Size (Pixelization)",
            ax=axes[7],
        )

        plt.tight_layout()
        self.output.subplot_to_figure(auto_filename="subplot_dataset")
        plt.close()

        self.use_log10 = use_log10_orig


class ImagingPlotter(AbstractPlotter):
    def __init__(
        self,
        dataset: Imaging,
        output: Output = None,
        cmap: Cmap = None,
        use_log10: bool = False,
        grid=None,
        positions=None,
        lines=None,
    ):
        super().__init__(output=output, cmap=cmap, use_log10=use_log10)
        self.dataset = dataset

        self._imaging_meta_plotter = ImagingPlotterMeta(
            dataset=self.dataset,
            output=self.output,
            cmap=self.cmap,
            use_log10=self.use_log10,
            grid=grid,
            positions=positions,
            lines=lines,
        )

        self.figures_2d = self._imaging_meta_plotter.figures_2d
        self.subplot_dataset = self._imaging_meta_plotter.subplot_dataset
