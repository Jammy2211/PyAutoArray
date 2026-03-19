import numpy as np

import matplotlib.pyplot as plt

from autoarray.plot.abstract_plotters import AbstractPlotter
from autoarray.plot.wrap.base.output import Output
from autoarray.plot.wrap.base.cmap import Cmap
from autoarray.plot.plots.array import plot_array
from autoarray.plot.plots.grid import plot_grid
from autoarray.plot.plots.yx import plot_yx
from autoarray.dataset.interferometer.dataset import Interferometer
from autoarray.structures.grids.irregular_2d import Grid2DIrregular
from autoarray.structures.plot.structure_plotters import (
    _auto_mask_edge,
    _output_for_plotter,
    _zoom_array,
)


class InterferometerPlotter(AbstractPlotter):
    def __init__(
        self,
        dataset: Interferometer,
        output: Output = None,
        cmap: Cmap = None,
        use_log10: bool = False,
    ):
        super().__init__(output=output, cmap=cmap, use_log10=use_log10)
        self.dataset = dataset

    @property
    def interferometer(self):
        return self.dataset

    def _plot_array(self, array, auto_filename: str, title: str, ax=None):
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
            title=title,
            colormap=self.cmap.cmap,
            use_log10=self.use_log10,
            output_path=output_path,
            output_filename=filename,
            output_format=fmt,
            structure=array,
        )

    def _plot_grid(self, grid, auto_filename: str, title: str, color_array=None, ax=None):
        if ax is None:
            output_path, filename, fmt = _output_for_plotter(self.output, auto_filename)
        else:
            output_path, filename, fmt = None, auto_filename, "png"

        plot_grid(
            grid=np.array(grid.array),
            ax=ax,
            color_array=color_array,
            title=title,
            output_path=output_path,
            output_filename=filename,
            output_format=fmt,
        )

    def _plot_yx(
        self,
        y,
        x,
        auto_filename: str,
        title: str,
        ylabel: str = "",
        xlabel: str = "",
        plot_axis_type: str = "linear",
        ax=None,
    ):
        if ax is None:
            output_path, filename, fmt = _output_for_plotter(self.output, auto_filename)
        else:
            output_path, filename, fmt = None, auto_filename, "png"

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
        u_wavelengths: bool = False,
        v_wavelengths: bool = False,
        uv_wavelengths: bool = False,
        amplitudes_vs_uv_distances: bool = False,
        phases_vs_uv_distances: bool = False,
        dirty_image: bool = False,
        dirty_noise_map: bool = False,
        dirty_signal_to_noise_map: bool = False,
    ):
        if data:
            self._plot_grid(
                grid=self.dataset.data.in_grid,
                auto_filename="data",
                title="Visibilities",
            )
        if noise_map:
            self._plot_grid(
                grid=self.dataset.data.in_grid,
                auto_filename="noise_map",
                title="Noise-Map",
                color_array=np.real(self.dataset.noise_map),
            )
        if u_wavelengths:
            self._plot_yx(
                y=self.dataset.uv_wavelengths[:, 0],
                x=None,
                auto_filename="u_wavelengths",
                title="U-Wavelengths",
                ylabel="Wavelengths",
                plot_axis_type="linear",
            )
        if v_wavelengths:
            self._plot_yx(
                y=self.dataset.uv_wavelengths[:, 1],
                x=None,
                auto_filename="v_wavelengths",
                title="V-Wavelengths",
                ylabel="Wavelengths",
                plot_axis_type="linear",
            )
        if uv_wavelengths:
            self._plot_grid(
                grid=Grid2DIrregular.from_yx_1d(
                    y=self.dataset.uv_wavelengths[:, 1] / 10**3.0,
                    x=self.dataset.uv_wavelengths[:, 0] / 10**3.0,
                ),
                auto_filename="uv_wavelengths",
                title="UV-Wavelengths",
            )
        if amplitudes_vs_uv_distances:
            self._plot_yx(
                y=self.dataset.amplitudes,
                x=self.dataset.uv_distances / 10**3.0,
                auto_filename="amplitudes_vs_uv_distances",
                title="Amplitudes vs UV-distances",
                ylabel="Jy",
                xlabel="k$\\lambda$",
                plot_axis_type="scatter",
            )
        if phases_vs_uv_distances:
            self._plot_yx(
                y=self.dataset.phases,
                x=self.dataset.uv_distances / 10**3.0,
                auto_filename="phases_vs_uv_distances",
                title="Phases vs UV-distances",
                ylabel="deg",
                xlabel="k$\\lambda$",
                plot_axis_type="scatter",
            )
        if dirty_image:
            self._plot_array(
                array=self.dataset.dirty_image,
                auto_filename="dirty_image",
                title="Dirty Image",
            )
        if dirty_noise_map:
            self._plot_array(
                array=self.dataset.dirty_noise_map,
                auto_filename="dirty_noise_map",
                title="Dirty Noise Map",
            )
        if dirty_signal_to_noise_map:
            self._plot_array(
                array=self.dataset.dirty_signal_to_noise_map,
                auto_filename="dirty_signal_to_noise_map",
                title="Dirty Signal-To-Noise Map",
            )

    def subplot_dataset(self):
        fig, axes = plt.subplots(2, 3, figsize=(21, 14))
        axes = axes.flatten()

        self._plot_grid(
            self.dataset.data.in_grid, "data", "Visibilities", ax=axes[0]
        )
        self._plot_grid(
            Grid2DIrregular.from_yx_1d(
                y=self.dataset.uv_wavelengths[:, 1] / 10**3.0,
                x=self.dataset.uv_wavelengths[:, 0] / 10**3.0,
            ),
            "uv_wavelengths",
            "UV-Wavelengths",
            ax=axes[1],
        )
        self._plot_yx(
            self.dataset.amplitudes,
            self.dataset.uv_distances / 10**3.0,
            "amplitudes_vs_uv_distances",
            "Amplitudes vs UV-distances",
            ylabel="Jy",
            xlabel="k$\\lambda$",
            plot_axis_type="scatter",
            ax=axes[2],
        )
        self._plot_yx(
            self.dataset.phases,
            self.dataset.uv_distances / 10**3.0,
            "phases_vs_uv_distances",
            "Phases vs UV-distances",
            ylabel="deg",
            xlabel="k$\\lambda$",
            plot_axis_type="scatter",
            ax=axes[3],
        )
        self._plot_array(
            self.dataset.dirty_image, "dirty_image", "Dirty Image", ax=axes[4]
        )
        self._plot_array(
            self.dataset.dirty_signal_to_noise_map,
            "dirty_signal_to_noise_map",
            "Dirty Signal-To-Noise Map",
            ax=axes[5],
        )

        plt.tight_layout()
        self.output.subplot_to_figure(auto_filename="subplot_dataset")
        plt.close()

    def subplot_dirty_images(self):
        fig, axes = plt.subplots(1, 3, figsize=(21, 7))

        self._plot_array(
            self.dataset.dirty_image, "dirty_image", "Dirty Image", ax=axes[0]
        )
        self._plot_array(
            self.dataset.dirty_noise_map,
            "dirty_noise_map",
            "Dirty Noise Map",
            ax=axes[1],
        )
        self._plot_array(
            self.dataset.dirty_signal_to_noise_map,
            "dirty_signal_to_noise_map",
            "Dirty Signal-To-Noise Map",
            ax=axes[2],
        )

        plt.tight_layout()
        self.output.subplot_to_figure(auto_filename="subplot_dirty_images")
        plt.close()
