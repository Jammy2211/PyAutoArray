import numpy as np

from autoarray.plot.abstract_plotters import AbstractPlotter
from autoarray.plot.mat_plot.one_d import MatPlot1D
from autoarray.plot.mat_plot.two_d import MatPlot2D
from autoarray.plot.auto_labels import AutoLabels
from autoarray.plot.plots.array import plot_array
from autoarray.plot.plots.grid import plot_grid
from autoarray.plot.plots.yx import plot_yx
from autoarray.dataset.interferometer.dataset import Interferometer
from autoarray.structures.grids.irregular_2d import Grid2DIrregular
from autoarray.structures.plot.structure_plotters import (
    _auto_mask_edge,
    _output_for_mat_plot,
    _zoom_array,
)


class InterferometerPlotter(AbstractPlotter):
    def __init__(
        self,
        dataset: Interferometer,
        mat_plot_1d: MatPlot1D = None,
        mat_plot_2d: MatPlot2D = None,
    ):
        super().__init__(mat_plot_1d=mat_plot_1d, mat_plot_2d=mat_plot_2d)
        self.dataset = dataset

    @property
    def interferometer(self):
        return self.dataset

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

    def subplot(
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
        auto_filename: str = "subplot_dataset",
    ):
        self._subplot_custom_plot(
            data=data,
            noise_map=noise_map,
            u_wavelengths=u_wavelengths,
            v_wavelengths=v_wavelengths,
            uv_wavelengths=uv_wavelengths,
            amplitudes_vs_uv_distances=amplitudes_vs_uv_distances,
            phases_vs_uv_distances=phases_vs_uv_distances,
            dirty_image=dirty_image,
            dirty_noise_map=dirty_noise_map,
            dirty_signal_to_noise_map=dirty_signal_to_noise_map,
            auto_labels=AutoLabels(filename=auto_filename),
        )

    def subplot_dataset(self):
        return self.subplot(
            data=True,
            uv_wavelengths=True,
            amplitudes_vs_uv_distances=True,
            phases_vs_uv_distances=True,
            dirty_image=True,
            dirty_signal_to_noise_map=True,
            auto_filename="subplot_dataset",
        )

    def subplot_dirty_images(self):
        return self.subplot(
            dirty_image=True,
            dirty_noise_map=True,
            dirty_signal_to_noise_map=True,
            auto_filename="subplot_dirty_images",
        )
