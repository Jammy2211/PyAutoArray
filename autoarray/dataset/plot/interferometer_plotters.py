from autoarray.plot.abstract_plotters import Plotter
from autoarray.plot.mat_wrap.visuals import Visuals1D
from autoarray.plot.mat_wrap.visuals import Visuals2D
from autoarray.plot.mat_wrap.include import Include1D
from autoarray.plot.mat_wrap.include import Include2D
from autoarray.plot.mat_wrap.mat_plot import MatPlot1D
from autoarray.plot.mat_wrap.mat_plot import MatPlot2D
from autoarray.plot.mat_wrap.mat_plot import AutoLabels
from autoarray.dataset.interferometer import Interferometer
from autoarray.structures.grids.two_d.grid_2d_irregular import Grid2DIrregular
import numpy as np


class InterferometerPlotter(Plotter):
    def __init__(
        self,
        interferometer: Interferometer,
        mat_plot_1d: MatPlot1D = MatPlot1D(),
        visuals_1d: Visuals1D = Visuals1D(),
        include_1d: Include1D = Include1D(),
        mat_plot_2d: MatPlot2D = MatPlot2D(),
        visuals_2d: Visuals2D = Visuals2D(),
        include_2d: Include2D = Include2D(),
    ):

        self.interferometer = interferometer

        super().__init__(
            mat_plot_1d=mat_plot_1d,
            include_1d=include_1d,
            visuals_1d=visuals_1d,
            mat_plot_2d=mat_plot_2d,
            include_2d=include_2d,
            visuals_2d=visuals_2d,
        )

    def figures_2d(
        self,
        visibilities: bool = False,
        noise_map: bool = False,
        u_wavelengths: bool = False,
        v_wavelengths: bool = False,
        uv_wavelengths: bool = False,
        amplitudes_vs_uv_distances: bool = False,
        phases_vs_uv_distances: bool = False,
        dirty_image: bool = False,
        dirty_noise_map: bool = False,
        dirty_signal_to_noise_map: bool = False,
        dirty_inverse_noise_map: bool = False,
    ):
        """
        Plot each attribute of the interferometer data_type as individual figures one by one (e.g. the dataset, noise_map, PSF, \
         Signal-to_noise-map, etc).

        Set *autolens.data_type.array.mat_plot_2d.mat_plot_2d* for a description of all innput parameters not described below.

        Parameters
        -----------
        interferometer : data_type.UVPlaneData
            The interferometer data_type, which include the observed data_type, noise_map, PSF, signal-to-noise_map, etc.
        origin : True
            If true, the origin of the dataset's coordinate system is plotted as a 'x'.
        """

        if visibilities:

            self.mat_plot_2d.plot_grid(
                grid=self.interferometer.visibilities.in_grid,
                visuals_2d=self.visuals_2d,
                auto_labels=AutoLabels(title="Visibilities", filename="visibilities"),
            )

        if noise_map:

            self.mat_plot_2d.plot_grid(
                grid=self.interferometer.visibilities.in_grid,
                visuals_2d=self.visuals_2d,
                color_array=self.interferometer.noise_map.real,
                auto_labels=AutoLabels(title="Noise-Map", filename="noise_map"),
            )

        if u_wavelengths:

            self.mat_plot_1d.plot_yx(
                y=self.interferometer.uv_wavelengths[:, 0],
                x=None,
                visuals_1d=self.visuals_1d,
                auto_labels=AutoLabels(
                    title="U-Wavelengths",
                    filename="u_wavelengths",
                    ylabel="Wavelengths",
                ),
                plot_axis_type_override="linear",
            )

        if v_wavelengths:

            self.mat_plot_1d.plot_yx(
                y=self.interferometer.uv_wavelengths[:, 1],
                x=None,
                visuals_1d=self.visuals_1d,
                auto_labels=AutoLabels(
                    title="V-Wavelengths",
                    filename="v_wavelengths",
                    ylabel="Wavelengths",
                ),
                plot_axis_type_override="linear",
            )

        if uv_wavelengths:

            self.mat_plot_2d.plot_grid(
                grid=Grid2DIrregular.from_yx_1d(
                    y=self.interferometer.uv_wavelengths[:, 1] / 10 ** 3.0,
                    x=self.interferometer.uv_wavelengths[:, 0] / 10 ** 3.0,
                ),
                visuals_2d=self.visuals_2d,
                auto_labels=AutoLabels(
                    title="UV-Wavelengths", filename="uv_wavelengths"
                ),
            )

        if amplitudes_vs_uv_distances:

            self.mat_plot_1d.plot_yx(
                y=self.interferometer.amplitudes,
                x=self.interferometer.uv_distances / 10 ** 3.0,
                visuals_1d=self.visuals_1d,
                auto_labels=AutoLabels(
                    title="Amplitudes vs UV-distances",
                    filename="amplitudes_vs_uv_distances",
                    ylabel="amplitude (Jy)",
                    xlabel="U-Wavelengths ($\lambda$)",
                ),
                plot_axis_type_override="scatter",
            )

        if phases_vs_uv_distances:

            self.mat_plot_1d.plot_yx(
                y=self.interferometer.phases,
                x=self.interferometer.uv_distances / 10 ** 3.0,
                visuals_1d=self.visuals_1d,
                auto_labels=AutoLabels(
                    title="Phases vs UV-distances",
                    filename="phases_vs_uv_distances",
                    ylabel="phase (deg)",
                    xlabel=r"UV$_{distance}$ (k$\lambda$)",
                ),
                plot_axis_type_override="scatter",
            )

        if dirty_image:

            self.mat_plot_2d.plot_array(
                array=self.interferometer.dirty_image,
                visuals_2d=self.get_2d.via_mask_from(
                    mask=self.interferometer.real_space_mask
                ),
                auto_labels=AutoLabels(title="Dirty Image", filename="dirty_image_2d"),
            )

        if dirty_noise_map:

            self.mat_plot_2d.plot_array(
                array=self.interferometer.dirty_noise_map,
                visuals_2d=self.get_2d.via_mask_from(
                    mask=self.interferometer.real_space_mask
                ),
                auto_labels=AutoLabels(
                    title="Dirty Noise Map", filename="dirty_noise_map_2d"
                ),
            )

        if dirty_signal_to_noise_map:

            self.mat_plot_2d.plot_array(
                array=self.interferometer.dirty_signal_to_noise_map,
                visuals_2d=self.get_2d.via_mask_from(
                    mask=self.interferometer.real_space_mask
                ),
                auto_labels=AutoLabels(
                    title="Dirty Signal-To-Noise Map",
                    filename="dirty_signal_to_noise_map_2d",
                ),
            )

        if dirty_inverse_noise_map:

            self.mat_plot_2d.plot_array(
                array=self.interferometer.dirty_inverse_noise_map,
                visuals_2d=self.get_2d.via_mask_from(
                    mask=self.interferometer.real_space_mask
                ),
                auto_labels=AutoLabels(
                    title="Dirty Inverse Noise Map",
                    filename="dirty_inverse_noise_map_2d",
                ),
            )

    def subplot(
        self,
        visibilities: bool = False,
        noise_map: bool = False,
        u_wavelengths: bool = False,
        v_wavelengths: bool = False,
        uv_wavelengths: bool = False,
        amplitudes_vs_uv_distances: bool = False,
        phases_vs_uv_distances: bool = False,
        dirty_image: bool = False,
        dirty_noise_map: bool = False,
        dirty_signal_to_noise_map: bool = False,
        dirty_inverse_noise_map: bool = False,
        auto_filename: str = "subplot_interferometer",
    ):

        self._subplot_custom_plot(
            visibilities=visibilities,
            noise_map=noise_map,
            u_wavelengths=u_wavelengths,
            v_wavelengths=v_wavelengths,
            uv_wavelengths=uv_wavelengths,
            amplitudes_vs_uv_distances=amplitudes_vs_uv_distances,
            phases_vs_uv_distances=phases_vs_uv_distances,
            dirty_image=dirty_image,
            dirty_noise_map=dirty_noise_map,
            dirty_signal_to_noise_map=dirty_signal_to_noise_map,
            dirty_inverse_noise_map=dirty_inverse_noise_map,
            auto_labels=AutoLabels(filename=auto_filename),
        )

    def subplot_interferometer(self):
        """Plot the interferometer data_type as a sub-mat_plot_2d of all its quantites (e.g. the dataset, noise_map, PSF, Signal-to_noise-map, \
         etc).

        Set *autolens.data_type.array.mat_plot_2d.mat_plot_2d* for a description of all innput parameters not described below.

        Parameters
        -----------
        interferometer : data_type.UVPlaneData
            The interferometer data_type, which include the observed data_type, noise_map, PSF, signal-to-noise_map, etc.
        origin : True
            If true, the origin of the dataset's coordinate system is plotted as a 'x'.
        image_plane_pix_grid or data_type.array.grid_stacks.PixGrid
            If an adaptive pixelization whose pixels are formed by tracing pixels from the dataset, this plots those pixels \
            over the immage.
        ignore_config : bool
            If `False`, the config file general.ini is used to determine whether the subpot is plotted. If `True`, the \
            config file is ignored.
        """
        return self.subplot(
            visibilities=True,
            uv_wavelengths=True,
            amplitudes_vs_uv_distances=True,
            phases_vs_uv_distances=True,
            auto_filename="subplot_interferometer",
        )

    def subplot_dirty_images(self):
        """Plot the interferometer data_type as a sub-mat_plot_2d of all its quantites (e.g. the dataset, noise_map, PSF, Signal-to_noise-map, \
         etc).

        Set *autolens.data_type.array.mat_plot_2d.mat_plot_2d* for a description of all innput parameters not described below.

        Parameters
        -----------
        interferometer : data_type.UVPlaneData
            The interferometer data_type, which include the observed data_type, noise_map, PSF, signal-to-noise_map, etc.
        origin : True
            If true, the origin of the dataset's coordinate system is plotted as a 'x'.
        image_plane_pix_grid or data_type.array.grid_stacks.PixGrid
            If an adaptive pixelization whose pixels are formed by tracing pixels from the dataset, this plots those pixels \
            over the immage.
        ignore_config : bool
            If `False`, the config file general.ini is used to determine whether the subpot is plotted. If `True`, the \
            config file is ignored.
        """
        return self.subplot(
            dirty_image=True,
            dirty_noise_map=True,
            dirty_signal_to_noise_map=True,
            dirty_inverse_noise_map=True,
            auto_filename="subplot_dirty_images",
        )
