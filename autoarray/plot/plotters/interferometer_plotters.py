from autoarray.plot.plotters import abstract_plotters
from autoarray.plot.mat_wrap import visuals as vis
from autoarray.plot.mat_wrap import include as inc
from autoarray.plot.mat_wrap import mat_plot
import typing
from autoarray.plot.plotters import structure_plotters
from autoarray.structures import grids
import numpy as np


class InterferometerPlotter(abstract_plotters.AbstractPlotter):
    def __init__(
        self,
        mat_plot_1d: mat_plot.MatPlot1D = mat_plot.MatPlot1D(),
        visuals_1d: vis.Visuals1D = vis.Visuals1D(),
        include_1d: inc.Include1D = inc.Include1D(),
        mat_plot_2d: mat_plot.MatPlot2D = mat_plot.MatPlot2D(),
        visuals_2d: vis.Visuals2D = vis.Visuals2D(),
        include_2d: inc.Include2D = inc.Include2D(),
    ):

        super().__init__(
            mat_plot_1d=mat_plot_1d,
            include_1d=include_1d,
            visuals_1d=visuals_1d,
            mat_plot_2d=mat_plot_2d,
            include_2d=include_2d,
            visuals_2d=visuals_2d,
        )

    def subplot_interferometer(self, interferometer):
        """Plot the interferometer data_type as a sub-mat_plot_2d of all its quantites (e.g. the dataset, noise_map, PSF, Signal-to_noise-map, \
         etc).

        Set *autolens.data_type.array.mat_plot_2d.mat_plot_2d* for a description of all innput parameters not described below.

        Parameters
        -----------
        interferometer : data_type.UVPlaneData
            The interferometer data_type, which include_2d the observed data_type, noise_map, PSF, signal-to-noise_map, etc.
        origin : True
            If true, the origin of the dataset's coordinate system is plotted as a 'x'.
        image_plane_pix_grid : np.ndarray or data_type.array.grid_stacks.PixGrid
            If an adaptive pixelization whose pixels are formed by tracing pixels from the dataset, this plots those pixels \
            over the immage.
        ignore_config : bool
            If `False`, the config file general.ini is used to determine whether the subpot is plotted. If `True`, the \
            config file is ignored.
        """

        mat_plot_2d = self.mat_plot_2d.plotter_for_subplot_from(
            func=self.subplot_interferometer
        )
        mat_plot_1d = self.mat_plot_1d.plotter_for_subplot_from(
            func=self.subplot_interferometer
        )

        number_subplots = 4

        mat_plot_2d.open_subplot_figure(number_subplots=number_subplots)

        mat_plot_2d.setup_subplot(number_subplots=number_subplots, subplot_index=1)

        self.uv_wavelengths(interferometer=interferometer)

        mat_plot_2d.setup_subplot(number_subplots=number_subplots, subplot_index=2)

        self.visibilities(interferometer=interferometer)

        mat_plot_2d.setup_subplot(number_subplots=number_subplots, subplot_index=3)

        self.amplitudes_vs_uv_distances(interferometer=interferometer)

        mat_plot_1d.setup_subplot(number_subplots=number_subplots, subplot_index=4)

        self.phases_vs_uv_distances(interferometer=interferometer)

        mat_plot_2d.output.subplot_to_figure()

        mat_plot_2d.figure.close()

    def individual(
        self,
        interferometer,
        plot_visibilities=False,
        plot_noise_map=False,
        plot_u_wavelengths=False,
        plot_v_wavelengths=False,
        plot_uv_wavelengths=False,
        plot_amplitudes_vs_uv_distances=False,
        plot_phases_vs_uv_distances=False,
    ):
        """
        Plot each attribute of the interferometer data_type as individual figures one by one (e.g. the dataset, noise_map, PSF, \
         Signal-to_noise-map, etc).

        Set *autolens.data_type.array.mat_plot_2d.mat_plot_2d* for a description of all innput parameters not described below.

        Parameters
        -----------
        interferometer : data_type.UVPlaneData
            The interferometer data_type, which include_2d the observed data_type, noise_map, PSF, signal-to-noise_map, etc.
        origin : True
            If true, the origin of the dataset's coordinate system is plotted as a 'x'.
        """

        if plot_visibilities:
            self.visibilities(interferometer=interferometer)
        if plot_noise_map:
            self.noise_map(interferometer=interferometer)
        if plot_u_wavelengths:
            self.uv_wavelengths(interferometer=interferometer)
        if plot_v_wavelengths:
            self.v_wavelengths(interferometer=interferometer)
        if plot_uv_wavelengths:
            self.uv_wavelengths(interferometer=interferometer)
        if plot_amplitudes_vs_uv_distances:
            self.amplitudes_vs_uv_distances(interferometer=interferometer)
        if plot_phases_vs_uv_distances:
            self.phases_vs_uv_distances(interferometer=interferometer)

    @abstract_plotters.set_labels
    def visibilities(self, interferometer):
        """Plot the observed image of the imaging data_type.

        Set *autolens.data_type.array.mat_plot_2d.mat_plot_2d* for a description of all input parameters not described below.

        Parameters
        -----------
        image : ScaledSquarePixelArray
            The image of the dataset.
        origin : True
            If true, the origin of the dataset's coordinate system is plotted as a 'x'.
        image_plane_pix_grid : np.ndarray or data_type.array.grid_stacks.PixGrid
            If an adaptive pixelization whose pixels are formed by tracing pixels from the dataset, this plots those pixels \
            over the immage.
        """
        self.mat_plot_2d.plot_grid(
            grid=interferometer.visibilities.in_grid,
            visuals_2d=self.visuals_2d
            + self.include_2d.visuals_from_interferometer(
                interferometer=interferometer
            ),
        )

    @abstract_plotters.set_labels
    def noise_map(self, interferometer):
        """Plot the observed image of the imaging data_type.

        Set *autolens.data_type.array.mat_plot_2d.mat_plot_2d* for a description of all input parameters not described below.

        Parameters
        -----------
        image : ScaledSquarePixelArray
            The image of the dataset.
        origin : True
            If true, the origin of the dataset's coordinate system is plotted as a 'x'.
        image_plane_pix_grid : np.ndarray or data_type.array.grid_stacks.PixGrid
            If an adaptive pixelization whose pixels are formed by tracing pixels from the dataset, this plots those pixels \
            over the immage.
        """
        self.mat_plot_2d.plot_grid(
            grid=interferometer.visibilities.in_grid,
            visuals_2d=self.visuals_2d
            + self.include_2d.visuals_from_interferometer(
                interferometer=interferometer
            ),
            color_array=interferometer.noise_map.real,
        )

    @abstract_plotters.set_labels
    def u_wavelengths(
        self, interferometer, label="Wavelengths", plot_axis_type="linear"
    ):
        """Plot the observed image of the imaging data_type.

        Set *autolens.data_type.array.mat_plot_1d.mat_plot_1d* for a description of all input parameters not described below.

        Parameters
        -----------
        image : ScaledSquarePixelArray
            The image of the dataset.
        origin : True
            If true, the origin of the dataset's coordinate system is plotted as a 'x'.
        image_plane_pix_grid : np.ndarray or data_type.array.grid_stacks.PixGrid
            If an adaptive pixelization whose pixels are formed by tracing pixels from the dataset, this plots those pixels \
            over the immage.
        """
        self.mat_plot_1d.plot_line(
            y=interferometer.uv_wavelengths[:, 0],
            x=None,
            label=label,
            plot_axis_type=plot_axis_type,
        )

    @abstract_plotters.set_labels
    def v_wavelengths(
        self, interferometer, label="Wavelengths", plot_axis_type="linear"
    ):
        """Plot the observed image of the imaging data_type.

        Set *autolens.data_type.array.mat_plot_1d.mat_plot_1d* for a description of all input parameters not described below.

        Parameters
        -----------
        image : ScaledSquarePixelArray
            The image of the dataset.
        origin : True
            If true, the origin of the dataset's coordinate system is plotted as a 'x'.
        image_plane_pix_grid : np.ndarray or data_type.array.grid_stacks.PixGrid
            If an adaptive pixelization whose pixels are formed by tracing pixels from the dataset, this plots those pixels \
            over the immage.
        """
        self.mat_plot_1d.plot_line(
            y=interferometer.uv_wavelengths[:, 1],
            x=None,
            label=label,
            plot_axis_type=plot_axis_type,
        )

    @abstract_plotters.set_labels
    def uv_wavelengths(
        self,
        interferometer,
        label_yunits="V-Wavelengths ($\lambda$)",
        label_xunits="U-Wavelengths ($\lambda$)",
    ):
        """Plot the observed image of the imaging data_type.

        Set *autolens.data_type.array.mat_plot_2d.mat_plot_2d* for a description of all input parameters not described below.

        Parameters
        -----------
        image : ScaledSquarePixelArray
            The image of the dataset.
        origin : True
            If true, the origin of the dataset's coordinate system is plotted as a 'x'.
        image_plane_pix_grid : np.ndarray or data_type.array.grid_stacks.PixGrid
            If an adaptive pixelization whose pixels are formed by tracing pixels from the dataset, this plots those pixels \
            over the immage.
        """
        self.mat_plot_2d.plot_grid(
            grid=grids.GridIrregularGrouped.from_yx_1d(
                y=interferometer.uv_wavelengths[:, 1] / 10 ** 3.0,
                x=interferometer.uv_wavelengths[:, 0] / 10 ** 3.0,
            ),
            visuals_2d=self.visuals_2d
            + self.include_2d.visuals_from_interferometer(
                interferometer=interferometer
            ),
            symmetric_around_centre=True,
        )

    @abstract_plotters.set_labels
    def amplitudes_vs_uv_distances(
        self,
        interferometer,
        label_yunits="amplitude (Jy)",
        label_xunits=r"UV$_{distance}$ (k$\lambda$)",
    ):

        self.mat_plot_1d.plot_line(
            y=interferometer.amplitudes,
            x=interferometer.uv_distances / 10 ** 3.0,
            plot_axis_type="scatter",
        )

    @abstract_plotters.set_labels
    def phases_vs_uv_distances(
        self,
        interferometer,
        label_yunits="phase (deg)",
        label_xunits=r"UV$_{distance}$ (k$\lambda$)",
    ):

        self.mat_plot_1d.plot_line(
            y=interferometer.phases,
            x=interferometer.uv_distances / 10 ** 3.0,
            plot_axis_type="scatter",
        )
