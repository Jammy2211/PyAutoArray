import autoarray as aa
import autoarray.plot as aplt
from os import path
import matplotlib.pyplot as plt
import pytest
import numpy as np
import shutil

directory = path.dirname(path.realpath(__file__))


@pytest.fixture(name="plot_path")
def make_plot_path_setup():
    return path.join(
        "{}".format(path.dirname(path.realpath(__file__))), "files", "plotter"
    )


class TestAbstractPlotterConfig:
    def test__uses_figure_or_subplot_configs_correctly(self):

        figure = aplt.Figure(figsize=(8, 8))
        cmap = aplt.Cmap(cmap="warm")

        plotter = aplt.Plotter(figure=figure, cmap=cmap)

        assert plotter.figure.kwargs["figsize"] == (8, 8)
        assert plotter.figure.kwargs["aspect"] == "square"
        assert plotter.cmap.kwargs["cmap"] == "warm"
        assert plotter.cmap.kwargs["norm"] == "linear"

        figure = aplt.Figure(use_subplot_defaults=True)
        cmap = aplt.Cmap(use_subplot_defaults=True)

        sub_plotter = aplt.Plotter(figure=figure, cmap=cmap)

        assert sub_plotter.figure.kwargs["figsize"] == None
        assert sub_plotter.figure.kwargs["aspect"] == "square"
        assert sub_plotter.cmap.kwargs["cmap"] == "jet"
        assert sub_plotter.cmap.kwargs["norm"] == "linear"


class TestAbstractPlotterPlots:
    def test__plot_array__works_with_all_extras_included(self, plot_path, plot_patch):

        array = aa.Array.ones(shape_2d=(31, 31), pixel_scales=(1.0, 1.0), sub_size=2)

        mask = aa.Mask2D.circular(
            shape_2d=array.shape_2d,
            pixel_scales=array.pixel_scales,
            radius=5.0,
            centre=(2.0, 2.0),
        )

        grid = aa.Grid.uniform(shape_2d=(11, 11), pixel_scales=0.5)

        plotter = aplt.Plotter(
            output=aplt.Output(path=plot_path, filename="array1", format="png")
        )

        plotter.plot_array(
            array=array,
            mask=mask,
            grid=grid,
            positions=aa.GridIrregularGrouped([(-1.0, -1.0)]),
            lines=aa.GridIrregularGrouped([(1.0, 1.0), (2.0, 2.0)]),
            array_overlay=array,
            include_origin=True,
            include_border=True,
        )

        assert path.join(plot_path, "array1.png") in plot_patch.paths

        plotter = aplt.Plotter(
            output=aplt.Output(path=plot_path, filename="array2", format="png")
        )

        plotter.plot_array(
            array=array,
            mask=mask,
            grid=grid,
            positions=aa.GridIrregularGrouped(
                [[(1.0, 1.0), (2.0, 2.0)], [(-1.0, -1.0)]]
            ),
            lines=aa.GridIrregularGrouped(
                [[(1.0, 1.0), (2.0, 2.0)], [(2.0, 4.0), (5.0, 6.0)]]
            ),
            array_overlay=array,
            include_origin=True,
            include_border=True,
        )

        assert path.join(plot_path, "array2.png") in plot_patch.paths

        aplt.Array(
            array=array,
            mask=mask,
            grid=grid,
            positions=aa.GridIrregularGrouped([(-1.0, -1.0)]),
            lines=aa.GridIrregularGrouped([(1.0, 1.0), (2.0, 2.0)]),
            array_overlay=array,
            include=aplt.Include(origin=True, border=True),
            plotter=aplt.Plotter(
                output=aplt.Output(path=plot_path, filename="array3", format="png")
            ),
        )

        assert path.join(plot_path, "array3.png") in plot_patch.paths

    def test__plot_array__fits_files_output_correctly(self, plot_path):

        plot_path = path.join(plot_path, "fits")

        if path.exists(plot_path):
            shutil.rmtree(plot_path)

        arr = aa.Array.ones(shape_2d=(31, 31), pixel_scales=(1.0, 1.0), sub_size=2)

        plotter = aplt.Plotter(
            output=aplt.Output(path=plot_path, filename="array", format="fits")
        )

        plotter.plot_array(array=arr)

        arr = aa.util.array.numpy_array_2d_from_fits(
            file_path=path.join(plot_path, "array.fits"), hdu=0
        )

        assert (arr == np.ones(shape=(31, 31))).all()

        mask = aa.Mask2D.circular(
            shape_2d=(31, 31), pixel_scales=(1.0, 1.0), radius=5.0, centre=(2.0, 2.0)
        )

        masked_array = aa.Array.manual_mask(array=arr, mask=mask)

        plotter.plot_array(array=masked_array)

        arr = aa.util.array.numpy_array_2d_from_fits(
            file_path=path.join(plot_path, "array.fits"), hdu=0
        )

        assert arr.shape == (13, 13)

    def test__plot_frame__works_with_all_extras_included(self, plot_path, plot_patch):

        frame = aa.Frame.ones(shape_2d=(31, 31), pixel_scales=(1.0, 1.0))

        plotter = aplt.Plotter(
            output=aplt.Output(path=plot_path, filename="frame1", format="png")
        )

        plotter.plot_frame(frame=frame, include_origin=True)

        assert path.join(plot_path, "frame1.png") in plot_patch.paths

        plotter = aplt.Plotter(
            output=aplt.Output(path=plot_path, filename="frame2", format="png")
        )

        plotter.plot_frame(frame=frame, include_origin=True)

        assert path.join(plot_path, "frame2.png") in plot_patch.paths

        aplt.Frame(
            frame=frame,
            plotter=aplt.Plotter(
                output=aplt.Output(path=plot_path, filename="frame3", format="png")
            ),
        )

        assert path.join(plot_path, "frame3.png") in plot_patch.paths

    def test__plot_frame__fits_files_output_correctly(self, plot_path):

        plot_path = path.join(plot_path, "fits")

        if path.exists(plot_path):
            shutil.rmtree(plot_path)

        frame = aa.Frame.ones(shape_2d=(31, 31), pixel_scales=(1.0, 1.0))

        plotter = aplt.Plotter(
            output=aplt.Output(path=plot_path, filename="frame", format="fits")
        )

        plotter.plot_frame(frame=frame)

        frame = aa.util.array.numpy_array_2d_from_fits(
            file_path=path.join(plot_path, "frame.fits"), hdu=0
        )

        assert (frame == np.ones(shape=(31, 31))).all()

        mask = aa.Mask2D.unmasked(shape_2d=(31, 31), pixel_scales=(1.0, 1.0))

        masked_frame = aa.Frame.manual_mask(array=frame, mask=mask)

        plotter.plot_frame(frame=masked_frame)

        frame = aa.util.array.numpy_array_2d_from_fits(
            file_path=path.join(plot_path, "frame.fits"), hdu=0
        )

        assert frame.shape == (31, 31)

    def test__plot_grid__works_with_all_extras_included(self, plot_path, plot_patch):
        grid = aa.Grid.uniform(shape_2d=(11, 11), pixel_scales=1.0)
        color_array = np.linspace(start=0.0, stop=1.0, num=grid.shape_1d)

        plotter = aplt.Plotter(
            output=aplt.Output(path=plot_path, filename="grid1", format="png")
        )

        plotter.plot_grid(
            grid=grid,
            color_array=color_array,
            axis_limits=[-1.5, 1.5, -2.5, 2.5],
            lines=aa.GridIrregularGrouped(
                [[(1.0, 1.0), (2.0, 2.0)], [(2.0, 4.0), (5.0, 6.0)]]
            ),
            indexes=[0, 1, 2, 14],
            symmetric_around_centre=False,
        )

        assert path.join(plot_path, "grid1.png") in plot_patch.paths

        plotter = aplt.Plotter(
            output=aplt.Output(path=plot_path, filename="grid2", format="png")
        )

        plotter.plot_grid(
            grid=grid,
            color_array=color_array,
            axis_limits=[-1.5, 1.5, -2.5, 2.5],
            lines=aa.GridIrregularGrouped(
                [[(1.0, 1.0), (2.0, 2.0)], [(2.0, 4.0), (5.0, 6.0)]]
            ),
            indexes=[0, 1, 2, 14],
            symmetric_around_centre=True,
        )

        assert path.join(plot_path, "grid2.png") in plot_patch.paths

        aplt.Grid(
            grid=grid,
            color_array=color_array,
            axis_limits=[-1.5, 1.5, -2.5, 2.5],
            lines=aa.GridIrregularGrouped(
                [[(1.0, 1.0), (2.0, 2.0)], [(2.0, 4.0), (5.0, 6.0)]]
            ),
            indexes=[0, 1, 2, 14],
            symmetric_around_centre=True,
            plotter=aplt.Plotter(
                output=aplt.Output(path=plot_path, filename="grid3", format="png")
            ),
        )

        assert path.join(plot_path, "grid3.png") in plot_patch.paths

    def test__plot_line__works_with_all_extras_included(self, plot_path, plot_patch):

        plotter = aplt.Plotter(
            output=aplt.Output(path=plot_path, filename="line1", format="png")
        )

        plotter.plot_line(
            y=np.array([1.0, 2.0, 3.0]),
            x=np.array([0.5, 1.0, 1.5]),
            plot_axis_type="loglog",
            vertical_lines=[1.0, 2.0],
            label="line0",
            vertical_line_labels=["line1", "line2"],
        )

        assert path.join(plot_path, "line1.png") in plot_patch.paths

        plotter = aplt.Plotter(
            output=aplt.Output(path=plot_path, filename="line2", format="png")
        )

        plotter.plot_line(
            y=np.array([1.0, 2.0, 3.0]),
            x=np.array([0.5, 1.0, 1.5]),
            plot_axis_type="semilogy",
            vertical_lines=[1.0, 2.0],
            label="line0",
            vertical_line_labels=["line1", "line2"],
        )

        assert path.join(plot_path, "line2.png") in plot_patch.paths

        aplt.Line(
            y=np.array([1.0, 2.0, 3.0]),
            x=np.array([0.5, 1.0, 1.5]),
            plot_axis_type="loglog",
            vertical_lines=[1.0, 2.0],
            label="line0",
            vertical_line_labels=["line1", "line2"],
            plotter=aplt.Plotter(
                output=aplt.Output(path=plot_path, filename="line3", format="png")
            ),
        )

        assert path.join(plot_path, "line3.png") in plot_patch.paths

    def test__plot_rectangular_mapper__works_with_all_extras_included(
        self, rectangular_mapper_7x7_3x3, plot_path, plot_patch
    ):

        plotter = aplt.Plotter(
            output=aplt.Output(path=plot_path, filename="mapper1", format="png")
        )

        plotter.plot_rectangular_mapper(
            mapper=rectangular_mapper_7x7_3x3,
            include_pixelization_grid=True,
            include_grid=True,
            include_border=True,
            image_pixel_indexes=[[(0, 0), (0, 1)], [(1, 2)]],
            source_pixel_indexes=[[0, 1], [2]],
        )

        assert path.join(plot_path, "mapper1.png") in plot_patch.paths

        plotter = aplt.Plotter(
            output=aplt.Output(path=plot_path, filename="mapper2", format="png")
        )

        plotter.plot_rectangular_mapper(
            mapper=rectangular_mapper_7x7_3x3,
            include_pixelization_grid=True,
            include_grid=True,
            include_border=True,
            image_pixel_indexes=[[(0, 0), (0, 1)], [(1, 2)]],
            source_pixel_indexes=[[0, 1], [2]],
        )

        assert path.join(plot_path, "mapper2.png") in plot_patch.paths

        aplt.MapperObj(
            mapper=rectangular_mapper_7x7_3x3,
            include=aplt.Include(
                inversion_pixelization_grid=True,
                inversion_grid=True,
                inversion_border=True,
            ),
            image_pixel_indexes=[[(0, 0), (0, 1)], [(1, 2)]],
            source_pixel_indexes=[[0, 1], [2]],
            plotter=aplt.Plotter(
                output=aplt.Output(path=plot_path, filename="mapper3", format="png")
            ),
        )

        assert path.join(plot_path, "mapper3.png") in plot_patch.paths

    def test__plot_voronoi_mapper__works_with_all_extras_included(
        self, voronoi_mapper_9_3x3, plot_path, plot_patch
    ):

        plotter = aplt.Plotter(
            output=aplt.Output(path=plot_path, filename="mapper1", format="png")
        )

        plotter.plot_voronoi_mapper(
            mapper=voronoi_mapper_9_3x3,
            include_pixelization_grid=True,
            include_grid=True,
            include_border=True,
            image_pixel_indexes=[[(0, 0), (0, 1)], [(1, 2)]],
            source_pixel_indexes=[[0, 1], [2]],
        )

        assert path.join(plot_path, "mapper1.png") in plot_patch.paths

        plotter = aplt.Plotter(
            output=aplt.Output(path=plot_path, filename="mapper2", format="png")
        )

        plotter.plot_voronoi_mapper(
            mapper=voronoi_mapper_9_3x3,
            include_pixelization_grid=True,
            include_grid=True,
            include_border=True,
            image_pixel_indexes=[[(0, 0), (0, 1)], [(1, 2)]],
            source_pixel_indexes=[[0, 1], [2]],
        )

        assert path.join(plot_path, "mapper2.png") in plot_patch.paths

        aplt.MapperObj(
            mapper=voronoi_mapper_9_3x3,
            include=aplt.Include(
                inversion_pixelization_grid=True,
                inversion_grid=True,
                inversion_border=True,
            ),
            image_pixel_indexes=[[(0, 0), (0, 1)], [(1, 2)]],
            source_pixel_indexes=[[0, 1], [2]],
            plotter=aplt.Plotter(
                output=aplt.Output(path=plot_path, filename="mapper3", format="png")
            ),
        )

        assert path.join(plot_path, "mapper3.png") in plot_patch.paths


class TestAbstractPlotterNew:
    def test__plotter_with_new_labels__new_labels_if_input__sizes_dont_change(self):

        plotter = aplt.Plotter(
            title=aplt.Title(label="OMG", fontsize=1),
            ylabel=aplt.YLabel(units="hi"),
            xlabel=aplt.XLabel(units="hi2"),
            tickparams=aplt.TickParams(labelsize=2),
        )

        print(plotter.title.kwargs)

        plotter = plotter.plotter_with_new_labels()

        assert plotter.title.kwargs["label"] == "OMG"
        assert plotter.title.kwargs["fontsize"] == 1
        assert plotter.ylabel._units == "hi"
        assert plotter.xlabel._units == "hi2"
        assert plotter.tickparams.kwargs["labelsize"] == 2

        plotter = plotter.plotter_with_new_labels(
            title_label="OMG0",
            title_fontsize=10,
            ylabel_units="hi0",
            xlabel_units="hi20",
            tick_params_labelsize=20,
        )

        assert plotter.title.kwargs["label"] == "OMG0"
        assert plotter.title.kwargs["fontsize"] == 10
        assert plotter.ylabel._units == "hi0"
        assert plotter.xlabel._units == "hi20"
        assert plotter.tickparams.kwargs["labelsize"] == 20

        plotter = plotter.plotter_with_new_labels(title_fontsize=2, title_label="OMG0")

        assert plotter.title.kwargs["label"] == "OMG0"
        assert plotter.title.kwargs["fontsize"] == 2
        assert plotter.ylabel._units == "hi0"
        assert plotter.xlabel._units == "hi20"
        assert plotter.tickparams.kwargs["labelsize"] == 20

    def test__plotter_with_new_cmap__new_labels_if_input__sizes_dont_change(self):

        plotter = aplt.Plotter(
            cmap=aplt.Cmap(
                cmap="cold", norm="log", vmin=0.1, vmax=1.0, linthresh=1.5, linscale=2.0
            )
        )

        assert plotter.cmap.kwargs["cmap"] == "cold"
        assert plotter.cmap.kwargs["norm"] == "log"
        assert plotter.cmap.kwargs["vmin"] == 0.1
        assert plotter.cmap.kwargs["vmax"] == 1.0
        assert plotter.cmap.kwargs["linthresh"] == 1.5
        assert plotter.cmap.kwargs["linscale"] == 2.0

        plotter = plotter.plotter_with_new_cmap(
            cmap="jet", norm="linear", vmin=0.12, vmax=1.2, linthresh=1.2, linscale=2.2
        )

        assert plotter.cmap.kwargs["cmap"] == "jet"
        assert plotter.cmap.kwargs["norm"] == "linear"
        assert plotter.cmap.kwargs["vmin"] == 0.12
        assert plotter.cmap.kwargs["vmax"] == 1.2
        assert plotter.cmap.kwargs["linthresh"] == 1.2
        assert plotter.cmap.kwargs["linscale"] == 2.2

        plotter = plotter.plotter_with_new_cmap(cmap="sand", norm="log", vmin=0.13)

        assert plotter.cmap.kwargs["cmap"] == "sand"
        assert plotter.cmap.kwargs["norm"] == "log"
        assert plotter.cmap.kwargs["vmin"] == 0.13
        assert plotter.cmap.kwargs["vmax"] == 1.2
        assert plotter.cmap.kwargs["linthresh"] == 1.2
        assert plotter.cmap.kwargs["linscale"] == 2.2

    def test__plotter_with_new_outputs__new_outputs_are_setup_correctly_if_input(self):

        plotter = aplt.Plotter(
            output=aplt.Output(path="Path", format="png", filename="file")
        )

        plotter = plotter.plotter_with_new_output()

        assert plotter.output.path == "Path"
        assert plotter.output._format == "png"
        assert plotter.output.format == "png"
        assert plotter.output.filename == "file"

        if path.exists(plotter.output.path):
            shutil.rmtree(plotter.output.path)

        plotter = plotter.plotter_with_new_output(path="Path0", filename="file0")

        if path.exists(plotter.output.path):
            shutil.rmtree(plotter.output.path)

        assert plotter.output.path == "Path0"
        assert plotter.output._format == "png"
        assert plotter.output.format == "png"
        assert plotter.output.filename == "file0"

        if path.exists(plotter.output.path):
            shutil.rmtree(plotter.output.path)

        plotter = plotter.plotter_with_new_output(
            path="Path1", filename="file1", format="fits"
        )

        assert plotter.output.path == "Path1"
        assert plotter.output._format == "fits"
        assert plotter.output.format == "fits"
        assert plotter.output.filename == "file1"

        if path.exists(plotter.output.path):
            shutil.rmtree(plotter.output.path)

    def test__plotter_with_new_units__new_outputs_are_setup_correctly_if_input(self):

        plotter = aplt.Plotter(
            units=aplt.Units(use_scaled=True, in_kpc=True, conversion_factor=1.0)
        )

        assert plotter.units.use_scaled == True
        assert plotter.units.in_kpc == True
        assert plotter.units.conversion_factor == 1.0

        plotter = plotter.plotter_with_new_units(
            use_scaled=False, in_kpc=False, conversion_factor=2.0
        )

        assert plotter.units.use_scaled == False
        assert plotter.units.in_kpc == False
        assert plotter.units.conversion_factor == 2.0

        plotter = plotter.plotter_with_new_units(conversion_factor=3.0)

        assert plotter.units.use_scaled == False
        assert plotter.units.in_kpc == False
        assert plotter.units.conversion_factor == 3.0

    def test__open_and_close_subplot_figures(self):

        plotter = aplt.Plotter()
        plotter.figure.open()

        assert plt.fignum_exists(num=1) == True

        plotter.figure.close()

        assert plt.fignum_exists(num=1) == False

        plotter = aplt.SubPlotter()

        assert plt.fignum_exists(num=1) == False

        plotter.open_subplot_figure(number_subplots=4)

        assert plt.fignum_exists(num=1) == True

        plotter.figure.close()

        assert plt.fignum_exists(num=1) == False


class TestSubPlotter:
    def test__subplot_figsize_for_number_of_subplots(self):

        plotter = aplt.SubPlotter()

        figsize = plotter.get_subplot_figsize(number_subplots=1)

        assert figsize == (18, 8)

        figsize = plotter.get_subplot_figsize(number_subplots=4)

        assert figsize == (13, 10)

        plotter = aplt.SubPlotter(figure=aplt.Figure(figsize=(20, 20)))

        figsize = plotter.get_subplot_figsize(number_subplots=4)

        assert figsize == (20, 20)

    def test__plotter_number_of_subplots(self):

        plotter = aplt.SubPlotter()

        rows, columns = plotter.get_subplot_rows_columns(number_subplots=1)

        assert rows == 1
        assert columns == 2

        rows, columns = plotter.get_subplot_rows_columns(number_subplots=4)

        assert rows == 2
        assert columns == 2


from autoarray.plot.plotter import plotter


class TestDecorator:
    def test__kpc_per_scaled_extacted_from_object_if_available(self):

        dictionary = {"hi": 1}

        kpc_per_scaled = plotter.kpc_per_scaled_of_object_from_dictionary(
            dictionary=dictionary
        )

        assert kpc_per_scaled == None

        class MockObj:
            def __init__(self, param1):

                self.param1 = param1

        obj = MockObj(param1=1)

        dictionary = {"hi": 1, "hello": obj}

        kpc_per_scaled = plotter.kpc_per_scaled_of_object_from_dictionary(
            dictionary=dictionary
        )

        assert kpc_per_scaled == None

        class MockObj:
            def __init__(self, param1, kpc_per_scaled):

                self.param1 = param1
                self.kpc_per_scaled = kpc_per_scaled

        obj = MockObj(param1=1, kpc_per_scaled=2)

        dictionary = {"hi": 1, "hello": obj}

        kpc_per_scaled = plotter.kpc_per_scaled_of_object_from_dictionary(
            dictionary=dictionary
        )

        assert kpc_per_scaled == 2
