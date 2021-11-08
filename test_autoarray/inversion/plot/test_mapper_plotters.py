from os import path
import pytest

import autoarray as aa
import autoarray.plot as aplt

directory = path.dirname(path.realpath(__file__))


@pytest.fixture(name="plot_path")
def make_plot_path_setup():
    return path.join(
        "{}".format(path.dirname(path.realpath(__file__))), "files", "structures"
    )


class TestMapperPlotter:
    def test__visuals_for_data_from_rectangular_mapper(
        self, rectangular_mapper_7x7_3x3
    ):
        include = aplt.Include2D(
            origin=True, mask=True, mapper_data_pixelization_grid=True, border=True
        )

        mapper_plotter = aplt.MapperPlotter(
            mapper=rectangular_mapper_7x7_3x3, include_2d=include
        )

        assert mapper_plotter.extractor_2d.via_mapper_for_data_from(
            mapper=rectangular_mapper_7x7_3x3
        ).origin.in_list == [(0.0, 0.0)]
        assert (
            mapper_plotter.extractor_2d.via_mapper_for_data_from(
                mapper=rectangular_mapper_7x7_3x3
            ).mask
            == rectangular_mapper_7x7_3x3.source_grid_slim.mask
        ).all()
        assert (
            mapper_plotter.extractor_2d.via_mapper_for_data_from(
                mapper=rectangular_mapper_7x7_3x3
            ).grid
            == None
        )
        #  assert visuals.border == (0, 2)

        include = aplt.Include2D(
            origin=False, mask=False, mapper_data_pixelization_grid=False, border=False
        )

        mapper_plotter = aplt.MapperPlotter(
            mapper=rectangular_mapper_7x7_3x3, include_2d=include
        )

        assert (
            mapper_plotter.extractor_2d.via_mapper_for_data_from(
                mapper=rectangular_mapper_7x7_3x3
            ).origin
            == None
        )
        assert (
            mapper_plotter.extractor_2d.via_mapper_for_data_from(
                mapper=rectangular_mapper_7x7_3x3
            ).mask
            == None
        )
        assert (
            mapper_plotter.extractor_2d.via_mapper_for_data_from(
                mapper=rectangular_mapper_7x7_3x3
            ).grid
            == None
        )
        assert (
            mapper_plotter.extractor_2d.via_mapper_for_data_from(
                mapper=rectangular_mapper_7x7_3x3
            ).border
            == None
        )

    def test__visuals_for_data_from_voronoi_mapper(self, voronoi_mapper_9_3x3):

        include = aplt.Include2D(
            origin=True, mask=True, mapper_data_pixelization_grid=True, border=True
        )

        mapper_plotter = aplt.MapperPlotter(
            mapper=voronoi_mapper_9_3x3, include_2d=include
        )

        assert mapper_plotter.extractor_2d.via_mapper_for_data_from(
            mapper=voronoi_mapper_9_3x3
        ).origin.in_list == [(0.0, 0.0)]
        assert (
            mapper_plotter.extractor_2d.via_mapper_for_data_from(
                mapper=voronoi_mapper_9_3x3
            ).mask
            == voronoi_mapper_9_3x3.source_grid_slim.mask
        ).all()
        assert (
            mapper_plotter.extractor_2d.via_mapper_for_data_from(
                mapper=voronoi_mapper_9_3x3
            ).pixelization_grid
            == aa.Grid2D.uniform(shape_native=(2, 2), pixel_scales=0.1)
        ).all()
        #      assert visuals.border.shape == (0, 2)

        include = aplt.Include2D(
            origin=False, mask=False, mapper_data_pixelization_grid=False, border=False
        )

        mapper_plotter = aplt.MapperPlotter(
            mapper=voronoi_mapper_9_3x3, include_2d=include
        )

        assert (
            mapper_plotter.extractor_2d.via_mapper_for_data_from(
                mapper=voronoi_mapper_9_3x3
            ).origin
            == None
        )
        assert (
            mapper_plotter.extractor_2d.via_mapper_for_data_from(
                mapper=voronoi_mapper_9_3x3
            ).mask
            == None
        )
        assert (
            mapper_plotter.extractor_2d.via_mapper_for_data_from(
                mapper=voronoi_mapper_9_3x3
            ).grid
            == None
        )
        assert (
            mapper_plotter.extractor_2d.via_mapper_for_data_from(
                mapper=voronoi_mapper_9_3x3
            ).pixelization_grid
            == None
        )
        assert (
            mapper_plotter.extractor_2d.via_mapper_for_data_from(
                mapper=voronoi_mapper_9_3x3
            ).border
            == None
        )

    def test__visuals_for_source_from_rectangular_mapper(
        self, rectangular_mapper_7x7_3x3
    ):

        include = aplt.Include2D(
            origin=True,
            mapper_source_grid_slim=True,
            mapper_source_pixelization_grid=True,
            border=True,
        )

        mapper_plotter = aplt.MapperPlotter(
            mapper=rectangular_mapper_7x7_3x3, include_2d=include
        )

        assert mapper_plotter.visuals_2d.origin == None
        assert mapper_plotter.extractor_2d.via_mapper_for_source_from(
            mapper=rectangular_mapper_7x7_3x3
        ).origin.in_list == [(0.0, 0.0)]
        assert (
            mapper_plotter.extractor_2d.via_mapper_for_source_from(
                mapper=rectangular_mapper_7x7_3x3
            ).grid
            == rectangular_mapper_7x7_3x3.source_grid_slim
        ).all()
        assert (
            mapper_plotter.extractor_2d.via_mapper_for_source_from(
                mapper=rectangular_mapper_7x7_3x3
            ).pixelization_grid
            == rectangular_mapper_7x7_3x3.source_pixelization_grid
        ).all()
        assert (
            mapper_plotter.extractor_2d.via_mapper_for_source_from(
                mapper=rectangular_mapper_7x7_3x3
            ).border
            == rectangular_mapper_7x7_3x3.source_grid_slim.sub_border_grid
        ).all()

        include = aplt.Include2D(
            origin=False,
            border=False,
            mapper_source_grid_slim=False,
            mapper_source_pixelization_grid=False,
        )

        mapper_plotter = aplt.MapperPlotter(
            mapper=rectangular_mapper_7x7_3x3, include_2d=include
        )

        assert (
            mapper_plotter.extractor_2d.via_mapper_for_source_from(
                mapper=rectangular_mapper_7x7_3x3
            ).origin
            == None
        )
        assert (
            mapper_plotter.extractor_2d.via_mapper_for_source_from(
                mapper=rectangular_mapper_7x7_3x3
            ).grid
            == None
        )
        assert (
            mapper_plotter.extractor_2d.via_mapper_for_source_from(
                mapper=rectangular_mapper_7x7_3x3
            ).pixelization_grid
            == None
        )
        assert (
            mapper_plotter.extractor_2d.via_mapper_for_source_from(
                mapper=rectangular_mapper_7x7_3x3
            ).border
            == None
        )

    def test__visuals_for_source_from_voronoi_mapper(self, voronoi_mapper_9_3x3):

        include = aplt.Include2D(
            origin=True,
            border=True,
            mapper_source_grid_slim=True,
            mapper_source_pixelization_grid=True,
        )

        mapper_plotter = aplt.MapperPlotter(
            mapper=voronoi_mapper_9_3x3, include_2d=include
        )

        assert mapper_plotter.visuals_2d.origin == None
        assert mapper_plotter.extractor_2d.via_mapper_for_source_from(
            mapper=voronoi_mapper_9_3x3
        ).origin.in_list == [(0.0, 0.0)]
        assert (
            mapper_plotter.extractor_2d.via_mapper_for_source_from(
                mapper=voronoi_mapper_9_3x3
            ).grid
            == voronoi_mapper_9_3x3.source_grid_slim
        ).all()
        assert (
            mapper_plotter.extractor_2d.via_mapper_for_source_from(
                mapper=voronoi_mapper_9_3x3
            ).pixelization_grid
            == voronoi_mapper_9_3x3.source_pixelization_grid
        ).all()
        assert (
            mapper_plotter.extractor_2d.via_mapper_for_source_from(
                mapper=voronoi_mapper_9_3x3
            ).border
            == voronoi_mapper_9_3x3.source_grid_slim.sub_border_grid
        ).all()

        include = aplt.Include2D(
            origin=False, border=False, mapper_source_pixelization_grid=False
        )

        mapper_plotter = aplt.MapperPlotter(
            mapper=voronoi_mapper_9_3x3, include_2d=include
        )

        assert (
            mapper_plotter.extractor_2d.via_mapper_for_source_from(
                mapper=voronoi_mapper_9_3x3
            ).origin
            == None
        )
        assert (
            mapper_plotter.extractor_2d.via_mapper_for_source_from(
                mapper=voronoi_mapper_9_3x3
            ).grid
            == None
        )
        assert (
            mapper_plotter.extractor_2d.via_mapper_for_source_from(
                mapper=voronoi_mapper_9_3x3
            ).border
            == None
        )

    def test__plot_rectangular_mapper__works_with_all_extras_included(
        self, rectangular_mapper_7x7_3x3, plot_path, plot_patch
    ):

        mapper_plotter = aplt.MapperPlotter(
            mapper=rectangular_mapper_7x7_3x3,
            visuals_2d=aplt.Visuals2D(
                indexes=[[(0, 0), (0, 1)], [(1, 2)]], pixelization_indexes=[[0, 1], [2]]
            ),
            mat_plot_2d=aplt.MatPlot2D(
                output=aplt.Output(path=plot_path, filename="mapper1", format="png")
            ),
        )

        mapper_plotter.figure_2d()

        assert path.join(plot_path, "mapper1.png") in plot_patch.paths

        mapper_plotter = aplt.MapperPlotter(
            mapper=rectangular_mapper_7x7_3x3,
            visuals_2d=aplt.Visuals2D(
                indexes=[[(0, 0), (0, 1)], [(1, 2)]], pixelization_indexes=[[0, 1], [2]]
            ),
            mat_plot_2d=aplt.MatPlot2D(
                output=aplt.Output(path=plot_path, filename="mapper2", format="png")
            ),
        )

        mapper_plotter.figure_2d()

        assert path.join(plot_path, "mapper2.png") in plot_patch.paths

        mapper_plotter = aplt.MapperPlotter(
            mapper=rectangular_mapper_7x7_3x3,
            visuals_2d=aplt.Visuals2D(
                indexes=[[(0, 0), (0, 1)], [(1, 2)]], pixelization_indexes=[[0, 1], [2]]
            ),
            mat_plot_2d=aplt.MatPlot2D(
                output=aplt.Output(path=plot_path, filename="mapper3", format="png")
            ),
            include_2d=aplt.Include2D(
                origin=True, mapper_source_pixelization_grid=True
            ),
        )

        mapper_plotter.figure_2d()

        assert path.join(plot_path, "mapper3.png") in plot_patch.paths

    def test__plot_voronoi_mapper__works_with_all_extras_included(
        self, voronoi_mapper_9_3x3, plot_path, plot_patch
    ):

        mapper_plotter = aplt.MapperPlotter(
            mapper=voronoi_mapper_9_3x3,
            visuals_2d=aplt.Visuals2D(
                indexes=[[(0, 0), (0, 1)], [(1, 2)]], pixelization_indexes=[[0, 1], [2]]
            ),
            mat_plot_2d=aplt.MatPlot2D(
                output=aplt.Output(path=plot_path, filename="mapper1", format="png")
            ),
        )

        mapper_plotter.figure_2d()

        assert path.join(plot_path, "mapper1.png") in plot_patch.paths

        mapper_plotter = aplt.MapperPlotter(
            visuals_2d=aplt.Visuals2D(
                indexes=[[(0, 0), (0, 1)], [(1, 2)]], pixelization_indexes=[[0, 1], [2]]
            ),
            mapper=voronoi_mapper_9_3x3,
            mat_plot_2d=aplt.MatPlot2D(
                output=aplt.Output(path=plot_path, filename="mapper2", format="png")
            ),
        )

        mapper_plotter.figure_2d()

        assert path.join(plot_path, "mapper2.png") in plot_patch.paths

        mapper_plotter = aplt.MapperPlotter(
            visuals_2d=aplt.Visuals2D(
                indexes=[[(0, 0), (0, 1)], [(1, 2)]], pixelization_indexes=[[0, 1], [2]]
            ),
            mapper=voronoi_mapper_9_3x3,
            mat_plot_2d=aplt.MatPlot2D(
                output=aplt.Output(path=plot_path, filename="mapper3", format="png")
            ),
        )

        mapper_plotter.figure_2d()

        assert path.join(plot_path, "mapper3.png") in plot_patch.paths

    def test__image_and_mapper_subplot_is_output_for_all_mappers(
        self,
        imaging_7x7,
        rectangular_mapper_7x7_3x3,
        voronoi_mapper_9_3x3,
        plot_path,
        plot_patch,
    ):

        mapper_plotter = aplt.MapperPlotter(
            mapper=rectangular_mapper_7x7_3x3,
            visuals_2d=aplt.Visuals2D(
                indexes=[[(0, 0), (0, 1)], [(1, 2)]], pixelization_indexes=[[0, 1], [2]]
            ),
            mat_plot_2d=aplt.MatPlot2D(
                output=aplt.Output(path=plot_path, format="png")
            ),
            include_2d=aplt.Include2D(mapper_source_pixelization_grid=True),
        )

        mapper_plotter.subplot_image_and_mapper(image=imaging_7x7.image)

        assert path.join(plot_path, "subplot_image_and_mapper.png") in plot_patch.paths

        mapper_plotter.subplot_image_and_mapper(image=imaging_7x7.image)

        assert path.join(plot_path, "subplot_image_and_mapper.png") in plot_patch.paths
