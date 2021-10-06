from os import path
import numpy as np
import pytest

import autoarray as aa

test_grid_dir = path.join(
    "{}".format(path.dirname(path.realpath(__file__))), "files", "grids"
)


class TestAPI:
    def test__manual(self):

        grid_1d = aa.Grid1D.manual_native(
            grid=[1.0, 2.0, 3.0, 4.0], pixel_scales=1.0, sub_size=2
        )

        assert type(grid_1d) == aa.Grid1D
        assert (grid_1d.native == np.array([1.0, 2.0, 3.0, 4.0])).all()
        assert (grid_1d.slim == np.array([1.0, 2.0, 3.0, 4.0])).all()
        assert (grid_1d.binned.native == np.array([1.5, 3.5])).all()
        assert (grid_1d.binned == np.array([1.5, 3.5])).all()
        assert grid_1d.pixel_scales == (1.0,)
        assert grid_1d.origin == (0.0,)

        grid_1d = aa.Grid1D.manual_slim(
            grid=[1.0, 2.0, 3.0, 4.0], pixel_scales=1.0, sub_size=2
        )

        assert type(grid_1d) == aa.Grid1D
        assert (grid_1d.native == np.array([1.0, 2.0, 3.0, 4.0])).all()
        assert (grid_1d.slim == np.array([1.0, 2.0, 3.0, 4.0])).all()
        assert (grid_1d.binned.native == np.array([1.5, 3.5])).all()
        assert (grid_1d.binned == np.array([1.5, 3.5])).all()
        assert grid_1d.pixel_scales == (1.0,)
        assert grid_1d.origin == (0.0,)

        grid_1d = aa.Grid1D.manual_slim(
            grid=[1.0, 2.0, 3.0, 4.0], pixel_scales=1.0, sub_size=2, origin=(1.0,)
        )

        assert type(grid_1d) == aa.Grid1D
        assert (grid_1d.native == np.array([1.0, 2.0, 3.0, 4.0])).all()
        assert (grid_1d.slim == np.array([1.0, 2.0, 3.0, 4.0])).all()
        assert (grid_1d.binned.native == np.array([1.5, 3.5])).all()
        assert (grid_1d.binned == np.array([1.5, 3.5])).all()
        assert grid_1d.pixel_scales == (1.0,)
        assert grid_1d.origin == (1.0,)

    def test__manual_mask(self):

        mask = aa.Mask1D.unmasked(shape_slim=(2,), pixel_scales=1.0, sub_size=2)
        grid = aa.Grid1D.manual_mask(grid=[1.0, 2.0, 3.0, 4.0], mask=mask)

        assert type(grid) == aa.Grid1D
        assert (grid.native == np.array([1.0, 2.0, 3.0, 4.0])).all()
        assert (grid.slim == np.array([1.0, 2.0, 3.0, 4.0])).all()
        assert (grid.binned.native == np.array([1.5, 3.5])).all()
        assert (grid.binned == np.array([1.5, 3.5])).all()
        assert grid.pixel_scales == (1.0,)
        assert grid.origin == (0.0,)

        mask = aa.Mask1D.manual(mask=[True, False, False], pixel_scales=1.0, sub_size=2)
        grid = aa.Grid1D.manual_mask(grid=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], mask=mask)

        assert type(grid) == aa.Grid1D
        assert (grid.native == np.array([0.0, 0.0, 3.0, 4.0, 5.0, 6.0])).all()
        assert (grid.slim == np.array([3.0, 4.0, 5.0, 6.0])).all()
        assert (grid.binned.native == np.array([0.0, 3.5, 5.5])).all()
        assert (grid.binned == np.array([3.5, 5.5])).all()
        assert grid.pixel_scales == (1.0,)
        assert grid.origin == (0.0,)

        assert (grid.slim.native == np.array([0.0, 0.0, 3.0, 4.0, 5.0, 6.0])).all()
        assert (grid.native.slim == np.array([3.0, 4.0, 5.0, 6.0])).all()

    def test__from_mask(self):

        mask = aa.Mask1D.unmasked(shape_slim=(4,), pixel_scales=1.0, sub_size=1)
        grid = aa.Grid1D.from_mask(mask=mask)

        assert type(grid) == aa.Grid1D
        assert (grid.native == np.array([-1.5, -0.5, 0.5, 1.5])).all()
        assert (grid.slim == np.array([-1.5, -0.5, 0.5, 1.5])).all()
        assert (grid.binned.native == np.array([-1.5, -0.5, 0.5, 1.5])).all()
        assert (grid.binned == np.array([-1.5, -0.5, 0.5, 1.5])).all()
        assert grid.pixel_scales == (1.0,)
        assert grid.origin == (0.0,)

        mask = aa.Mask1D.manual(mask=[True, False], pixel_scales=1.0, sub_size=2)
        grid = aa.Grid1D.from_mask(mask=mask)

        assert type(grid) == aa.Grid1D
        assert (grid.native == np.array([0.0, 0.0, 0.25, 0.75])).all()
        assert (grid.slim == np.array([0.25, 0.75])).all()
        assert (grid.binned.native == np.array([0.0, 0.5])).all()
        assert (grid.binned == np.array([0.5])).all()
        assert grid.pixel_scales == (1.0,)
        assert grid.origin == (0.0,)

        mask = aa.Mask1D.manual(
            mask=[True, False, False, False], pixel_scales=1.0, sub_size=1
        )
        grid = aa.Grid1D.from_mask(mask=mask)

        assert type(grid) == aa.Grid1D
        assert (grid.native == np.array([0.0, -0.5, 0.5, 1.5])).all()
        assert (grid.slim == np.array([-0.5, 0.5, 1.5])).all()
        assert (grid.binned.native == np.array([0.0, -0.5, 0.5, 1.5])).all()
        assert (grid.binned == np.array([-0.5, 0.5, 1.5])).all()
        assert grid.pixel_scales == (1.0,)
        assert grid.origin == (0.0,)

    def test__uniform(self):

        grid_1d = aa.Grid1D.uniform(
            shape_native=(2,), pixel_scales=1.0, sub_size=1, origin=(0.0,)
        )

        assert type(grid_1d) == aa.Grid1D
        assert (grid_1d.native == np.array([-0.5, 0.5])).all()
        assert (grid_1d.slim == np.array([-0.5, 0.5])).all()
        assert (grid_1d.binned.native == np.array([-0.5, 0.5])).all()
        assert grid_1d.pixel_scales == (1.0,)
        assert grid_1d.origin == (0.0,)

        grid_1d = aa.Grid1D.uniform(
            shape_native=(2,), pixel_scales=1.0, sub_size=2, origin=(0.0,)
        )

        assert type(grid_1d) == aa.Grid1D
        assert (grid_1d.native == np.array([-0.75, -0.25, 0.25, 0.75])).all()
        assert (grid_1d.slim == np.array([-0.75, -0.25, 0.25, 0.75])).all()
        assert (grid_1d.binned.native == np.array([-0.5, 0.5])).all()
        assert grid_1d.pixel_scales == (1.0,)
        assert grid_1d.origin == (0.0,)

        grid_1d = aa.Grid1D.uniform(
            shape_native=(2,), pixel_scales=1.0, sub_size=1, origin=(1.0,)
        )

        assert type(grid_1d) == aa.Grid1D
        assert (grid_1d.native == np.array([0.5, 1.5])).all()
        assert (grid_1d.slim == np.array([0.5, 1.5])).all()
        assert (grid_1d.binned.native == np.array([0.5, 1.5])).all()
        assert grid_1d.pixel_scales == (1.0,)
        assert grid_1d.origin == (1.0,)

    def test__uniform_from_zero(self):

        grid_1d = aa.Grid1D.uniform_from_zero(
            shape_native=(2,), pixel_scales=1.0, sub_size=1
        )

        assert type(grid_1d) == aa.Grid1D
        assert (grid_1d.native == np.array([0.0, 1.0])).all()
        assert (grid_1d.slim == np.array([0.0, 1.0])).all()
        assert (grid_1d.binned.native == np.array([0.0, 1.0])).all()
        assert grid_1d.pixel_scales == (1.0,)
        assert grid_1d.origin == (0.0,)

        grid_1d = aa.Grid1D.uniform_from_zero(
            shape_native=(3,), pixel_scales=1.5, sub_size=1
        )

        assert type(grid_1d) == aa.Grid1D
        assert (grid_1d.native == np.array([0.0, 1.5, 3.0])).all()
        assert (grid_1d.slim == np.array([0.0, 1.5, 3.0])).all()
        assert (grid_1d.binned.native == np.array([0.0, 1.5, 3.0])).all()
        assert grid_1d.pixel_scales == (1.5,)
        assert grid_1d.origin == (0.0,)

        grid_1d = aa.Grid1D.uniform_from_zero(
            shape_native=(3,), pixel_scales=1.5, sub_size=2
        )

        assert type(grid_1d) == aa.Grid1D
        assert (grid_1d.native == np.array([0.0, 0.75, 1.5, 2.25, 3.0, 3.75])).all()
        assert (grid_1d.slim == np.array([0, 0.75, 1.5, 2.25, 3.0, 3.75])).all()
        assert (grid_1d.binned.native == np.array([0.375, 1.875, 3.375])).all()
        assert grid_1d.pixel_scales == (1.5,)
        assert grid_1d.origin == (0.0,)

    def test__recursive_shape_storage(self):

        mask = aa.Mask1D.unmasked(shape_slim=(2,), pixel_scales=1.0, sub_size=2)
        grid = aa.Grid1D.manual_mask(grid=[1.0, 2.0, 3.0, 4.0], mask=mask)

        assert (grid.slim.native.slim == np.array([1.0, 2.0, 3.0, 4.0])).all()
        assert (grid.native.slim.native == np.array([1.0, 2.0, 3.0, 4.0])).all()

        mask = aa.Mask1D.manual(mask=[True, False, False], pixel_scales=1.0, sub_size=2)
        grid = aa.Grid1D.manual_mask(grid=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], mask=mask)

        assert (grid.slim.native.slim == np.array([3.0, 4.0, 5.0, 6.0])).all()
        assert (
            grid.native.slim.native == np.array([0.0, 0.0, 3.0, 4.0, 5.0, 6.0])
        ).all()


class TestGrid1D:
    def test__grid_2d_with_other_value_out(self):

        grid_1d = aa.Grid1D.manual_native(
            grid=[1.0, 2.0, 3.0, 4.0], pixel_scales=1.0, sub_size=1
        )

        grid_2d = grid_1d.project_to_radial_grid_2d(angle=0.0)

        assert type(grid_2d) == aa.Grid2DIrregular
        assert grid_2d.slim == pytest.approx(
            np.array([[0.0, 1.0], [0.0, 2.0], [0.0, 3.0], [0.0, 4.0]]), 1.0e-4
        )

        grid_2d = grid_1d.project_to_radial_grid_2d(angle=90.0)

        assert grid_2d.slim == pytest.approx(
            np.array([[-1.0, 0.0], [-2.0, 0.0], [-3.0, 0.0], [-4.0, 0.0]]), 1.0e-4
        )

        grid_2d = grid_1d.project_to_radial_grid_2d(angle=45.0)

        assert grid_2d.slim == pytest.approx(
            np.array(
                [
                    [-0.5 * np.sqrt(2), 0.5 * np.sqrt(2)],
                    [-1.0 * np.sqrt(2), 1.0 * np.sqrt(2)],
                    [-1.5 * np.sqrt(2), 1.5 * np.sqrt(2)],
                    [-2.0 * np.sqrt(2), 2.0 * np.sqrt(2)],
                ]
            ),
            1.0e-4,
        )

    def test__structure_2d_from__maps_numpy_array_to__auto_array_or_grid(self):

        mask = np.array([True, False, False, True])

        mask = aa.Mask1D.manual(mask=mask, pixel_scales=(1.0,), sub_size=1)

        grid_1d = aa.Grid1D.from_mask(mask=mask)

        result = grid_1d.structure_2d_from(result=np.array([1.0, 2.0]))

        assert isinstance(result, aa.Array1D)
        assert (result.native == np.array([0.0, 1.0, 2.0, 0.0])).all()

        result = grid_1d.structure_2d_from(result=np.array([[1.0, 1.0], [2.0, 2.0]]))

        assert isinstance(result, aa.Grid2D)
        assert (
            result.native
            == np.array([[[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [0.0, 0.0]]])
        ).all()

    def test__structure_2d_list_from__maps_list_to_auto_arrays_or_grids(self):

        mask = np.array([True, False, False, True])

        mask = aa.Mask1D.manual(mask=mask, pixel_scales=(1.0,), sub_size=1)

        grid_1d = aa.Grid1D.from_mask(mask=mask)

        result = grid_1d.structure_2d_list_from(result_list=[np.array([1.0, 2.0])])

        assert isinstance(result[0], aa.Array1D)
        assert (result[0].native == np.array([0.0, 1.0, 2.0, 0.0])).all()

        result = grid_1d.structure_2d_list_from(
            result_list=[np.array([[1.0, 1.0], [2.0, 2.0]])]
        )

        assert isinstance(result[0], aa.Grid2D)
        assert (
            result[0].native
            == np.array([[[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [0.0, 0.0]]])
        ).all()
