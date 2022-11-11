import autoarray as aa
import autoarray.plot as aplt


def test__from_config_or_via_manual_input():

    grid_errorbar = aplt.GridErrorbar()

    assert grid_errorbar.config_dict["marker"] == "o"
    assert grid_errorbar.config_dict["c"] == "k"

    grid_errorbar = aplt.GridErrorbar(marker="x")

    assert grid_errorbar.config_dict["marker"] == "x"
    assert grid_errorbar.config_dict["c"] == "k"

    grid_errorbar = aplt.GridErrorbar()
    grid_errorbar.is_for_subplot = True

    assert grid_errorbar.config_dict["marker"] == "."
    assert grid_errorbar.config_dict["c"] == "b"

    grid_errorbar = aplt.GridErrorbar(c=["r", "b"])
    grid_errorbar.is_for_subplot = True

    assert grid_errorbar.config_dict["marker"] == "."
    assert grid_errorbar.config_dict["c"] == ["r", "b"]


def test__errorbar_grid():

    errorbar = aplt.GridErrorbar(marker="x", c="k")

    errorbar.errorbar_grid(
        grid=aa.Grid2D.uniform(shape_native=(3, 3), pixel_scales=1.0)
    )

    errorbar = aplt.GridErrorbar(marker="x", c="k")

    errorbar.errorbar_grid(
        grid=aa.Grid2D.uniform(shape_native=(3, 3), pixel_scales=1.0),
        y_errors=[1.0] * 9,
        x_errors=[1.0] * 9,
    )


def test__errorbar_coordinates():

    errorbar = aplt.GridErrorbar(marker="x", c="k")

    errorbar.errorbar_grid_list(
        grid_list=[aa.Grid2DIrregular([(1.0, 1.0), (2.0, 2.0)])],
        y_errors=[1.0] * 2,
        x_errors=[1.0] * 2,
    )
