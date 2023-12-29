import autoarray.plot as aplt


def test__from_config_or_via_manual_input():

    # Testing for config loading, could be any matplot object but use GridScatter as example

    grid_scatter = aplt.GridScatter()

    assert grid_scatter.config_dict["marker"] == "x"
    assert grid_scatter.config_dict["c"] == "y"

    grid_scatter = aplt.GridScatter(marker="x")

    assert grid_scatter.config_dict["marker"] == "x"
    assert grid_scatter.config_dict["c"] == "y"

    grid_scatter = aplt.GridScatter()
    grid_scatter.is_for_subplot = True

    assert grid_scatter.config_dict["marker"] == "."
    assert grid_scatter.config_dict["c"] == "r"

    grid_scatter = aplt.GridScatter(c=["r", "b"])
    grid_scatter.is_for_subplot = True

    assert grid_scatter.config_dict["marker"] == "."
    assert grid_scatter.config_dict["c"] == ["r", "b"]