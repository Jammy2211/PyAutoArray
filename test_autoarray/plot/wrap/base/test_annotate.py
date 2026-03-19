import autoarray.plot as aplt


def test__loads_values_from_config_if_not_manually_input():
    title = aplt.Annotate()

    assert title.config_dict["fontsize"] == 16

    title = aplt.Annotate(fontsize=1)

    assert title.config_dict["fontsize"] == 1

    title = aplt.Annotate(fontsize=2)

    assert title.config_dict["fontsize"] == 2
