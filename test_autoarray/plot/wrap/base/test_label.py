import autoarray.plot as aplt


def test__ylabel__loads_values_from_config_if_not_manually_input():
    ylabel = aplt.YLabel()

    assert ylabel.config_dict["fontsize"] == 16

    ylabel = aplt.YLabel(fontsize=11)

    assert ylabel.config_dict["fontsize"] == 11

    ylabel = aplt.YLabel(fontsize=12)

    assert ylabel.config_dict["fontsize"] == 12


def test__xlabel__loads_values_from_config_if_not_manually_input():
    xlabel = aplt.XLabel()

    assert xlabel.config_dict["fontsize"] == 16

    xlabel = aplt.XLabel(fontsize=11)

    assert xlabel.config_dict["fontsize"] == 11

    xlabel = aplt.XLabel(fontsize=12)

    assert xlabel.config_dict["fontsize"] == 12
