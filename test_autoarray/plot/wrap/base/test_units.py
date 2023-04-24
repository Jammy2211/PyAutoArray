import autoarray.plot as aplt


def test__loads_values_from_config_if_not_manually_input():
    units = aplt.Units()

    assert units.use_scaled is True
    assert units.ticks_convert_factor == None

    units = aplt.Units(ticks_convert_factor=2.0)

    assert units.use_scaled is True
    assert units.ticks_convert_factor == 2.0
