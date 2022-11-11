import autoarray.plot as aplt


def test__loads_values_from_config_if_not_manually_input():

    units = aplt.Units()

    assert units.use_scaled is True
    assert units.in_kpc is False
    assert units.conversion_factor == None

    units = aplt.Units(in_kpc=True, conversion_factor=2.0)

    assert units.use_scaled is True
    assert units.in_kpc is True
    assert units.conversion_factor == 2.0
