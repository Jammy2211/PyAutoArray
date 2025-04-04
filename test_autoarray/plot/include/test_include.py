import autoarray.plot as aplt


def test__loads_default_values_from_config_if_not_input():
    include = aplt.Include2D()

    assert include.origin is True
    assert include.mask == True
    assert include.border is False
    assert include.parallel_overscan is True
    assert include.serial_prescan is True
    assert include.serial_overscan is False

    include = aplt.Include2D(origin=False, border=False, serial_overscan=True)

    assert include.origin is False
    assert include.mask == True
    assert include.border is False
    assert include.parallel_overscan is True
    assert include.serial_prescan is True
    assert include.serial_overscan is True
