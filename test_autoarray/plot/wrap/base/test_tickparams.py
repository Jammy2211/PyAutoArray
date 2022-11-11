import autoarray.plot as aplt


def test__loads_values_from_config_if_not_manually_input():
    tick_params = aplt.TickParams()

    assert tick_params.config_dict["labelsize"] == 16

    tick_params = aplt.TickParams(labelsize=24)
    assert tick_params.config_dict["labelsize"] == 24

    tick_params = aplt.TickParams()
    tick_params.is_for_subplot = True

    assert tick_params.config_dict["labelsize"] == 10

    tick_params = aplt.TickParams(labelsize=25)
    tick_params.is_for_subplot = True

    assert tick_params.config_dict["labelsize"] == 25
