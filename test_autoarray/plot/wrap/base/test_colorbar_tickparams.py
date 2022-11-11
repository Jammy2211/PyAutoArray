import autoarray.plot as aplt


def test__loads_values_from_config_if_not_manually_input():

    colorbar_tickparams = aplt.ColorbarTickParams()

    assert colorbar_tickparams.config_dict["labelsize"] == 1

    colorbar_tickparams = aplt.ColorbarTickParams(labelsize=20)

    assert colorbar_tickparams.config_dict["labelsize"] == 20

    colorbar_tickparams = aplt.ColorbarTickParams()
    colorbar_tickparams.is_for_subplot = True

    assert colorbar_tickparams.config_dict["labelsize"] == 1

    colorbar_tickparams = aplt.ColorbarTickParams(labelsize=10)
    colorbar_tickparams.is_for_subplot = True

    assert colorbar_tickparams.config_dict["labelsize"] == 10
