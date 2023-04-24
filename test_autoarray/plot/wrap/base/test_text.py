import autoarray.plot as aplt


def test__loads_values_from_config_if_not_manually_input():
    title = aplt.Text()

    assert title.config_dict["fontsize"] == 16

    title = aplt.Text(fontsize=1)

    assert title.config_dict["fontsize"] == 1

    title = aplt.Text()
    title.is_for_subplot = True

    assert title.config_dict["fontsize"] == 10

    title = aplt.Text(fontsize=2)
    title.is_for_subplot = True

    assert title.config_dict["fontsize"] == 2
