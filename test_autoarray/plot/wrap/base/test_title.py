import autoarray.plot as aplt


def test__loads_values_from_config_if_not_manually_input():
    title = aplt.Title()

    assert title.manual_label == None
    assert title.config_dict["fontsize"] == 11

    title = aplt.Title(label="OMG", fontsize=1)

    assert title.manual_label == "OMG"
    assert title.config_dict["fontsize"] == 1

    title = aplt.Title()
    title.is_for_subplot = True

    assert title.manual_label == None
    assert title.config_dict["fontsize"] == 15

    title = aplt.Title(label="OMG2", fontsize=2)
    title.is_for_subplot = True

    assert title.manual_label == "OMG2"
    assert title.config_dict["fontsize"] == 2
