import autoarray as aa
import autoarray.plot as aplt


def test__loads_values_from_config_if_not_manually_input():
    axis = aplt.Axis()

    assert axis.config_dict["emit"] is True

    axis = aplt.Axis(emit=False)

    assert axis.config_dict["emit"] is False

    axis = aplt.Axis()
    axis.is_for_subplot = True

    assert axis.config_dict["emit"] is False

    axis = aplt.Axis(emit=True)
    axis.is_for_subplot = True

    assert axis.config_dict["emit"] is True


def test__sets_axis_correct_for_different_settings():
    axis = aplt.Axis(symmetric_source_centre=False)

    axis.set(extent=[0.1, 0.2, 0.3, 0.4])

    axis = aplt.Axis(symmetric_source_centre=True)

    grid = aa.Grid2D.uniform(shape_native=(3, 3), pixel_scales=2.0)

    axis.set(extent=[0.1, 0.2, 0.3, 0.4], grid=grid)
