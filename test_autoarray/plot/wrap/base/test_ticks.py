import autoarray as aa
import autoarray.plot as aplt

from autoarray.plot.wrap.base.ticks import LabelMaker


def test__labels_with_suffix_from():
    label_maker = LabelMaker(
        tick_values=[1.0, 2.0, 3.0],
        min_value=1.0,
        max_value=3.0,
        units=aplt.Units(use_scaled=False),
        manual_suffix="",
    )

    labels = label_maker.with_appended_suffix(labels=["hi", "hello"])

    assert labels == ["hi", "hello"]

    label_maker = LabelMaker(
        tick_values=[1.0, 2.0, 3.0],
        min_value=1.0,
        max_value=3.0,
        units=aplt.Units(use_scaled=False),
        manual_suffix="11",
    )

    labels = label_maker.with_appended_suffix(labels=["hi", "hello"])

    assert labels == ["hi11", "hello11"]


def test__yticks_loads_values_from_config_if_not_manually_input():
    yticks = aplt.YTicks()

    assert yticks.config_dict["fontsize"] == 16
    assert yticks.manual_values == None
    assert yticks.manual_values == None

    yticks = aplt.YTicks(fontsize=24, manual_values=[1.0, 2.0])

    assert yticks.config_dict["fontsize"] == 24
    assert yticks.manual_values == [1.0, 2.0]

    yticks = aplt.YTicks()
    yticks.is_for_subplot = True

    assert yticks.config_dict["fontsize"] == 10
    assert yticks.manual_values == None

    yticks = aplt.YTicks(fontsize=25, manual_values=[1.0, 2.0])
    yticks.is_for_subplot = True

    assert yticks.config_dict["fontsize"] == 25
    assert yticks.manual_values == [1.0, 2.0]


def test__yticks__set():
    array = aa.Array2D.ones(shape_native=(2, 2), pixel_scales=1.0)
    units = aplt.Units(use_scaled=True, ticks_convert_factor=None)

    yticks = aplt.YTicks(fontsize=34)
    zoom = aa.Zoom2D(mask=array.mask)
    array_zoom = zoom.array_2d_from(array=array, buffer=0)
    extent = array_zoom.geometry.extent
    yticks.set(min_value=extent[2], max_value=extent[3], units=units)

    yticks = aplt.YTicks(fontsize=34)
    units = aplt.Units(use_scaled=False, ticks_convert_factor=None)
    yticks.set(min_value=extent[2], max_value=extent[3], pixels=2, units=units)

    yticks = aplt.YTicks(fontsize=34)
    units = aplt.Units(use_scaled=True, ticks_convert_factor=2.0)
    yticks.set(min_value=extent[2], max_value=extent[3], units=units)

    yticks = aplt.YTicks(fontsize=34)
    units = aplt.Units(use_scaled=False, ticks_convert_factor=2.0)
    yticks.set(min_value=extent[2], max_value=extent[3], pixels=2, units=units)


def test__xticks_loads_values_from_config_if_not_manually_input():
    xticks = aplt.XTicks()

    assert xticks.config_dict["fontsize"] == 17
    assert xticks.manual_values == None
    assert xticks.manual_values == None

    xticks = aplt.XTicks(fontsize=24, manual_values=[1.0, 2.0])

    assert xticks.config_dict["fontsize"] == 24
    assert xticks.manual_values == [1.0, 2.0]

    xticks = aplt.XTicks()
    xticks.is_for_subplot = True

    assert xticks.config_dict["fontsize"] == 11
    assert xticks.manual_values == None

    xticks = aplt.XTicks(fontsize=25, manual_values=[1.0, 2.0])
    xticks.is_for_subplot = True

    assert xticks.config_dict["fontsize"] == 25
    assert xticks.manual_values == [1.0, 2.0]


def test__xticks__set():
    array = aa.Array2D.ones(shape_native=(2, 2), pixel_scales=1.0)
    units = aplt.Units(use_scaled=True, ticks_convert_factor=None)
    xticks = aplt.XTicks(fontsize=34)
    zoom = aa.Zoom2D(mask=array.mask)
    array_zoom = zoom.array_2d_from(array=array, buffer=0)
    extent = array_zoom.geometry.extent
    xticks.set(min_value=extent[0], max_value=extent[1], units=units)

    xticks = aplt.XTicks(fontsize=34)
    units = aplt.Units(use_scaled=False, ticks_convert_factor=None)
    xticks.set(min_value=extent[0], max_value=extent[1], pixels=2, units=units)

    xticks = aplt.XTicks(fontsize=34)
    units = aplt.Units(use_scaled=True, ticks_convert_factor=2.0)
    xticks.set(min_value=extent[0], max_value=extent[1], units=units)

    xticks = aplt.XTicks(fontsize=34)
    units = aplt.Units(use_scaled=False, ticks_convert_factor=2.0)
    xticks.set(min_value=extent[0], max_value=extent[1], pixels=2, units=units)
