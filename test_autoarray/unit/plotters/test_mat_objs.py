from os import path
from autoarray import conf
from autoarray.plotters import mat_objs

import pytest
import os, shutil

directory = path.dirname(path.realpath(__file__))


@pytest.fixture(autouse=True)
def set_config_path():
    conf.instance = conf.Config(
        path.join(directory, "../test_files/plotters"), path.join(directory, "output")
    )


class TestTicks:
    def test__tick_settings_setup_correctly_from_config(self):

        ticks = mat_objs.Ticks(
            ysize=34,
            xsize=35,
            y_manual=[1.0, 2.0],
            x_manual=[3.0, 4.0],
            unit_conversion_factor=2.0,
        )

        assert ticks.ysize == 34
        assert ticks.xsize == 35
        assert ticks.y_manual == [1.0, 2.0]
        assert ticks.x_manual == [3.0, 4.0]
        assert ticks.unit_conversion_factor == 2.0


class TestLabels:
    def test__title__yx_units_setup_correctly_from_config(self):

        labels = mat_objs.Labels(
            title="OMG",
            yunits="hi",
            xunits="hi2",
            use_scaled_units=True,
            titlesize=30,
            ysize=31,
            xsize=32,
        )

        assert labels.use_scaled_units == True
        assert labels.title == "OMG"
        assert labels._yunits == "hi"
        assert labels._xunits == "hi2"
        assert labels.yunits == "hi"
        assert labels.xunits == "hi2"
        assert labels.titlesize == 30
        assert labels.ysize == 31
        assert labels.xsize == 32

    def test__yx_units_use_plot_in_kpc_if_it_is_passed(self):

        labels = mat_objs.Labels(plot_in_kpc=True)

        assert labels.plot_in_kpc == True
        assert labels._yunits == None
        assert labels._xunits == None
        assert labels.yunits == "kpc"
        assert labels.xunits == "kpc"

        labels = mat_objs.Labels(plot_in_kpc=False)

        assert labels.plot_in_kpc == False
        assert labels._yunits == None
        assert labels._xunits == None
        assert labels.yunits == "arcsec"
        assert labels.xunits == "arcsec"

        labels = mat_objs.Labels(yunits="hi", xunits="hi2", plot_in_kpc=True)

        assert labels.plot_in_kpc == True
        assert labels._yunits == "hi"
        assert labels._xunits == "hi2"
        assert labels.yunits == "hi"
        assert labels.xunits == "hi2"

        labels = mat_objs.Labels(yunits="hi", xunits="hi2", plot_in_kpc=False)

        assert labels.plot_in_kpc == False
        assert labels._yunits == "hi"
        assert labels._xunits == "hi2"
        assert labels.yunits == "hi"
        assert labels.xunits == "hi2"

    def test__title_from_func__uses_func_name_if_title_is_none(self):
        def toy_func():
            pass

        labels = mat_objs.Labels(title=None)

        title_from_func = labels.title_from_func(func=toy_func)

        assert title_from_func == "Toy_func"

        labels = mat_objs.Labels(title="Hi")

        title_from_func = labels.title_from_func(func=toy_func)

        assert title_from_func == "Hi"

    def test__yx_units_from_func__uses_function_inputs_if_available(self):
        def toy_func():
            pass

        labels = mat_objs.Labels(yunits=None, xunits=None)

        yunits_from_func = labels.yunits_from_func(func=toy_func)
        xunits_from_func = labels.xunits_from_func(func=toy_func)

        assert yunits_from_func == None
        assert xunits_from_func == None

        def toy_func(label_yunits="Hi", label_xunits="Hi0"):
            pass

        labels = mat_objs.Labels()

        yunits_from_func = labels.yunits_from_func(func=toy_func)
        xunits_from_func = labels.xunits_from_func(func=toy_func)

        assert yunits_from_func == "Hi"
        assert xunits_from_func == "Hi0"

        labels = mat_objs.Labels(yunits="Hi1", xunits="Hi2")

        yunits_from_func = labels.yunits_from_func(func=toy_func)
        xunits_from_func = labels.xunits_from_func(func=toy_func)

        assert yunits_from_func == "Hi1"
        assert xunits_from_func == "Hi2"

        def toy_func(argument, label_yunits="Hi", label_xunits="Hi0"):
            pass

        labels = mat_objs.Labels()

        yunits_from_func = labels.yunits_from_func(func=toy_func)
        xunits_from_func = labels.xunits_from_func(func=toy_func)

        assert yunits_from_func == "Hi"
        assert xunits_from_func == "Hi0"

        labels = mat_objs.Labels(yunits="Hi1", xunits="Hi2")

        yunits_from_func = labels.yunits_from_func(func=toy_func)
        xunits_from_func = labels.xunits_from_func(func=toy_func)

        assert yunits_from_func == "Hi1"
        assert xunits_from_func == "Hi2"


class TestOutput:
    def test__input_path_is_created(self):

        test_path = path.join(directory, "../test_files/output_path")

        if os.path.exists(test_path):
            shutil.rmtree(test_path)

        assert not os.path.exists(test_path)

        output = mat_objs.Output(path=test_path)

        assert os.path.exists(test_path)

    def test__filename_from_func__returns_function_name_if_no_filename(self):
        def toy_func():
            pass

        output = mat_objs.Output(filename=None)

        filename_from_func = output.filename_from_func(func=toy_func)

        assert filename_from_func == "toy_func"

        output = mat_objs.Output(filename="Hi")

        filename_from_func = output.filename_from_func(func=toy_func)

        assert filename_from_func == "Hi"
