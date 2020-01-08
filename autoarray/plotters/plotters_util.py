from autoarray import conf
from autoarray import exc
from autoarray.plotters import plotters

def setting(section, name, python_type):
    return conf.instance.visualize.get(section, name, python_type)

def load_setting(value, name, python_type):
    return (
        value
        if value is not None
        else setting(section="settings", name=name, python_type=python_type)
    )

def load_figure_setting(value, name, python_type):
    return (
        value
        if value is not None
        else setting(section="figures", name=name, python_type=python_type)
    )

def load_subplot_setting(value, name, python_type):
    return (
        value
        if value is not None
        else setting(section="subplots", name=name, python_type=python_type)
    )

def label_title_from_plotter(plotter, func):
    if plotter.label_title is None:

        return func.__name__.capitalize()

    else:

        return plotter.label_title

def label_yunits_from_plotter(plotter):

    if plotter.label_yunits is None:
        if plotter.use_scaled_units:
            return "scaled"
        else:
            return "pixels"

    else:

        return plotter.label_yunits

def label_xunits_from_plotter(plotter):

    if plotter.label_xunits is None:
        if plotter.use_scaled_units:
            return "scaled"
        else:
            return "pixels"

    else:

        return plotter.label_xunits

def output_filename_from_plotter_and_func(plotter, func):

    if plotter.output_filename is None:
        return func.__name__
    else:

        return plotter.output_filename

def plotter_key_from_dictionary(dictionary):

    plotter_key = None

    for key, value in dictionary.items():
        if isinstance(value, plotters.Plotter):
            plotter_key = key

    if plotter_key is None:
        raise exc.PlottingException("The plot function called could not locate a Plotter in the kwarg arguments"
                                    "in order to set the labels.")

    return plotter_key