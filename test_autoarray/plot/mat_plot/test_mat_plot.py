import autoarray.plot as aplt


def test__add_mat_plot_objects_together():

    extent = [1.0, 2.0, 3.0, 4.0]
    fontsize = 20

    mat_plot_2d_0 = aplt.MatPlot2D(axis=aplt.Axis(extent=extent))

    mat_plot_2d_1 = aplt.MatPlot2D(ylabel=aplt.YLabel(fontsize=fontsize))

    mat_plot_2d = mat_plot_2d_0 + mat_plot_2d_1

    assert mat_plot_2d.axis.config_dict["extent"] == extent
    assert mat_plot_2d.ylabel.config_dict["fontsize"] == 20

    mat_plot_2d = mat_plot_2d_1 + mat_plot_2d_0

    assert mat_plot_2d.axis.config_dict["extent"] == extent
    assert mat_plot_2d.ylabel.config_dict["fontsize"] == 20

    units = aplt.Units()
    output = aplt.Output(format="png")

    mat_plot_2d_0 = aplt.MatPlot2D(
        axis=aplt.Axis(extent=extent), units=units, output=output
    )
    mat_plot_2d_1 = aplt.MatPlot2D(ylabel=aplt.YLabel(fontsize=fontsize))

    mat_plot_2d = mat_plot_2d_0 + mat_plot_2d_1

    assert mat_plot_2d.output.format == "png"

    mat_plot_2d = mat_plot_2d_1 + mat_plot_2d_0

    assert mat_plot_2d.output.format == "png"
