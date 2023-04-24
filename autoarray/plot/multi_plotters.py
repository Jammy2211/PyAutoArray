class MultiFigurePlotter:
    def __init__(self, plotter_list):
        self.plotter_list = plotter_list

    def subplot_of_figure(self, func_name, figure_name, filename_suffix="", **kwargs):
        number_subplots = len(self.plotter_list)

        self.plotter_list[0].open_subplot_figure(number_subplots=number_subplots)

        for i, plotter in enumerate(self.plotter_list):
            func = getattr(plotter, func_name)
            func(**{**{figure_name: True}, **kwargs})

        if self.plotter_list[0].mat_plot_1d is not None:
            self.plotter_list[0].mat_plot_1d.output.subplot_to_figure(
                auto_filename=f"subplot_{figure_name}{filename_suffix}_list"
            )
        if self.plotter_list[0].mat_plot_2d is not None:
            self.plotter_list[0].mat_plot_2d.output.subplot_to_figure(
                auto_filename=f"subplot_{figure_name}{filename_suffix}_list"
            )
        self.plotter_list[0].close_subplot_figure()


class MultiYX1DPlotter:
    def __init__(self, plotter_list, color_list=None, legend_labels=None):
        self.plotter_list = plotter_list

        if color_list is None:
            color_list = 10 * ["k", "r", "b", "g", "c", "m", "y"]

        self.color_list = color_list
        self.legend_labels = legend_labels

    def figure_1d(self, func_name, figure_name, **kwargs):
        self.plotter_list[0].mat_plot_1d.figure.open()

        for i, plotter in enumerate(self.plotter_list):
            plotter.set_mat_plot_1d_for_multi_plot(
                is_for_multi_plot=True, color=self.color_list[i]
            )

            if self.legend_labels is not None:
                plotter.mat_plot_1d.yx_plot.label = self.legend_labels[i]

            func = getattr(plotter, func_name)
            func(**{**{figure_name: True}, **kwargs})

            plotter.set_mat_plot_1d_for_multi_plot(is_for_multi_plot=False, color=None)

        self.plotter_list[0].mat_plot_1d.output.subplot_to_figure(
            auto_filename=f"multi_{figure_name}"
        )
        self.plotter_list[0].mat_plot_1d.figure.close()
