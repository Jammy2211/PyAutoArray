from autoarray import conf

def setting(section, name, python_type):
    return conf.instance.visualize.get(section, name, python_type)


def figure_setting(value, name, python_type):
    return value if value is not None else setting(section="figures", name=name, python_type=python_type)


class PlotterSettings(object):

    def __init__(self,
                 include_origin=None,
                 include_mask=None,
                 include_border=None,
                 include_positions=None,
    use_scaled_units=None,
    unit_conversion_factor=None,
    unit_label=None,
    figsize=None,
    aspect=None,
    cmap=None,
    norm=None,
    norm_min=None,
    norm_max=None,
    linthresh=None,
    linscale=None,
    cb_ticksize=None,
    cb_fraction=None,
    cb_pad=None,
    titlesize=None,
    xlabelsize=None,
    ylabelsize=None,
    xyticksize=None,
    mask_pointsize=None,
    position_pointsize=None,
    grid_pointsize=None,
    cb_tick_values=None,
    cb_tick_labels=None,
    title=None,
    output_path=None,
    output_format="show",
    output_filename=None):

        self.include_origin = figure_setting(value=include_origin, name="include_origin", python_type=bool)
        self.include_mask = figure_setting(value=include_mask, name="include_mask", python_type=bool)
        self.include_border = figure_setting(value=include_border, name="include_border", python_type=bool)
        self.include_positions = figure_setting(value=include_positions, name="include_positions", python_type=bool)

        self.use_scaled_units = figure_setting(value=use_scaled_units, name="use_scaled_units", python_type=bool)
        self.unit_conversion_factor = unit_conversion_factor
        self.unit_label = figure_setting(value=unit_label, name="unit_label", python_type=str)
        self.figsize = figure_setting(value=figsize, name="figsize", python_type=str)
        if isinstance(self.figsize, str):
            self.figsize = tuple(map(int, self.figsize[1:-1].split(',')))
        self.aspect = figure_setting(value=aspect, name="aspect", python_type=str)
        self.cmap = figure_setting(value=cmap, name="cmap", python_type=str)
        self.norm = figure_setting(value=norm, name="norm", python_type=str)
        self.norm_min = figure_setting(value=norm_min, name="norm_min", python_type=float)
        self.norm_max = figure_setting(value=norm_max, name="norm_max", python_type=float)
        self.linthresh = figure_setting(value=linthresh, name="linthresh", python_type=float)
        self.linscale = figure_setting(value=linscale, name="linscale", python_type=float)
        self.cb_ticksize = figure_setting(value=cb_ticksize, name="cb_ticksize", python_type=int)
        self.cb_fraction = figure_setting(value=cb_fraction, name="cb_fraction", python_type=float)
        self.cb_pad = figure_setting(value=cb_pad, name="cb_pad", python_type=float)
        self.titlesize = figure_setting(value=titlesize, name="titlesize", python_type=int)
        self.xlabelsize = figure_setting(value=xlabelsize, name="xlabelsize", python_type=int)
        self.ylabelsize = figure_setting(value=ylabelsize, name="ylabelsize", python_type=int)
        self.xyticksize = figure_setting(value=xyticksize, name="xyticksize", python_type=int)
        self.mask_pointsize = figure_setting(value=mask_pointsize, name="mask_pointsize", python_type=int)
        self.position_pointsize = figure_setting(value=position_pointsize, name="position_pointsize", python_type=int)
        self.grid_pointsize = figure_setting(value=grid_pointsize, name="grid_pointsize", python_type=int)

        self.cb_tick_values = cb_tick_values
        self.cb_tick_labels = cb_tick_labels
        self.title = title
        self.output_path = output_path
        self.output_format = output_format
        self.output_filename = output_filename