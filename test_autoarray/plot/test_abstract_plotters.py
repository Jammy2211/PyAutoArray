from autoarray.plot.abstract_plotters import AbstractPlotter
from autoarray.plot.wrap.base.output import Output
from autoarray.plot.wrap.base.cmap import Cmap


def test__abstract_plotter__basic():
    plotter = AbstractPlotter()
    assert plotter.output is not None
    assert plotter.cmap is not None
    assert plotter.use_log10 is False


def test__abstract_plotter__set_title():
    plotter = AbstractPlotter()
    plotter.set_title("test label")
    assert plotter.title == "test label"


def test__abstract_plotter__set_filename():
    plotter = AbstractPlotter()
    plotter.set_filename("my_file")
    assert plotter.output.filename == "my_file"


def test__abstract_plotter__custom_output_and_cmap():
    output = Output(path="/tmp", format="png")
    cmap = Cmap(cmap="hot")
    plotter = AbstractPlotter(output=output, cmap=cmap, use_log10=True)
    assert plotter.output.path == "/tmp"
    assert plotter.cmap.cmap_name == "hot"
    assert plotter.use_log10 is True
