from autoarray.plot.wrap.base.abstract import set_backend

set_backend()

from autoarray.plot.wrap.base.output import Output
from autoarray.plot.wrap.base.cmap import Cmap
from autoarray.plot.wrap.base.title import Title


class AbstractPlotter:
    def __init__(
        self,
        output: Output = None,
        cmap: Cmap = None,
        use_log10: bool = False,
        title: Title = None,
    ):
        self.output = output or Output()
        self.cmap = cmap or Cmap()
        self.use_log10 = use_log10
        self.title = title or Title()

    def set_title(self, label):
        self.title.manual_label = label

    def set_filename(self, filename):
        self.output.filename = filename

    def set_format(self, format):
        self.output._format = format
