def _set_backend():
    try:
        import matplotlib
        from autoconf import conf
        backend = conf.get_matplotlib_backend()
        if backend not in "default":
            matplotlib.use(backend)
        try:
            hpc_mode = conf.instance["general"]["hpc"]["hpc_mode"]
        except KeyError:
            hpc_mode = False
        if hpc_mode:
            matplotlib.use("Agg")
    except Exception:
        pass


_set_backend()

from autoarray.plot.wrap.base.output import Output
from autoarray.plot.wrap.base.cmap import Cmap


class AbstractPlotter:
    def __init__(
        self,
        output: Output = None,
        cmap: Cmap = None,
        use_log10: bool = False,
        title: str = None,
    ):
        self.output = output or Output()
        self.cmap = cmap or Cmap()
        self.use_log10 = use_log10
        self.title = title

    def set_title(self, label):
        self.title = label

    def set_filename(self, filename):
        self.output.filename = filename

    def set_format(self, format):
        self.output._format = format
