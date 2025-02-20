import logging
import matplotlib.pyplot as plt
from os import path
import os
from typing import Union, List, Optional

from autoarray.structures.abstract_structure import Structure

logger = logging.getLogger(__name__)


class Output:
    def __init__(
        self,
        path: Optional[str] = None,
        filename: Optional[str] = None,
        prefix: Optional[str] = None,
        suffix: Optional[str] = None,
        format: Union[str, List[str]] = None,
        format_folder: bool = False,
        bypass: bool = False,
        bbox_inches: str = "tight",
        **kwargs,
    ):
        """
        Sets how the figure or subplot is output, either by displaying it on the screen or writing it to hard-disk.

        This object wraps the following Matplotlib methods:

        - plt.show: https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.pyplot.show.html
        - plt.savefig: https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.pyplot.savefig.html

        The default behaviour is the display the figure on the computer screen, as opposed to outputting to hard-disk
        as a file.

        Parameters
        ----------
        path
            If the figure is output to hard-disk the path of the folder it is saved to.
        filename
            If the figure is output to hard-disk the filename used to save it.
        prefix
            A prefix appended before the file name, e.g. ("prefix_filename").
        prefix
            A prefix appended after the file name, e.g. ("filenam_suffix").
        format
            The format of the output, 'show' displays on the computer screen, 'png' outputs to .png, 'fits' outputs to
            `.fits` format.
        format_folder
            If `True`, all images are output in a folder giving the format name, for
            example `path/to/output/png/filename.png`. This can make managing large image catalogues easier.
        bypass
            Whether to bypass the `plt.show` or `plt.savefig` methods, used when plotting a subplot.
        """
        self.path = path

        self.filename = filename
        self.prefix = prefix
        self.suffix = suffix
        self._format = format
        self.format_folder = format_folder
        self.bypass = bypass
        self.bbox_inches = bbox_inches
        self._tag_fits_multi = None

        self.kwargs = kwargs

    @property
    def format(self) -> str:
        if self._format is None:
            return "show"
        return self._format

    @property
    def format_list(self):
        if not isinstance(self.format, list):
            return [self.format]
        return self.format

    def output_path_from(self, format):
        if format in "show":
            return None

        if self.format_folder:
            output_path = path.join(self.path, format)
        else:
            output_path = self.path

        os.makedirs(output_path, exist_ok=True)

        return output_path

    def filename_from(self, auto_filename):
        filename = auto_filename if self.filename is None else self.filename

        if self.prefix is not None:
            filename = f"{self.prefix}{filename}"

        if self.suffix is not None:
            filename = f"{filename}{self.suffix}"

        return filename

    def savefig(self, filename: str, output_path: str, format: str):
        try:
            plt.savefig(
                path.join(output_path, f"{filename}.{format}"),
                bbox_inches=self.bbox_inches, pad_inches=0
            )
        except ValueError as e:
            logger.info(
                f"""
                Failed to output figure as a .{format} or .fits due to the following error:

                {e}
            """
            )

    def to_figure(
        self, structure: Optional[Structure], auto_filename: Optional[str] = None
    ):
        """
        Output the figure, by either displaying it on the user's screen or to the hard-disk as a .png or .fits file.

        Parameters
        ----------
        structure
            The 2D array of image to be output, required for outputting the image as a fits file.
        auto_filename
            If the filename is not manually specified this name is used instead, which is defined in the parent plotter.
        """

        filename = self.filename_from(auto_filename=auto_filename)

        for format in self.format_list:
            output_path = self.output_path_from(format=format)

            if format != "show":
                os.makedirs(output_path, exist_ok=True)

            if not self.bypass:
                if os.environ.get("PYAUTOARRAY_OUTPUT_MODE") == "1":
                    return self.to_figure_output_mode(filename=filename)

                if format == "show":
                    plt.show()
                elif format == "png" or format == "pdf":
                    self.savefig(filename, output_path, format)
                elif format == "fits":
                    if structure is not None:
                        structure.output_to_fits(
                            file_path=path.join(output_path, f"{filename}.fits"),
                            overwrite=True,
                        )
                elif format == "fits_multi":
                    if structure is not None:

                        from autoarray.structures.arrays.array_2d_util import update_fits_file

                        update_fits_file(
                            arr=structure.native,
                            file_path=path.join(output_path, f"{filename}.fits"),
                            tag=self._tag_fits_multi
                        )


    def subplot_to_figure(self, auto_filename: Optional[str] = None):
        """
        Output a subplot figure, either as an image on the screen or to the hard-disk as a png or fits file.

        Parameters
        ----------
        auto_filename
            If the filename is not manually specified this name is used instead, which is defined in the parent plotter.
        """

        filename = self.filename_from(auto_filename=auto_filename)

        for format in self.format_list:
            output_path = self.output_path_from(format=format)

            if format != "show":
                os.makedirs(output_path, exist_ok=True)

            if os.environ.get("PYAUTOARRAY_OUTPUT_MODE") == "1":
                return self.to_figure_output_mode(filename=filename)

            if format == "show":
                plt.show()
            elif format == "png" or format == "pdf":
                self.savefig(filename, output_path, format)

    def to_figure_output_mode(self, filename: str):
        global COUNT

        try:
            COUNT += 1
        except NameError:
            COUNT = 0

        import sys

        script_name = path.split(sys.argv[0])[-1].replace(".py", "")

        output_path = path.join(os.getcwd(), "output_mode", script_name)
        os.makedirs(output_path, exist_ok=True)

        self.savefig(f"{COUNT}_{filename}", output_path, "png")
