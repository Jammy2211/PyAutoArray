import logging
from astropy import time
import numpy as np

from autoarray.dataset import preprocess

logging.basicConfig()
logger = logging.getLogger(__name__)


def convert_array(array):
    """
    If the input array input a convert is of type list, convert it to type NumPy array.

    Parameters
    ----------
    array : list or ndarray
        The array which may be converted to an ndarray
    """

    if type(array) is list:
        array = np.asarray(array)

    return array


class Header:
    def __init__(
        self, header_sci_obj, header_hdu_obj, readout_offsets: (int, int) = (0, 0)
    ):

        self.header_sci_obj = header_sci_obj
        self.header_hdu_obj = header_hdu_obj
        self.readout_offsets = readout_offsets

    @property
    def date_of_observation(self):
        return self.header_sci_obj["DATE-OBS"]

    @property
    def time_of_observation(self):
        return self.header_sci_obj["TIME-OBS"]

    @property
    def exposure_time(self):
        return self.header_sci_obj["EXPTIME"]

    @property
    def modified_julian_date(self):
        if (
            self.date_of_observation is not None
            and self.time_of_observation is not None
        ):
            t = time.Time(self.date_of_observation + "T" + self.time_of_observation)
            return t.mjd
        return None

    def array_eps_to_counts(self, array_eps):
        raise NotImplementedError()

    def array_counts_to_counts_per_second(self, array_counts):
        return preprocess.array_counts_to_counts_per_second(
            array_counts=array_counts, exposure_time=self.exposure_time
        )
