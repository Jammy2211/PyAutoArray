from __future__ import annotations
import logging
import copy
import numpy as np
from typing import TYPE_CHECKING, List, Tuple, Union

if TYPE_CHECKING:
    from autoarray.structures.arrays.uniform_2d import Array2D
    from autoarray.structures.grids.uniform_2d import Grid2D

from autoconf import cached_property

from autoarray.mask.abstract_mask import Mask

from autoarray import exc
from autoarray import type as ty
from autoarray.structures.arrays import array_2d_util
from autoarray.geometry import geometry_util
from autoarray.structures.grids import grid_2d_util
from autoarray.mask import mask_2d_util

logging.basicConfig()
logger = logging.getLogger(__name__)

class Geometry2D:

    def __init__(self, mask):

        self.mask = mask

    @property
    def central_pixel_coordinates(self):
        return geometry_util.central_pixel_coordinates_2d_from(
            shape_native=self.mask.shape_native
        )