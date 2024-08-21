import numpy as np
import ctypes
import os


try:
    _file = (
        os.path.realpath(os.path.dirname(__file__)) + "/src/nn/libnnhpi_customized.so"
    )
    _mod = ctypes.cdll.LoadLibrary(_file)

    # interpolate_weights_from is for creating a mapper
    interpolate_weights_from = _mod.interpolate_weights_from
    interpolate_weights_from.argtypes = (
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_int),
        ctypes.c_int,
    )
    interpolate_weights_from.restype = ctypes.c_int

    # interpolate_from is for plotting
    interpolate_from = _mod.interpolate_from
    interpolate_from.argtypes = (
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_int),
        ctypes.c_int,
    )
    interpolate_from.restype = ctypes.c_int

    def natural_interpolation_weights(x_in, y_in, x_target, y_target, max_nneighbours):
        nin = len(x_in)
        nout = len(x_target)

        z_in = np.zeros(len(x_in), dtype=np.double)

        weights_out = np.zeros(nout * max_nneighbours, dtype=np.double)
        neighbour_indexes_out = np.zeros(nout * max_nneighbours, dtype=np.intc) - 1

        flag = interpolate_weights_from(
            (x_in.astype(np.double)).ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            (y_in.astype(np.double)).ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            z_in.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            ctypes.c_int(nin),
            (x_target.astype(np.double)).ctypes.data_as(
                ctypes.POINTER(ctypes.c_double)
            ),
            (y_target.astype(np.double)).ctypes.data_as(
                ctypes.POINTER(ctypes.c_double)
            ),
            ctypes.c_int(nout),
            weights_out.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            neighbour_indexes_out.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            ctypes.c_int(max_nneighbours),
        )

        return (
            weights_out.reshape((nout, max_nneighbours)),
            neighbour_indexes_out.reshape((nout, max_nneighbours)),
        )

    def natural_interpolation(x_in, y_in, z_in, x_target, y_target):
        nin = len(x_in)
        nout = len(x_target)

        z_target = np.zeros(nout, dtype=np.double)
        z_marker = np.zeros(nout, dtype=np.intc)

        flag = interpolate_from(
            (x_in.astype(np.double)).ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            (y_in.astype(np.double)).ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            (z_in.astype(np.double)).ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            ctypes.c_int(nin),
            (x_target.astype(np.double)).ctypes.data_as(
                ctypes.POINTER(ctypes.c_double)
            ),
            (y_target.astype(np.double)).ctypes.data_as(
                ctypes.POINTER(ctypes.c_double)
            ),
            z_target.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            z_marker.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            ctypes.c_int(nout),
        )

        for i in range(nout):
            if z_marker[i] == -1:
                cloest_point_index = np.argmin(
                    (x_in - x_target[i]) ** 2.0 + (y_in - y_target[i]) ** 2.0
                )
                z_target[i] = z_in[cloest_point_index]

        return z_target

except OSError:
    print("natural neighbour interpolation not loaded.")
    raise ImportError

natural_interpolation_weights