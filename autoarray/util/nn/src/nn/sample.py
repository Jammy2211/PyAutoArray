import ctypes
import os
import numpy as np
import time
import matplotlib.pyplot as plt


def xy_function(x, y):

    return x * x + 2 * y - x * y


np.random.seed(2)

nin = 4000
x = 10.0 * np.random.random(nin)
y = 10.0 * np.random.random(nin)
z = xy_function(x, y)

nout_1d = int(5)
nout = nout_1d * nout_1d
xout_oneside = np.linspace(10.0, 20.0, nout_1d)
yout_oneside = np.linspace(10.0, 20.0, nout_1d)
xout_2d, yout_2d = np.meshgrid(xout_oneside, yout_oneside)
xout = xout_2d.ravel()
yout = yout_2d.ravel()

max_nneighbor = 25
weights_out = np.zeros(nout * max_nneighbor, dtype=np.double)
neighbor_index = np.zeros(nout * max_nneighbor, dtype=np.intc) - 1


# x = np.array(x, dtype='float')
# y = np.array(y, dtype='float')

_file = "libqiuhan.so"
_mod = ctypes.cdll.LoadLibrary("./" + _file)

interpolate_from_input = _mod.interpolate_weights_from_input

interpolate_from_input.argtypes = (
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

interpolate_from_input.restype = ctypes.c_int

t1 = time.time()
answer = interpolate_from_input(
    x.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
    y.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
    z.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
    ctypes.c_int(nin),
    xout.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
    yout.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
    ctypes.c_int(nout),
    weights_out.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
    neighbor_index.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
    ctypes.c_int(max_nneighbor),
)
t2 = time.time()
print("Time cost is {:.2f}".format(t2 - t1))


print(weights_out)
print(neighbor_index[0])

weights_out_2d = weights_out.reshape((nout, max_nneighbor))
neighbor_index_2d = neighbor_index.reshape((nout, max_nneighbor))

print("answer from python: {}".format(answer))

ind = 1165
print("Point Value {} {}".format(ind, z[ind]))


ind = 228
print("Point Value {} {}".format(ind, z[ind]))

print(neighbor_index_2d)
print(weights_out_2d)


# plt.imshow(zout.reshape((nout_1d, nout_1d)))
# plt.colorbar()
# plt.show()
