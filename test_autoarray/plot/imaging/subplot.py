import autoarray as aa
import autoarray.plot as aplt

array = aa.Array.ones(shape_2d=(31, 31), pixel_scales=(1.0, 1.0), sub_size=1)
array[0] = 3.0

kernel = aa.Kernel.ones(shape_2d=(3, 3), pixel_scales=(1.0, 1.0))
kernel[0] = 3.0

imaging = aa.Imaging(image=array, noise_map=array, psf=kernel)

# aplt.Imaging.image(imaging=imaging)

aplt.Imaging.subplot_imaging(imaging=imaging)
