class DatasetInterface:
    def __init__(
        self,
        data,
        noise_map,
        over_sampler=None,
        border_relocator=None,
        convolver=None,
        transformer=None,
        w_tilde=None,
        grid=None,
        grid_pixelization=None,
        blurring_grid=None,
        noise_covariance_matrix=None,
    ):
        """
        Generic class which acts as an interface between a dataset and an inversion.

        The inputs to the inversion module are an `Imaging` or `Interferometer` dataset, and it is recommend
        instances of these objects should be input into an inversion whenever possible.

        However, the dataset's attributes are commonly modified before being input into the inversion, for example:

        - In PyAutoGalaxy and PyAutoLens, the data may have the light profiles of certain galaxies or the background
        sky subtracted from it.

        - The noise-map may be scaled to put large values in regions of the data identified to be difficult to fit.

        - The PSF may be a part of the model where it is customized by free parameters which vary.

        In all cases, the dataset's attributes (modified and unmodified) can be passed through this class and into the
        inversion.

        Parameters
        ----------
        data
            The array of the image data containing the signal that is fitted (in PyAutoGalaxy and PyAutoLens the
            recommended units are electrons per second).
        noise_map
            An array describing the RMS standard deviation error in each pixel used for computing quantities like the
            chi-squared in a fit (in PyAutoGalaxy and PyAutoLens the recommended units are electrons per second).
        over_sampler
            Performs over-sampling whereby the masked image pixels are split into sub-pixels, which are all
            mapped via the mapper with sub-fractional values of flux.
        border_relocator
           The border relocator, which relocates coordinates outside the border of the source-plane data grid to its
           edge.
        convolver
            Perform 2D convolution of the imaigng data's PSF when computing the operated mapping matrix.
        transformer
            Performs a Fourier transform of the image-data from real-space to visibilities when computing the
            operated mapping matrix.
        w_tilde
            The w_tilde matrix used by the w-tilde formalism to construct the data vector and
            curvature matrix during an inversion efficiently..
        grid
            The grid of (y,x) Cartesian coordinates that the image data is paired with, which is used for calculations
            not associated with a pixelization (e.g. linear func objects which in PyAutoGalaxy and PyAutooLens represent
            linear light profiles).
        grid_pixelization
            The grid of (y,x) Cartesian coordinates that the image data is paired with, which is used for calculations
            associated with a pixelization.
        blurring_grid
            The grid of (y,x) Cartesian coordinates of every pixel in the image data which is outside the masked
            region fitted but that is close enough to the masked region that its light is blurred into the mask
            after PSF convolution. The blurring grid is used for linear func calculations and not pixelization
            calculations.
        noise_covariance_matrix
            A noise-map covariance matrix representing the covariance between noise in every `data` value, which
            can be used via a bespoke fit to account for correlated noise in the data.
        """
        self.data = data
        self.noise_map = noise_map
        self.over_sampler = over_sampler
        self.border_relocator = border_relocator
        self.convolver = convolver
        self.transformer = transformer
        self.w_tilde = w_tilde
        self.grid = grid
        self.grid_pixelization = grid_pixelization
        self.blurring_grid = blurring_grid
        self.noise_covariance_matrix = noise_covariance_matrix

    @property
    def mask(self):
        return self.grid.mask
