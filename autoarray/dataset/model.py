class DatasetModel:
    def __init__(self, background_sky_level: float = 0.0):
        """
        Attributes which allow for parts of a dataset to be treated as a model, meaning they can be fitted
        for in the `fit` module.

        The following aspects of a dataset can be treated as a model:

         - `background_sky_level`: The data may have a constant signal in the background which is estimated
         and subtracted from the data beforehand with a degree of uncertainty. By including it in the model it can be
         marginalized over. Units are dimensionless and derived from the data.

        Parameters
        ----------
        background_sky_level
            Overall normalisation of the sky which is added or subtracted from the data. Units are dimensionless and
            derived from the data, which is expected to be electrons per second in Astronomy analyses.
        """
        self.background_sky_level = background_sky_level
