from typing import Dict, Optional

from autoarray.dataset.mock.mock_dataset import MockDataset
from autoarray.fit.fit_imaging import FitImaging


class MockFitImaging(FitImaging):
    def __init__(
        self,
        dataset=MockDataset(),
        use_mask_in_fit: bool = False,
        noise_map=None,
        model_data=None,
        inversion=None,
        blurred_image=None,
        profiling_dict: Optional[Dict] = None,
    ):
        super().__init__(
            dataset=dataset,
            use_mask_in_fit=use_mask_in_fit,
            profiling_dict=profiling_dict,
        )

        self._noise_map = noise_map
        self._model_data = model_data
        self._inversion = inversion
        self._blurred_image = blurred_image

    @property
    def noise_map(self):
        return self._noise_map if self._noise_map is not None else super().noise_map

    @property
    def model_data(self):
        return self._model_data

    @property
    def inversion(self):
        return self._inversion if self._inversion is not None else super().inversion

    @property
    def blurred_image(self):
        return (
            self._blurred_image
            if self._blurred_image is not None
            else super().blurred_image
        )
