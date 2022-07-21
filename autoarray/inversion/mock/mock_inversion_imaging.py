from autoarray.inversion.inversion.imaging.mapping import InversionImagingMapping
from autoarray.inversion.inversion.settings import SettingsInversion
from autoarray.preloads import Preloads


class MockInversionImaging(InversionImagingMapping):
    def __init__(
        self,
        data=None,
        noise_map=None,
        convolver=None,
        linear_obj_list=None,
        regularization_list=None,
        settings: SettingsInversion = SettingsInversion(),
        preloads: Preloads = Preloads(),
    ):

        # self.__dict__["curvature_matrix"] = curvature_matrix
        # self.__dict__["curvature_reg_matrix_cholesky"] = curvature_reg_matrix_cholesky
        # self.__dict__["regularization_matrix"] = regularization_matrix
        # self.__dict__["curvature_reg_matrix"] = curvature_reg_matrix
        # self.__dict__["reconstruction"] = reconstruction
        # self.__dict__["mapped_reconstructed_image"] = mapped_reconstructed_image

        super().__init__(
            data=data,
            noise_map=noise_map,
            convolver=convolver,
            linear_obj_list=linear_obj_list,
            regularization_list=regularization_list,
            settings=settings,
            preloads=preloads,
        )
