from autofit.exc import FitException

class ArrayException(Exception):
    pass


class ScaledException(Exception):
    pass


class GridException(Exception):
    pass


class KernelException(Exception):
    pass


class ConvolutionException(Exception):
    pass


class MaskException(Exception):
    pass


class MappingException(Exception):
    pass


class DataException(Exception):
    pass


class PixelizationException(Exception):
    pass


class FitException(Exception):
    pass


class InversionException(FitException):
    pass
