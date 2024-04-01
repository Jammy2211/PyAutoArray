from autoarray.structures.over_sample.uniform import OverSampleUniformFunc
from autoarray.inversion.pixelization.border_relocator import BorderRelocator

class MapperTools:

    def __init__(self, over_sample : OverSampleUniformFunc = None, border_relocator : BorderRelocator = None):

        self.over_sample = over_sample
        self.border_relocator = border_relocator