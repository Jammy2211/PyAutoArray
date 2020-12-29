from autoconf import conf
from autoarray.structures import grids
from autoarray.plot.mat_wrap import visuals as vis


class AbstractInclude:
    def __init__(self, origin=None, mask=None):

        section = conf.instance["visualize"]["include"]["include"]

        self.origin = section["origin"] if origin is None else origin
        self.mask = section["mask"] if mask is None else mask


class Include1D(AbstractInclude):
    def __init__(self, origin=None, mask=None):

        super().__init__(origin=origin, mask=mask)

    def visuals_from_line(self, line):

        origin = line.origin if self.origin else None
        mask = line.mask if self.mask else None

        return vis.Visuals1D(origin=origin, mask=mask)


class Include2D:
    def __init__(
        self,
        origin=None,
        mask=None,
        border=None,
        mapper_data_grid=None,
        mapper_source_grid=None,
        mapper_source_border=None,
        parallel_overscan=None,
        serial_prescan=None,
        serial_overscan=None,
    ):

        section = conf.instance["visualize"]["include"]["include"]

        self.origin = section["origin"] if origin is None else origin
        self.mask = section["mask"] if mask is None else mask
        self.border = section["border"] if border is None else border
        self.mapper_data_grid = (
            section["mapper_data_grid"]
            if mapper_data_grid is None
            else mapper_data_grid
        )
        self.mapper_source_grid = (
            section["mapper_source_grid"]
            if mapper_source_grid is None
            else mapper_source_grid
        )
        self.mapper_source_border = (
            section["mapper_source_border"]
            if mapper_source_border is None
            else mapper_source_border
        )
        self.parallel_overscan = (
            section["parallel_overscan"]
            if parallel_overscan is None
            else parallel_overscan
        )
        self.serial_prescan = (
            section["serial_prescan"] if serial_prescan is None else serial_prescan
        )
        self.serial_overscan = (
            section["serial_overscan"] if serial_overscan is None else serial_overscan
        )

    def visuals_from_structure(self, structure):

        origin = grids.GridIrregular(grid=[structure.origin]) if self.origin else None

        mask = structure.mask if self.mask else None

        border = (
            structure.mask.geometry.border_grid_sub_1.in_1d_binned
            if self.border
            else None
        )

        return vis.Visuals2D(origin=origin, mask=mask, border=border)

    def visuals_from_array(self, array):

        return self.visuals_from_structure(structure=array)

    def visuals_from_grid(self, grid):

        if isinstance(grid, grids.Grid):
            return self.visuals_from_structure(structure=grid)
        return vis.Visuals2D()

    def visuals_from_frame(self, frame):

        visuals_structure = self.visuals_from_structure(structure=frame)

        parallel_overscan = (
            frame.scans.parallel_overscan if self.parallel_overscan else None
        )
        serial_prescan = frame.scans.serial_prescan if self.serial_prescan else None
        serial_overscan = frame.scans.serial_overscan if self.serial_overscan else None

        return visuals_structure + vis.Visuals2D(
            parallel_overscan=parallel_overscan,
            serial_prescan=serial_prescan,
            serial_overscan=serial_overscan,
        )

    def visuals_of_data_from_mapper(self, mapper):

        origin = (
            grids.GridIrregular(grid=[mapper.source_full_grid.mask.origin])
            if self.origin
            else None
        )
        grid = mapper.data_pixelization_grid if self.mapper_data_grid else None

        border = (
            mapper.source_full_grid.mask.geometry.border_grid_sub_1.in_1d_binned
            if self.border
            else None
        )

        return vis.Visuals2D(origin=origin, grid=grid, border=border)

    def visuals_of_source_from_mapper(self, mapper):

        origin = (
            grids.GridIrregular(grid=[mapper.source_pixelization_grid.origin])
            if self.origin
            else None
        )
        grid = mapper.source_pixelization_grid if self.mapper_source_grid else None
        #     border = mapper.source_grid.sub_border_grid if self.mapper_source_grid else None

        return vis.Visuals2D(origin=origin, grid=grid)  # , border=border)
