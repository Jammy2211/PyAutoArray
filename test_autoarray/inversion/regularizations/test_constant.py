import autoarray as aa
import numpy as np
np.set_printoptions(threshold=np.inf)
from autoarray.mock import fixtures
from autoarray.mock.mock import MockPixelizationGrid, MockMapper


def test__regularization_matrix__matches_util():

    pixel_neighbors = np.array(
        [
            [1, 3, 7, 2],
            [4, 2, 0, -1],
            [1, 5, 3, -1],
            [4, 6, 0, -1],
            [7, 1, 5, 3],
            [4, 2, 8, -1],
            [7, 3, 0, -1],
            [4, 8, 6, -1],
            [7, 5, -1, -1],
        ]
    )

    pixel_neighbors_sizes = np.array([4, 3, 3, 3, 4, 3, 3, 3, 2])

    pixelization_grid = MockPixelizationGrid(
        pixel_neighbors=pixel_neighbors, pixel_neighbors_sizes=pixel_neighbors_sizes
    )

    mapper = MockMapper(source_pixelization_grid=pixelization_grid)

    reg = aa.reg.Constant(coefficient=1.0)
    regularization_matrix = reg.regularization_matrix_from(mapper=mapper)

    regularization_matrix_util = aa.util.regularization.constant_regularization_matrix_from(
        coefficient=1.0,
        pixel_neighbors=pixel_neighbors,
        pixel_neighbors_sizes=pixel_neighbors_sizes,
    )

    assert (regularization_matrix == regularization_matrix_util).all()


#def test__ConstantSplit():
# I used 'reg_output' to help manually check the coefficients. Haven't get a clear idea on how to write the test part here,
# so I comment all the lines here and will update it soon. 
#
#
#    pixelization_grid = aa.Grid2D.manual_slim(                                                       
#        [[0.1, 0.1], [1.1, 0.6], [2.1, 0.1], [0.4, 1.1], [1.1, 7.1], [2.1, 1.1]],                    
#        shape_native=(3, 2),                                                                         
#        pixel_scales=1.0,                                                                            
#    )
#
#    nearest_pixelization_index_for_slim_index = np.array([0, 0, 1, 0, 0, 1, 2, 2, 3])
#
#    pixelization_grid = aa.Grid2DVoronoi(                                                            
#        grid=pixelization_grid,                                                                      
#        nearest_pixelization_index_for_slim_index=nearest_pixelization_index_for_slim_index,         
#        uses_interpolation=True,                                                                    
#    )
#
#    grid_2d_7x7 = fixtures.make_grid_2d_7x7()
#
#    mapper = aa.MapperVoronoi(source_grid_slim=grid_2d_7x7, source_pixelization_grid=pixelization_grid)
#
#    mappings = mapper.pix_sub_weights_split_cross.mappings
#    sizes = mapper.pix_sub_weights_split_cross.sizes
#    weights = mapper.pix_sub_weights_split_cross.weights
#
#    reg = aa.reg.ConstantSplit(coefficient=1.0)
#
#    reg_output = ''
#    for i in range(len(mappings)):
#        pix_index = i // 4
#        reg_output += ' + (a{}'.format(pix_index)
#        for j in range(sizes[i]):
#            reg_output += ' - {:.6f}a{}'.format(weights[i][j], mappings[i][j])
#        reg_output += ')^2'
#
#    print(reg_output)
#    #print(reg.regularization_matrix_from(mapper))
#
#    #print(mapper.pix_sub_weights_split_cross.mappings)
#    #print(mapper.pix_sub_weights_split_cross.sizes)
#    #print(mapper.pix_sub_weights_split_cross.weights)

