## nn ##
(Natural Neighbours interpolation for PyAutoLens)

If you want to use the `VoronoiNN` pixelization, which applies natural neighbor 
interpolation (https://en.wikipedia.org/wiki/Natural_neighbor_interpolation) to a Voronoi pixelization you must 
install this C package. 

This currently requires that PyAutoLens is built from source, e.g. via cloning PyAutoLens and its parent packagees 
from GitHub (https://pyautolens.readthedocs.io/en/latest/installation/source.html).

The code is a slightly modified version of "https://github.com/sakov/nn-c", a natural neighboring interpolation C 
code written by Pavel Sakov. 

To install nn for PyAutoLens on a linux machine (and presumably a MAC) follow the steps below:

1. Put directory 'nn' folders in your global command line variable LD_LIBRARY_PATH. 

    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/your/path/to/autoarray/util/nn/src/nn

You may wish to add this to your ~/.bashrc file or virtual enviroment activate script so you do not need to re-enter 
the path every time you open a new enviroment.

2. go to directory 'nn'

    cd /your/path/to/autoarray/util/nn/src/nn

3. run:
 
    ./configure

4. run the following command is to backup the automatically generated makefile.

    cp makefile_autolens makefile

5. Run the make command:

    make

If you see libnnhpi_customized.so, it should be correctly compiled. 

To test the installation go to the folder test_autoarray/util, and run 'pytest'

To clean all compiled fields, run:

    make distclean 


