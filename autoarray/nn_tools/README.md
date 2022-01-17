## nn ##
(Natural Neighbours interpolation for PyAutoLens)

This code is mainly copied from "https://github.com/sakov/nn-c", a natural neighboring interpolation C code written 
by Pavel Sakov. Here, to make it run for PyAutoLens, we have slightly modefied it.

To install nn for PyAutoLens on a linux machine, follow the steps below:

1. Put directory 'nn' folders under LD_LIBRARY_PATH. 
e.g. add a line to ~/.bashrc like 

    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/your/path/to/autoarray/nn_tools/nn_c_sources/nn

2. go to directory 'nn'

    cd /your/path/to/autoarray/nn_tools/nn_c_sources/nn

3. run: 

    ./configure

4. run (This command is to cover the automatically generated makefile).:

    cp makefile_autolens makefile

5. run:

    make

If you see libnnhpi_customized.so, then it should be correctly compiled. To make sure it works, you can go to 
test_autoarray, and run 'pytest test_autoarray/nn_tools'

To clean all compiled fields, run:

   make distclean


