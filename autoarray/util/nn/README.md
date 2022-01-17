## nn ##
(Natural Neighbours interpolation for PyAutoLens)

This code is mainly copied from "https://github.com/sakov/nn-c", a natural neighboring interpolation C code written 
by Pavel Sakov. Here, to make it run for PyAutoLens, we have slightly modefied it.

To install nn for PyAutoLens on a linux machine, follow the steps below:

1. Put directory 'nn' folders under LD_LIBRARY_PATH. 

 e.g. add a line to ~/.bashrc like "export export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/your/path/to/autoarray/util/nn/src/nn"

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


