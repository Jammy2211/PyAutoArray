## nn ##
(Natural Neighbours interpolation for PyAutoLens)

To install nn for PyAutoLens on a linux machine, follow the steps below:

1. Put directory 'nn' folders under LD\_LIBRARY\_PATH. 
e.g. add a line to ~/.bashrc like "export export LD\_LIBRARY\_PATH=$LD\_LIBRARY\_PATH:/your/path/to/autoarray/nn\_tools/nn\_c\_sources/nn"

2. go to directory 'nn'

3. run: 
./configure

4. run:
cp makefile\_autolens makefile
This command is to cover the aumatically generated makefile. 

5. run:
make

If you see libnnhpi\_customized.so, then it should be correctly compiled. To make sure it works, you can go to test\_autoarray, and run 'pytest nn\_tools/test\_nn\_c\_tools.py'


To clean all compiled fiels, please run:
make distclean 


