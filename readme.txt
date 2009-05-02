Before starting you'll need the following installed;

	swig, sh, gcc, jdk, cuda

This library has been created using eclipse, so if you have it installed 
it's probably best to use it.

The java source code is located in the 'src' directory in the package 
'org.lambdacomplex.nn.javacuda'.
The java class files can be built where ever you want, eclipse creates a 
'bin' folder and puts them in there.

The java source wont compile until you have generated the interface 
files using swig. I've set up a script to generate both the interface 
files and the C source for the wrapper dll and compile them.

This script is located in the 'swig' directory. It is written for sh so 
if you're running windows download MSYS or use cygwin. 

Open up the 'config.sh' file in the swig directory and set the path to 
your JDK include folder, you also need to set the path to the 
subdirectory that contains the platform specific headers.

If you're using the mingw compiler leave the linker options be, if 
you're using linux gcc then remove those options. If you're using the 
cygwin gcc then you'll probably have add an option to stop it linking to 
cygwin.dll (unless that's what you want).

If you're using MSVC then you'll probably have to edit the whole compile 
script too.

On windows the CUDA intaller automatically sets the environment 
variables CUDA_INC_PATH and CUDA_LIB_PATH to the directories that 
contain the CUDA include files and CUDA libraries respectively. I don't 
know if the linux installer does this so you may need to set them 
yourself (just stick them in config).

Now run 'compile.sh'. It will place the interface files where they need 
to go and put the wrapper DLL in the 'working' directory. Now compile 
the java source. If you decide to run some tests then your working 
director will have to be the 'working' directory (so that the JRE can 
find the wrapper DLL).

The 'test_src' folder contains tests and examples.
