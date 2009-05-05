/*
* Copyright 2009 Gabriel Collin
*
* This file is part of javaCUDA.
*
* javaCUDA is free software: you can redistribute it and/or modify it
* under the terms of the GNU Lesser General Public License as published
* by the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* javaCUDA is distributed in the hope that it will be useful, but WITHOUT
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public
* License for more details.
*
* You should have received a copy of the GNU Lesser General Public
* License along with javaCUDA. If not, see <http://www.gnu.org/licenses/>.
*/

 %module Cuda
 %{
 /* Includes the header in the wrapper code */
 #include "cuda.h"
 %}
 
 /* Parse the header file to generate wrappers */
 %include "cuda.h"

/*%typemap(javacode) SWIGTYPE, SWIGTYPE * %{
  public boolean equals(Object o) {
    if (!(o instanceof $javaclassname)) return false;
    return (( $javaclassname )o).swigCPtr == swigCPtr;
  }

  public int hashCode() {
    return (int)(swigCPtr % Integer.MAX_VALUE);
  }
%}*/


%include "arrays_java.i"
//%apply signed char[] {signed char *};

%include "carrays.i"
%array_class(signed char,CUByteArray);

%include "cpointer.i"
%typemap(javacode) SWIGTYPE %{
  protected boolean swigMemManage = true;

  public void setMemoryManaged(boolean b) {
    swigMemManage = b;
  }
%}

%typemap(javafinalize) SWIGTYPE %{
  protected void finalize() {
    if (!swigMemManage) return;
    delete();
  }
%}

%inline %{
typedef void* void_p;
%}


%pointer_class(int,CPint);
%pointer_class(float,CPfloat);
%pointer_class(void_p,CUPvoid);
%pointer_class(CUcontext,CUPContext);
%pointer_class(CUmodule,CUPModule);
%pointer_class(CUfunction,CUPFunction);
%pointer_class(CUdeviceptr,CUDevicePointer);
%pointer_class(CUstream,CUPStream);
%pointer_class(CUevent,CUPEvent);


%pointer_cast(void *, signed char *, toByteArray);
%pointer_cast(signed char *, void *, toPVoid);
%pointer_cast(int, CUstream, toStream);
