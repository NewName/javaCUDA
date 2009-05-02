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

package org.lambdacomplex.nn.javacuda;

import org.lambdacomplex.nn.javacuda.array.NativeByteArray;
import org.lambdacomplex.nn.javacuda.swig.*;

/**
 * A pointer to memory on the GPU.
 * 
 * This class provides methods for moving data to and from the GPU.
 * @author NewName
 *
 */
public class DevicePointer {
	/*public static DevicePointer toDevice(byte[] data) {
		DevicePointer result = new DevicePointer(data.length);
		result.copy(data);
		return result;
	}*/
	
	/**
	 * Automatically allocate space for and move the data inside the given native byte array to the device.
	 * @param data The NativeByteArray object containing the data.
	 * @return A pointer to the data on the device.
	 */
	public static DevicePointer toDevice(NativeByteArray data) {
		DevicePointer result = new DevicePointer(data.getByteSize());
		result.copyFrom(data);
		return result;
	}
	
	private CUDevicePointer devPtr;
	private long size;
	protected boolean freed;
	
	/**
	 * Allocate memory on the device. This pointer will then point to it.
	 * @param size The amount of memory to allocate in bytes.
	 */
	public DevicePointer(long size) {
		devPtr = new CUDevicePointer();
		this.size = size;
		Util.safeCall(Cuda.cuMemAlloc(devPtr.cast(), size));
		freed = false;
	}

	/*public void copy(byte[] data) {
		if (data.length > size) throw new IllegalArgumentException();

		//SWIGTYPE_p_void voidData = Cuda.toPVoid(data);
		CUByteArray b = new CUByteArray(data.length);
		for (int i = 0; i <data.length; i++) b.setitem(i, data[i]);
		SWIGTYPE_p_void voidData = Cuda.toPVoid(b.cast());
		
		Util.safeCall(Cuda.cuMemcpyHtoD(devPtr.value(), voidData, 1 + 0*size));
		
	}*/
	
	/**
	 * Move the data inside the given native byte array into the memory pointed to by this pointer.
	 * @param data The NativeByteArray object containing the data.
	 */
	public void copyFrom(NativeByteArray data) {
		if (freed) throw new IllegalStateException();
		if (data.getByteSize() > size) throw new IllegalArgumentException();
		
		SWIGTYPE_p_void voidData = Cuda.toPVoid(data.getNativePointer());
		Util.safeCall(Cuda.cuMemcpyHtoD(devPtr.value(), voidData, size));
	}
	
	/**
	 * Move the data pointed to by this pointer into the given native byte array.
	 * @param dest The NativeByteArray object to copy the data into.
	 */
	public void copyTo(NativeByteArray dest) {
		if (freed) throw new IllegalStateException();
		if (dest.getByteSize() < size) throw new IllegalArgumentException();
		
		SWIGTYPE_p_void voidDest = Cuda.toPVoid(dest.getNativePointer());
		Util.safeCall(Cuda.cuMemcpyDtoH(voidDest, devPtr.value(), size));
	}
	
	/**
	 * Deallocate the memory pointed to by this pointer.
	 */
	public void free() {
		Util.safeCall(Cuda.cuMemFree(devPtr.value()));
	}
	
	/*public void finalize() throws Throwable {
		try {
		
		} finally {
			super.finalize();
		}
	}*/
	
	protected long getValue() {
		return devPtr.value();
	}
}
