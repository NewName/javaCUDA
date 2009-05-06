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
	
	/**
	 * Automatically allocate space for and move the data inside the given native byte array to the device.
	 * @param data The NativeByteArray object containing the data.
	 * @return A pointer to the data on the device.
	 */
	public static DevicePointer toDevice(Context ctx, NativeByteArray data) {
		DevicePointer result = new DevicePointer(ctx, data.getByteSize());
		result.copyFrom(data);
		return result;
	}
	
	public static DevicePointer toDeviceAsync(Context ctx, NativeByteArray data, Stream stream) {
		DevicePointer result = new DevicePointer(ctx, data.getByteSize());
		result.copyFromAsync(data, stream);
		return result;
	}
	
	private Context context;
	private CUDevicePointer devPtr;
	private long size;
	
	/**
	 * Allocate memory on the device. This pointer will then point to it.
	 * @param size The amount of memory to allocate in bytes.
	 */
	protected DevicePointer(Context ctx, long size) {
		context = ctx;
		devPtr = new CUDevicePointer();
		devPtr.setMemoryManaged(false);
		this.size = size;
		synchronized (context) {
			context.push();
			Util.safeCall(Cuda.cuMemAlloc(devPtr.cast(), size));
			Context.popCurrent();
		}
	}
	
	/**
	 * Move the data inside the given native byte array into the memory pointed to by this pointer.
	 * @param data The NativeByteArray object containing the data.
	 */
	public void copyFrom(NativeByteArray data) {
		if (isFreed()) throw new IllegalStateException();
		if (data.getByteSize() > size) throw new IllegalArgumentException();
		
		SWIGTYPE_p_void voidData = Cuda.toPVoid(data.getNativePointer());
		synchronized (context) {
			context.push();
			Util.safeCall(Cuda.cuMemcpyHtoD(devPtr.value(), voidData, data.getByteSize()));
			Context.popCurrent();
		}
	}
	
	public void copyFromAsync(NativeByteArray data, Stream stream) {
		if (isFreed()) throw new IllegalStateException();
		if (data.getByteSize() > size) throw new IllegalArgumentException();
		
		SWIGTYPE_p_void voidData = Cuda.toPVoid(data.getNativePointer());
		synchronized (context) {
			context.push();
			Util.safeCall(Cuda.cuMemcpyHtoDAsync(devPtr.value(), voidData, data.getByteSize(), stream.getValue().value()));
			Context.popCurrent();
		}
	}
	
	/**
	 * Move the data pointed to by this pointer into the given native byte array.
	 * @param dest The NativeByteArray object to copy the data into.
	 */
	public void copyTo(NativeByteArray dest) {
		if (isFreed()) throw new IllegalStateException();
		if (dest.getByteSize() < size) throw new IllegalArgumentException();
		
		SWIGTYPE_p_void voidDest = Cuda.toPVoid(dest.getNativePointer());
		synchronized (context) {
			context.push();
			Util.safeCall(Cuda.cuMemcpyDtoH(voidDest, devPtr.value(), size));
			Context.popCurrent();
		}
	}
	
	public void copyToAsync(NativeByteArray dest, Stream stream) {
		if (isFreed()) throw new IllegalStateException();
		if (dest.getByteSize() < size) throw new IllegalArgumentException();
		
		SWIGTYPE_p_void voidDest = Cuda.toPVoid(dest.getNativePointer());
		synchronized (context) {
			context.push();
			Util.safeCall(Cuda.cuMemcpyDtoHAsync(voidDest, devPtr.value(), size, stream.getValue().value()));
			Context.popCurrent();
		}
	}
	
	/**
	 * Returns whether the memory pointed to by this poiner has been freed.
	 * @return Whether the memory is freed.
	 */
	public boolean isFreed() {
		return devPtr == null;
	}
	
	/**
	 * Deallocate the memory pointed to by this pointer.
	 */
	public void free() {
		synchronized (context) {
			if (isFreed()) return;
			context.push();
			Util.safeCall(Cuda.cuMemFree(devPtr.value()));
			Context.popCurrent();
			devPtr.delete();
			devPtr = null;
		}
	}
	
	public void finalize() throws Throwable {
		try {
			free();
		} finally {
			super.finalize();
		}
	}
	
	protected long getValue() {
		return devPtr.value();
	}
}
