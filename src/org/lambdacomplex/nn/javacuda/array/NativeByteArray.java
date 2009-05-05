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

package org.lambdacomplex.nn.javacuda.array;

import org.lambdacomplex.nn.javacuda.swig.*;
import org.lambdacomplex.nn.javacuda.*;

public class NativeByteArray {
	static {
		System.loadLibrary("CUDAWrapper");
	}
	
	private CUByteArray backer;
	private int size;
	private boolean pageLocked;
	private Context context;
	
	public enum Type {
		Paged,
		PageLocked
	}
	
	public NativeByteArray(int size) {
		backer = new CUByteArray(size);
		this.size = size;
		pageLocked = false;
	}
	
	public NativeByteArray(Context ctx, final int size) {
		final CUPvoid mem = new CUPvoid();
		ctx.run(new Runnable(){
			public void run() { 
				CUresult result = Cuda.cuMemAllocHost(mem.cast(), size);
				if ( result != CUresult.CUDA_SUCCESS)
					throw new CudaAPIError(result.toString());
			}
		});
		backer = CUByteArray.frompointer(Cuda.toByteArray(mem.value()));
		context = ctx;
		this.size = size;
		pageLocked = true;
	}
	
	public void setByte(int index, byte b) {
		backer.setitem(index, b);
	}
	
	public byte getByte(int index) {
		return backer.getitem(index);
	}
	
	public SWIGTYPE_p_signed_char getNativePointer() {
		return backer.cast();
	}
	
	public int getSize() {
		return getByteSize();
	}
	
	public int getByteSize() {
		return size;
	}
	
	public boolean isFreed() {
		return backer == null;
	}
	
	public void free() {
		if (isFreed()) return;
		if (!pageLocked) {
			backer.delete();
		} else {
			context.run(new Runnable(){
				public void run() { 
					CUresult result = Cuda.cuMemFreeHost(Cuda.toPVoid(backer.cast()));
					if ( result != CUresult.CUDA_SUCCESS)
						throw new CudaAPIError(result.toString());
				}
			});
		}
		backer = null;
	}
	
	public void finalize() throws Throwable {
		try {
			free();
		} finally {
			super.finalize();
		}
	}
}
