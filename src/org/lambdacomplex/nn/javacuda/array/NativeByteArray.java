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

public class NativeByteArray {
	static {
		System.loadLibrary("CUDAWrapper");
	}
	
	private CUByteArray backer;
	private int size;
	
	public NativeByteArray(int size) {
		backer = new CUByteArray(size);
		this.size = size;
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
}
