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

import org.lambdacomplex.nn.javacuda.*;

public class NativeIntArray extends NativeByteArray {
	public static final int intSize = 4;
	
	public NativeIntArray(int size) {
		super(size*intSize);
	}
	
	public NativeIntArray(Context ctx, int size) {
		super(ctx,size*intSize);
	}
	
	public void setInt(int index, int i) {
		index = index * intSize;
		setByte(index+0,(byte)i);
		setByte(index+1,(byte)(i>>8));
		setByte(index+2,(byte)(i>>16));
		setByte(index+3,(byte)(i>>24));
	}
	
	public int getInt(int index) {
		index = index*intSize;
		return
			 (((int)getByte(index+0) & 0xFF))
			+(((int)getByte(index+1) & 0xFF) << 8)
			+(((int)getByte(index+2) & 0xFF) << 16)
			+(((int)getByte(index+3) & 0xFF) << 24);
	}
	
	public int getSize() {
		return getByteSize()/intSize;
	}
}
