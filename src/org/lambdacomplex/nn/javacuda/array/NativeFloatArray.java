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

public class NativeFloatArray extends NativeIntArray {
	
	public NativeFloatArray(int size) {
		// float is same size as int
		super(size);
	}
	
	public NativeFloatArray(Context ctx, int size) {
		super(ctx, size);
	}
	
	public void setFloat(int index, float f) {
		setInt(index, Float.floatToIntBits(f));
	}
	
	public float getFloat(int index) {
		return Float.intBitsToFloat(getInt(index));
	}
}
