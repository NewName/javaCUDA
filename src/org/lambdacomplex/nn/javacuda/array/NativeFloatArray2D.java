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

public class NativeFloatArray2D {
	private NativeFloatArray backer;
	int xDim, yDim;
	
	public NativeFloatArray2D(int x, int y) {
		backer = new NativeFloatArray(x*y);
		xDim = x; yDim = y;
	}

	public void setFloat(int x, int y, float f) {
		backer.setFloat(x + y * xDim, f);
	}
	
	public float getFloat(int x, int y) {
		return backer.getFloat(x + y * xDim);
	}
	
	public NativeFloatArray get1DArray() {
		return backer;
	}
}
