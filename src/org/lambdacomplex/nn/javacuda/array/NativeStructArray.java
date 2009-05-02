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

import org.lambdacomplex.nn.javacuda.NativeStruct;

public class NativeStructArray<T extends NativeStruct> extends NativeByteArray {
	private int length;
	
	public NativeStructArray(int size, int length) {
		super(size * length);
		this.length = length;
	}
	
	public void setStruct(int index, NativeStruct s) {
		s.serialiseInto(this, index * length);
	}
	
	public int getSize() {
		return getByteSize()/length;
	}
}
