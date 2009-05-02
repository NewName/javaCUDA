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

/**
 * An interface for objects that can be serialised into byte streams that resemble a native struct.
 * @author NewName
 *
 */
public interface NativeStruct {
	/**
	 * Serialise this object into the given native byte array starting at the given index.
	 * @param array The array to serialise into.
	 * @param startIndex The index in the given array to start from.
	 */
	public void serialiseInto(NativeByteArray array, int startIndex);
	
	/**
	 * Reconstruct this object from data given in the native byte array starting from the given index.
	 * @param array The array to reconstruct from.
	 * @param startIndex The index in the given array to start from.
	 */
	public void reconstructFrom(NativeByteArray array, int startIndex);
	
	/**
	 * The size of the byte stream that represents this object.
	 * 
	 * This size should include padding.
	 * @return The size in bytes.
	 */
	public int length();
}
