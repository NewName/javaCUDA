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

import org.lambdacomplex.nn.javacuda.swig.*;

/**
 * A function inside a module.
 * 
 * This class provides methods for calling the function.
 * 
 * Create Function objects using the Module class.
 * @author NewName
 *
 */
public class Function {
	private CUPFunction function;
	private GridSize grid;
	private BlockSize block;
	
	protected Function(CUPModule module, String name) {
		function = new CUPFunction();
		Util.safeCall(Cuda.cuModuleGetFunction(function.cast(), module.value(), name));
	}
	
	/**
	 * Set the block size.
	 * @param size The size of the block.
	 */
	public void setBlockSize(BlockSize size) {
		Util.safeCall(Cuda.cuFuncSetBlockShape(function.value(), size.x, size.y, size.z));
		block = size;
	}
	
	/**
	 * Get the block size.
	 * @return The size of the block.
	 */
	public BlockSize getBlockSize() {
		return block;
	}
	
	/**
	 * Set the grid size.
	 * @param size The size of the grid.
	 */
	public void setGridSize(GridSize size) {
		grid = size;
	}
	
	/**
	 * Set the amount of shared memory for each thread block to use.
	 * @param size The amount of memory in bytes.
	 */
	public void setSharedMemory(long size) {
		Util.safeCall(Cuda.cuFuncSetSharedSize(function.value(), size));
	}
	
	/**
	 * Call the function with the given arguments. This will launch the module on the associated device. 
	 * 
	 * The arguments are arranged in an array. Each element of the array corresponds to a single argument for the function.
	 * 
	 * Use the correct Argument types to ensure type safety.
	 * @param args An array containing the arguments for the function.
	 */
	public void call(Argument[] args) {
		int offset = 0;
		for (Argument a : args) {
			offset += a.setParam(function, offset);
		}
		Util.safeCall(Cuda.cuParamSetSize(function.value(), offset));
		Util.safeCall(Cuda.cuLaunchGrid(function.value(), grid.x, grid.y));
	}
	
	public static class BlockSize {
		protected int x,y,z;
		public BlockSize(int x, int y, int z) {
			this.x = x; this.y = y; this.z = z;
		}
	}
	
	public static class GridSize {
		protected int x,y;
		public GridSize(int x, int y) {
			this.x = x; this.y = y;
		}
		public static GridSize toFit(BlockSize block, int x, int y) {
			return new GridSize(x/block.x + 1, y/block.y + 1);
		}
	}
	
	public static abstract class Argument {
		protected abstract int setParam(CUPFunction func, int offset);
	}
	
	/**
	 * A device pointer argument to a function.
	 * @author NewName
	 *
	 */
	public static class PointerArgument extends Argument {
		private DevicePointer ptr;
		public PointerArgument(DevicePointer p) { ptr = p; }
		protected int setParam(CUPFunction func, int offset) {
			Util.safeCall(Cuda.cuParamSeti(func.value(), offset, ptr.getValue()));
			return 4;
		}
	}
	
	/**
	 * An integer argument to a function.
	 * @author NewName
	 *
	 */
	public static class IntegerArgument extends Argument {
		private long integer;
		public IntegerArgument(int i) { integer = i; }
		protected int setParam(CUPFunction func, int offset) {
			Util.safeCall(Cuda.cuParamSeti(func.value(), offset, integer));
			return 4;
		}
	}
}
