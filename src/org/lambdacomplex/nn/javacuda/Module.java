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
 * A module currently loaded on the GPU.
 * @author NewName
 *
 */
public class Module {
	private Context context;
	private CUPModule module;
	protected boolean loaded;
	
	protected Module(Context ctx, Cubin cubin) {
		context = ctx;
		module = new CUPModule();
		synchronized (context) {
			context.push();
			Util.safeCall(Cuda.cuModuleLoad(module.cast(), cubin.getCubinFile().getAbsolutePath()));
			Context.popCurrent();
		}
		loaded = true;
	}
	
	/**
	 * Get the function from this module with the given name.
	 * @param name The name of the function.
	 * @return The requested function.
	 */
	public Function getFunction(String name) {
		if (!loaded) throw new IllegalStateException();
		return new Function(context, module, name);
	}
	
	/**
	 * Unload the module from the device.
	 * 
	 * Call this method once the module becomes unused. This will prevent memory leaks.
	 */
	public void unload() {
		synchronized (context) {
			context.push();
			Util.safeCall(Cuda.cuModuleUnload(module.value()));
			Context.popCurrent();
		}
		loaded = false;
	}
	
	/*public void finalize() throws Throwable {
		try {

		} finally {
			super.finalize();
		}
	}*/
}
