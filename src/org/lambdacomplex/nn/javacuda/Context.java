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
 * This class encapsulates the CUDA context.
 * 
 * CUDA contexts are often associated with threads, so many static methods are used.
 * 
 * Create contexts using the Device class.
 * @author NewName
 *
 */
public class Context {
	
	/**
	 * Increment the usage count on the context currently associated with this thread. When done with this context, the detach method should be called on it.
	 * 
	 * Equivalent to cuCtxAttach.
	 * @return The context currently associated with this thread.
	 */
	public static Context attachCurrent() {
		CUPContext ctx = new CUPContext();
		int flags = 0; // this is a must, as stated in the reference
		Util.safeCall(Cuda.cuCtxAttach(ctx.cast(), flags));
		
		CPint device = new CPint();
		Util.safeCall(Cuda.cuCtxGetDevice(device.cast()));
		
		Context result = new Context(ctx);
		result.deviceID = device.value();
		return result; 
	}
	
	/**
	 * Disassociates the current context from the thread (popping it from the context stack).
	 * 
	 * The context must have a usage count of 1. The context then becomes a floating context.
	 * @return The context previously associated with this thread (the context on the top of the stack).
	 */
	public static Context popCurrent() {
		CUPContext ctx = new CUPContext();
		Util.safeCall(Cuda.cuCtxPopCurrent(ctx.cast()));
		return new Context(ctx);
	}
	
	/**
	 * Get the device associated with the context which is associated with the current thread.
	 * 
	 * Equivalent to cuCtxGetDevice.
	 * @return The device associated with the current context.
	 */
	public static Device getCurrentDevice() {
		CPint result = new CPint();
		Util.safeCall(Cuda.cuCtxGetDevice(result.cast()));
		return Device.getDevice(result.value());
	}
	
	/**
	 * Block the thread until the current context's tasks are completed.
	 * 
	 * This method throws an exception if one of the tasks fails.
	 */
	public static void syncWithCurrent() {
		Util.safeCall(Cuda.cuCtxSynchronize());
	}
	
	/**
	 * Load the given Cubin onto the GPU, to ready it for use.
	 * @param cubin The Cubin object to be loaded.
	 * @return The resulting module on the GPU.
	 */
	public static Module loadCubin(Cubin cubin) {
		return new Module(cubin);
	}
	
	protected CUPContext context;
	private Flags flag;
	private int deviceID;
	
	protected Context(int deviceID) {
		context = null;
		this.deviceID = deviceID;
		flag = Flags.SCHEDULER_AUTO;
	}
	
	protected Context(CUPContext context) {
		this.context = context;
	}
	
	public enum Flags {
		SCHEDULER_AUTO (CUctx_flags.CU_CTX_SCHED_AUTO),
		SCHEDULER_SPIN (CUctx_flags.CU_CTX_SCHED_SPIN),
		SCHEDULER_YIELD (CUctx_flags.CU_CTX_SCHED_YIELD);
		
		protected CUctx_flags flag;
		Flags (CUctx_flags f) { flag = f; }
	}
	
	/**
	 * Set the flag this context uses during initialisation.
	 * @param flag The flag to be used.
	 */
	public void setFlag(Flags flag) {
		if (context != null) throw new IllegalStateException("Context already created.");
		this.flag = flag;
	}
	
	/**
	 * Initialise this context and associate it with the current thread.
	 * 
	 * Equivalent to cuCtxCreate. Flags are set using the setFlag method.
	 */
	public void create() {
		if (context != null) throw new IllegalStateException("Context allready created.");
		context = new CUPContext();
		Util.safeCall(Cuda.cuCtxCreate(context.cast(), flag.flag.swigValue(), deviceID));
	}
	
	/**
	 * Decrement the current usage count by 1, if the count reaches zero then destroy this context.
	 * 
	 * This context must be associated with the calling thread.
	 */
	public void detach() {
		Util.safeCall(Cuda.cuCtxDetach(context.value()));
	}
	
	/**
	 * Associate this context with the current thread (push it onto the stack).
	 * 
	 * The context must be floating (not associated with any thread).
	 */
	public void push() {
		Util.safeCall(Cuda.cuCtxPushCurrent(context.value()));
	}
	
	/**
	 * Destroy this context.
	 * 
	 * If the usage count is not 1, or this context is being used by another thread this function will fail with an exception. Equivalent to cuCtxDestroy.
	 */
	public void destroy() {
		Util.safeCall(Cuda.cuCtxDestroy(context.value()));
	}
}
