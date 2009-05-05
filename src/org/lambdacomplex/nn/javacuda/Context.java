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
	protected static Context attachCurrent() {
		CUPContext ctx = new CUPContext();
		int flags = 0; // this is a must, as stated in the reference
		Util.safeCall(Cuda.cuCtxAttach(ctx.cast(), flags));
		
		CPint device = new CPint();
		Util.safeCall(Cuda.cuCtxGetDevice(device.cast()));
		
		Context result = new Context(ctx);
		return result; 
	}
	
	/**
	 * Disassociates the current context from the thread (popping it from the context stack).
	 * 
	 * The context must have a usage count of 1. The context then becomes a floating context.
	 */
	protected static void popCurrent() {
		Util.safeCall(Cuda.cuCtxPopCurrent(null));
	}
	
	/**
	 * Get the device associated with the context which is associated with the current thread.
	 * 
	 * Equivalent to cuCtxGetDevice.
	 * @return The device associated with the current context.
	 */
	protected static Device getCurrentDevice() {
		CPint result = new CPint();
		Util.safeCall(Cuda.cuCtxGetDevice(result.cast()));
		return Device.getDevice(result.value());
	}
	
	/**
	 * Block the thread until the current context's tasks are completed.
	 * 
	 * This method throws an exception if one of the tasks fails.
	 */
	protected static void syncWithCurrent() {
		Util.safeCall(Cuda.cuCtxSynchronize());
	}
	
	protected CUPContext context;
	private Stream zeroStream = null;
	
	protected Context(int deviceID, Flags flag) {
		context = new CUPContext();
		context.setMemoryManaged(false);
		Util.safeCall(Cuda.cuCtxCreate(context.cast(), flag.flag.swigValue(), deviceID));
		popCurrent();
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
	
	public void synchronise() {
		synchronized (this) {
			push();
			syncWithCurrent();
			popCurrent();
		}
	}
	
	/**
	 * Load the given Cubin onto the GPU, to ready it for use.
	 * @param cubin The Cubin object to be loaded.
	 * @return The resulting module on the GPU.
	 */
	public Module loadCubin(Cubin cubin) {
		return new Module(this, cubin);
	}
	
	/**
	 * Decrement the current usage count by 1, if the count reaches zero then destroy this context.
	 * 
	 * This context must be associated with the calling thread.
	 */
	protected void detach() {
		Util.safeCall(Cuda.cuCtxDetach(context.value()));
	}
	
	/**
	 * Associate this context with the current thread (push it onto the stack).
	 * 
	 * The context must be floating (not associated with any thread).
	 */
	protected void push() {
		Util.safeCall(Cuda.cuCtxPushCurrent(context.value()));
	}
	
	public void run(Runnable r) {
		push();
		r.run();
		popCurrent();
	}
	
	public Stream getDefaultStream() {
		if (zeroStream == null) {
			CUPStream result = new CUPStream();
			result.assign(Cuda.toStream(0));
			zeroStream = new Stream(this, result);
		}
		
		return zeroStream;
	}
	
	/**
	 * Destroy this context.
	 */
	public void destroy() {
		synchronized (this) {
			if (context == null) return;
			push();
			Util.safeCall(Cuda.cuCtxDestroy(context.value()));
			context.delete();
			context = null;
		}
	}
	
	public void finalize() throws Throwable {
		try {
			destroy();
		} finally {
			super.finalize();
		}
	}
}
