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
 * A stream for managing asynchronous activities.
 * @author NewName
 *
 */
public class Stream {
	private Context context;
	private CUPStream stream;
	
	/**
	 * Create a stream
	 */
	protected Stream(Context ctx) {
		context = ctx;
		stream = new CUPStream();
		stream.setMemoryManaged(false);
		long flags = 0; // currently required by the API
		synchronized (context) {
			context.push();
			Util.safeCall(Cuda.cuStreamCreate(stream.cast(), flags));
			Context.popCurrent();
		}
	}
	
	protected Stream(Context ctx, CUPStream st) {
		stream = st;
		context = ctx;
	}
	
	public boolean isDestroyed() {
		return stream == null;
	}
	
	/**
	 * Destroy this stream
	 */
	public void destroy() {
		synchronized (context) {
			if (isDestroyed()) return;
			context.push();
			Util.safeCall(Cuda.cuStreamDestroy(stream.value()));
			Context.popCurrent();
			stream.delete();
			stream = null;
		}
	}
	
	public void finalize() throws Throwable {
		try {
			destroy();
		} finally {
			super.finalize();
		}
	}
	
	/**
	 * Queries this stream on the status of it's tasks.
	 * 
	 * Returns true if all tasks are complete, false if tasks still remain to be completed.
	 * @return Whether tasks still remain to be completed.
	 */
	public boolean isReady() {
		CUresult result;
		synchronized (context) {
			context.push();
			result = Cuda.cuStreamQuery(stream.value());
			Context.popCurrent();
		}
		
		if (result == CUresult.CUDA_SUCCESS) {
			return true;
		} else if (result == CUresult.CUDA_ERROR_NOT_READY) {
			return false;
		} else {
			// this will throw an exception, so the return value is unimportant
			Util.safeCall(result);
			return false;
		}
	}
	
	/**
	 * Blocks the current thread until all associated tasks are complete.
	 */
	public void synchronise() {
		synchronized (context) {
			context.push();
			Util.safeCall(Cuda.cuStreamSynchronize(stream.value()));
			Context.popCurrent();
		}
	}
	
	/**
	 * An event in the processing queue of a stream.
	 * @author NewName
	 *
	 */
	public static class Event {
		private Context context;
		private CUPEvent event;
		
		protected Event(Stream stream) {
			context = stream.context;
			event = new CUPEvent();
			event.setMemoryManaged(false);
			long flags = 0; // currently required by the API
			synchronized (context) {
				context.push();
				Util.safeCall(Cuda.cuEventCreate(event.cast(), flags));
				Util.safeCall(Cuda.cuEventRecord(
						event.value(), 
						stream.stream.value()
					));
				Context.popCurrent();
			}
		}
		
		/**
		 * Returns true if the associated stream has reached this event.
		 * @return If this event has been reached.
		 */
		public boolean isReached() {
			CUresult result;
			synchronized (context) {
				context.push();
				result = Cuda.cuEventQuery(event.value());
				Context.popCurrent();
			}
			
			if (result == CUresult.CUDA_SUCCESS) {
				return true;
			} else if (result == CUresult.CUDA_ERROR_NOT_READY) {
				return false;
			} else {
				// this will throw an exception, so the return value is unimportant
				Util.safeCall(result);
				return false;
			}
		}
		
		/**
		 * Blocks the current thread until this event has been reached by the associated stream.
		 */
		public void synchronise() {
			synchronized (context) {
				context.push();
				Util.safeCall(Cuda.cuEventSynchronize(event.value()));
				Context.popCurrent();
			}
		}
		
		public boolean isDestroyed() {
			return event == null;
		}
		
		/**
		 * Destroy this event.
		 */
		public void destroy() {
			synchronized (context) {
				if (isDestroyed()) return;
				context.push();
				Util.safeCall(Cuda.cuEventDestroy(event.value()));
				Context.popCurrent();
				event.delete();
				event = null;
			}
		}
		
		public void finalize() throws Throwable {
			try {
				destroy();
			} finally {
				super.finalize();
			}
		}
		
		/**
		 * Find the amount of time between two events in milliseconds.
		 * 
		 * The resolution of this timing is 0.5 microseconds. This method is only defined for events that are associated with the default stream.
		 * @param start The starting event.
		 * @param end The finishing event.
		 * @return The time in milliseconds between when each event was reached.
		 */
		public static float elapsedTime(Event start, Event end) {
			// TODO: create a specfic exception if the events have not been recorded yet.
			// TODO: find out the requirements on context.
			Context context = start.context;
			CPfloat result = new CPfloat();
			synchronized (context) {
				context.push();
				Util.safeCall(Cuda.cuEventElapsedTime(
					result.cast(), start.event.value(), end.event.value()
				));
				Context.popCurrent();
			}
			return result.value();
		}
	}
	
	/**
	 * Create an event associated with this stream.
	 * 
	 * The event is then automatically placed at this point in the stream's queue.
	 * @return The new Event.
	 */
	public Event createEvent() {
		return new Event(this);
	}
	
	protected CUPStream getValue() {
		return stream;
	}
}
