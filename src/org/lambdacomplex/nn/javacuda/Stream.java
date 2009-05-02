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
	private CUPStream stream;
	
	/**
	 * Create a stream
	 */
	public Stream() {
		stream = new CUPStream();
		long flags = 0; // currently required by the API
		Util.safeCall(Cuda.cuStreamCreate(stream.cast(), flags));
	}
	
	protected Stream(CUPStream st) {
		stream = st;
	}
	
	/**
	 * Destroy this stream
	 */
	public void destroy() {
		Util.safeCall(Cuda.cuStreamDestroy(stream.value()));
	}
	
	/**
	 * Queries this stream on the status of it's tasks.
	 * 
	 * Returns true if all tasks are complete, false if tasks still remain to be completed.
	 * @return Whether tasks still remain to be completed.
	 */
	public boolean isReady() {
		CUresult result = Cuda.cuStreamQuery(stream.value());
		
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
		Util.safeCall(Cuda.cuStreamSynchronize(stream.value()));
	}
	
	/**
	 * An event in the processing queue of a stream.
	 * @author NewName
	 *
	 */
	public static class Event {
		private CUPEvent event;
		
		protected Event(Stream stream) {
			event = new CUPEvent();
			long flags = 0; // currently required by the API
			Util.safeCall(Cuda.cuEventCreate(event.cast(), flags));
			Util.safeCall(Cuda.cuEventRecord(
					event.value(), 
					stream.stream.value()
				));
		}
		
		/**
		 * Returns true if the associated stream has reached this event.
		 * @return If this event has been reached.
		 */
		public boolean isReached() {
			CUresult result = Cuda.cuEventQuery(event.value());
			
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
			Util.safeCall(Cuda.cuEventSynchronize(event.value()));
		}
		
		/**
		 * Destroy this event.
		 */
		public void destory() {
			Util.safeCall(Cuda.cuEventDestroy(event.value()));
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
			CPfloat result = new CPfloat();
			Util.safeCall(Cuda.cuEventElapsedTime(
					result.cast(), start.event.value(), end.event.value()
				));
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
	
	/**
	 * Create an event associated with the default stream.
	 * @return The new Event.
	 */
	public static Event createDefaultEvent() {
		return new Event(Util.zeroStream());
	}
	
	protected CUPStream getValue() {
		return stream;
	}
}
