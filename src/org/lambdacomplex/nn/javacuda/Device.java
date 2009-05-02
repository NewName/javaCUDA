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
 * This class describes a physical GPU on the computer, and provides methods for obtaining devices.
 * @author NewName
 *
 */
public class Device {
	static int deviceCount;
	
	static {
		System.loadLibrary("CUDAWrapper");
		
		CPint count = new CPint();
		count.assign(0);
		if (Cuda.cuInit(0) == CUresult.CUDA_SUCCESS) 
			Util.safeCall(Cuda.cuDeviceGetCount(count.cast()));
		if (count.value() == 0)
			throw new NoCudaDeviceError();
		deviceCount = count.value();
	}
	
	/**
	 * Get the device with the given device-ID.
	 * 
	 * Device IDs start from 0.
	 * @param ord The device-ID.
	 * @return The Device object describing it.
	 */
	public static Device getDevice(int ord) {
		if (ord < 0 || ord >= deviceCount) throw new IndexOutOfBoundsException();
		
		return new Device(ord);
	}
	
	// TODO: CUdevprop doesn't have a multiprocessor count, find out why/alternatives
	/**
	 * Find the device with the most processing power.
	 * @return The Device object describing it.
	 */
	public static Device getMaxGFlopsDevice() {
		CPint count = new CPint();
		Util.safeCall(Cuda.cuDeviceGetCount(count.cast()));
		
		CUdevprop deviceProperties = new CUdevprop();
		int currentDevice = 0, maxDevice = 0, maxGFlops = -1;
		
		while (currentDevice < count.value()) {
			Util.safeCall(Cuda.cuDeviceGetProperties(deviceProperties, currentDevice));
			int GFlops = deviceProperties.getClockRate();
			
			if (GFlops > maxGFlops) {
				maxGFlops = GFlops;
				maxDevice = currentDevice;
			}
			currentDevice++;
		}
		
		return getDevice(maxDevice);
	}
	
	private int deviceID;
	
	protected Device(int ord) {
		deviceID = ord;
	}
	
	/**
	 * Gets a CUDA context for this device.
	 * @return The Context object encapsulating the CUDA context.
	 */
	public Context getContext() {
		return new Context(deviceID);
	}
}
