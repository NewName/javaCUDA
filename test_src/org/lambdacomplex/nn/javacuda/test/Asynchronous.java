package org.lambdacomplex.nn.javacuda.test;

import java.io.*;

import org.lambdacomplex.nn.javacuda.*;
import org.lambdacomplex.nn.javacuda.array.*;

public class Asynchronous {
	public static void main(String args[]) {
		doit();
	}
	
	private static int inputSize = 2*1024*1024;
	
	public static void doit() {
		System.out.println("Allocating device");
		Device dev = Device.getMaxGFlopsDevice();
		Context ctx = dev.getContext();
		
		System.out.println("Compiling kernel");
		Cubin cubin = new Cubin(getKernel());
		try {
			cubin.compile();
		} catch (CudaCompileException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		
		System.out.println("Loading kernel");
		Module module = ctx.loadCubin(cubin);
		Function function = module.getFunction("increment_kernel");
		
		System.out.println("Allocating page-locked input");
		NativeIntArray input = new NativeIntArray(ctx, inputSize);
		for (int i = 0; i < inputSize; i++) {
			input.setInt(i, i);
		}
		
		System.out.println("Loading input and calling kernel");
		
		function.setBlockSize(new Function.BlockSize(256,1,1));
		function.setGridSize(new Function.GridSize(inputSize/256,1));
		
		long startTime = System.nanoTime();
		Stream.Event start = ctx.getDefaultStream().createEvent();		
			
		DevicePointer input_gpu = DevicePointer.toDeviceAsync(ctx, input, ctx.getDefaultStream());

		function.call(new Function.Argument[]{
				new Function.PointerArgument(input_gpu),
				new Function.IntegerArgument(25)
			});
		
		input_gpu.copyToAsync(input, ctx.getDefaultStream());
		
		long stopTime = System.nanoTime();
		Stream.Event stop = ctx.getDefaultStream().createEvent();
		
		long counter = 0;
		while (!stop.isReached()) counter++;
		
		float gpuTime = Stream.Event.elapsedTime(start, stop);
		float cpuTime = (float)(stopTime - startTime)/1000000;
		
		System.out.println("Time spent in CPU: " + cpuTime);
		System.out.println("Time spent in GPU: " + gpuTime);
		System.out.println("Number of CPU cycles elapsed waiting :" + counter);
		
		for (int i = 0; i < inputSize; i++) {
			if (input.getInt(i) - 25 != i) {
				System.out.println("GPU data incorrect!");
				System.out.println("Value at " + i + " is " + input.getInt(i));
				return;
			}
		}
		System.out.println("GPU data correct");
	}
	
	public static Reader getKernel() {
		String kernel = 
		"extern \"C\" __global__ void increment_kernel(int *g_data, int inc_value)" +
		"{"+ 
			"int idx = blockIdx.x * blockDim.x + threadIdx.x;"+
			"g_data[idx] = g_data[idx] + inc_value;"+
		"}"
		;
		return new StringReader(kernel);
	}
}
