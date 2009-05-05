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

package org.lambdacomplex.nn.javacuda.test;

import java.io.*;
import java.awt.*;
import javax.swing.*;

import org.lambdacomplex.nn.javacuda.*;
import org.lambdacomplex.nn.javacuda.array.*;

public class Mandelbrot {
	
	
	public static void main(String[] args) throws Throwable {
		doit();
	}
	
	public static void doit() throws Throwable {
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
		Function function = module.getFunction("squarecmplx");
		
		System.out.println("Allocating input");
		NativeFloatArray2D real = new NativeFloatArray2D(1000, 1000);
		NativeFloatArray2D imaginary = new NativeFloatArray2D(1000, 1000);
		for (int y = 0; y < 1000; y++) {
			for (int x = 0; x < 1000; x++) {
				real.setFloat(x,y, -2 + x*(4f/1000f));
				imaginary.setFloat(x,y, -2 + y*(4f/1000f));
			}
		}
		
		
		System.out.println("Loading input");
		DevicePointer real_gpu = DevicePointer.toDevice(ctx, real.get1DArray());
		DevicePointer imaginary_gpu = DevicePointer.toDevice(ctx, imaginary.get1DArray());
		
		System.out.println("Calling kernel");
		function.setBlockSize(new Function.BlockSize(16,16,1));
		function.setGridSize(Function.GridSize.toFit(function.getBlockSize(), 1000, 1000));
		
		function.setSharedMemory(33);
		function.call(new Function.Argument[]{
				new Function.PointerArgument(real_gpu),
				new Function.PointerArgument(imaginary_gpu),
				new Function.IntegerArgument(1000),
				new Function.IntegerArgument(1000),
			});
		
		System.out.println("Retreiving data");
		NativeFloatArray resultBytes = new NativeFloatArray(1000*1000);
		real_gpu.copyTo(resultBytes);
		
		System.out.println("Displaying image");
		float[][] result = new float[1000][1000];
		for (int y = 0; y < 1000; y++) {
			for (int x = 0; x < 1000; x++) {
				int i = x + y * 1000;
				result[x][y] = resultBytes.getFloat(i);
			}
		}
		
		displayResults(result);
	}
	
	public static void displayResults(final float[][] results) {
		SwingUtilities.invokeLater(new Runnable(){
			public void run() {
				JFrame frame = new JFrame("test");
				frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
				frame.setPreferredSize(new Dimension(1000,1000));
				frame.add(new MyPanel(results));
				frame.pack();
				frame.setVisible(true);
			}
		});
	}
	
	public static class MyPanel extends JPanel {
		private static final long serialVersionUID = 1L;
		float[][] results;
		public MyPanel(float[][] r) {
			super();
			results = r;
			setPreferredSize(new Dimension(r.length, r[0].length));
		}
		protected void paintComponent(Graphics g) {
			Graphics2D g2 = (Graphics2D)g;
			for (int x=0;x<results.length; x++) {
				for (int y=0;y<results[0].length; y++) {
					g2.setColor(Color.getHSBColor(
							results[x][y]/5f,
							((results[x][y]/13f)%1),
							results[x][y] < -100 ? 0f : 0.7f
						));
					g2.drawLine(x, y, x+1, y);
				}
			}
		}
	}
	
	public static Reader getKernel() {
		String kernel = 
		"extern \"C\" __global__ void squarecmplx(float *real, float *imaginary,"+
		"			 int resX, int resY) {"+ //int *itr, , int maxItr
		"    int x = blockIdx.x * blockDim.x + threadIdx.x;"+
		"    int y = blockIdx.y * blockDim.y + threadIdx.y;"+

		"    if (x >= resX || y >= resY) return;"+

		"    int index = x + y * resX;"+
		"    float initRx = real[index];"+
		"    float initIm = imaginary[index];"+
		"    float rx = initRx, im = initIm;"+

		"	int i;"+
		"    for (i = 0; i < 100000; i++) {"+
		"        float oldrx = rx;"+
		"        rx = rx*rx - im*im + initRx;"+
		"        im = 2 * oldrx * im + initIm;"+
		"		if (rx*rx + im*im > 4) {"+
		"			break;"+
		"		}"+
		"    }"+

		"	for (int j = 0; j < 3; j++) {"+
		"        float oldrx = rx;"+
		"        rx = rx*rx - im*im + initRx;"+
		"        im = 2 * oldrx * im + initIm;"+
		"	}"+
		
		"	if (i == 100000) {"+
		"		real[index] = -101;"+
		"	} else {"+
		"		real[index] = ((float)i) + 1 - (logf(logf(sqrtf(rx*rx+im*im))))/(logf(2));"+
		"	}"+
		"}"
		;
		return new StringReader(kernel);
	}
}
