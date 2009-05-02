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

import java.io.*;

/**
 * This class encapsulates modules in their source and compiled forms.
 * 
 * The compiled data exists as a temporary file on the hard drive of the computer.
 * @author NewName
 *
 */
public class Cubin {
	
	private Reader source;
	private File cubin;
	private boolean compiled;
	
	/**
	 * Create a Cubin object from a pre-compiled module.
	 * @param moduleFile The file containing the pre-compiled module.
	 */
	public Cubin(File moduleFile) {
		cubin = moduleFile;
		compiled = true;
	}
	
	/**
	 * Create a Cubin from source code.
	 * @param moduleSource A Reader object from which the source will be read from.
	 */
	public Cubin(Reader moduleSource) {
		source = moduleSource;
		compiled = false;
	}
	
	/**
	 * Compile this Cubin.
	 * @throws CudaCompileException
	 */
	public void compile() throws CudaCompileException {
		try {
			compile(null);
		} catch (IOException e) {
			// never happens
		}
	}
	
	/**
	 * Compile this Cubin and write the output to an output stream.
	 * @param output The Writer object to which the output of the compilation will be written.
	 * @throws CudaCompileException
	 * @throws IOException
	 */
	public void compile(Writer output) throws CudaCompileException,IOException {
		if (compiled) throw new IllegalStateException("Cubin allready compiled");
		
		File cu;
		
		FileWriter writer;
		try {
			cu = File.createTempFile("javaCuda", ".cu");
			cu.deleteOnExit();
			writer = new FileWriter(cu);
		} catch (IOException e) {
			throw new CudaCompileException("Unable to create temporary file: " + e.getLocalizedMessage());
		}
		int c;
		try {
			while ((c = source.read()) != -1) writer.write(c);
			writer.close();
		} catch (IOException e) {
			throw new CudaCompileException("Unable to copy module to cu file: " + e.getLocalizedMessage());
		}
		
		try {
			cubin = File.createTempFile("javaCuda", ".cubin");
			cubin.deleteOnExit();
		} catch (IOException e) {
			throw new CudaCompileException("Unable to create temporary file: " + e.getLocalizedMessage());
		}
		
		Process p;
		try {
			String cudaDir = System.getenv("CUDA_BIN_PATH");
			p = Runtime.getRuntime().exec(
					cudaDir + "/nvcc.exe -cubin " + 
					cu.getName() + 
					" -o " + cubin.getName(),
					null, cu.getParentFile());
		} catch (IOException e) {
			throw new CudaCompileException("Unable to execute nvcc: " + e.getLocalizedMessage());
		}
		
		if (output != null) {
			try {
				BufferedReader input = new BufferedReader(new InputStreamReader(p
						.getInputStream()));
				while ((c = input.read()) != -1) {
					output.write(c);
				}
				input.close();
			} catch (IOException e) {
				throw e;
			}
		} 
		
		boolean finished = false;
		while (!finished) {
			try {
				p.waitFor();
				finished = true;
			} catch (InterruptedException e1) {

			}
		}
		if (p.exitValue() == -1) {
			StringBuffer error = new StringBuffer();
			try {
				String line;
				BufferedReader input = new BufferedReader(new InputStreamReader(p
						.getErrorStream()));
				while ((line = input.readLine()) != null) {
					error.append(line);
					error.append('\n');
				}
				input.close();
			} catch (IOException e) {
				throw new CudaCompileException("Error compiling module:\nError while reading error.");
			}
			throw new CudaCompileException("Error compiling module:\n" + error);
		}
		
		compiled = true;
	}
	
	/**
	 * Get the compiled Cubin file on the hard drive.
	 * @return A File object describing the location of the Cubin file.
	 */
	
	protected File getCubinFile() {
		return cubin;
	}
}
