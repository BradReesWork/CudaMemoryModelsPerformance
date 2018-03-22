/*
 * Copyright (c) 2017, NVIDIA CORPORATION. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */


#include "cmp_global.h"
#include "cuda_global.h"

#include "ArgParser.hpp"

//-----------------------------------------------------------------
/**
 * Main entry point
 */
int main(int argc, char * argv[]) {

	std::cout << "Memory Model performance Test - UNIFIED" << std::endl;

	//-----------------------------------------------------------------------------------
	//			Parse the arguments and compute some variables, then print
	//-----------------------------------------------------------------------------------
	auto args = ArgParser::parseArgs(argc, argv);
	ArgParser::printArgs(args);

    omp_set_num_threads(args->threads);
	std::cout << "Num of CPUs     " << omp_get_num_procs() << std::endl;
    std::cout << "Max threads     " << omp_get_max_threads() << std::endl;

	long numBuffElements =	args->buffElements; 		// this is just for readability

	// grab the cuda device
	init_cuda(args->device);		// init will exit on error

	//-----------------------------------------------------------------------------------
	//			Allocate Host Memory
	//-----------------------------------------------------------------------------------

	// the data buffer - what data is read into
	std::cout << "Allocating System Memory" << std::endl;
	Complex	*buffer;

	CHECK_CUDA(cudaMallocManaged((void**)&buffer, (sizeof(Complex) * numBuffElements) ));


	//-----------------------------------------------------------------------------------
	//			Initialize CUDA and Allocate memory
	//-----------------------------------------------------------------------------------
	cufftHandle		plan;
	std::cout << "Allocating GPU Memory - nothing to allocate" << std::endl;

	// create the FFT plan
	CHECK_FFT_STATUS( cufftPlan1d(&plan, args->fftSize, FFT_TYPE, BATCH_SIZE) );


	//-----------------------------------------------------------------------------------
	//			Process
	//-----------------------------------------------------------------------------------
	long recProcessed 	= 0;
	long recsToGet		= 0;
	long leftToProcess 	= args->records;

	while (leftToProcess > 0) {

		recsToGet = (leftToProcess > numBuffElements) ?  numBuffElements : leftToProcess;

		getComplex(buffer, numBuffElements, recsToGet);

		//CHECK_CUDA(cudaMemcpy(inDeviceMem, inBuffer, (sizeof(Complex) * numBuffElements), cudaMemcpyHostToDevice));

		CHECK_FFT_STATUS( cufftExecC2C(plan, (cufftComplex *)buffer, (cufftComplex *)buffer, CUFFT_FORWARD) );

		//CHECK_CUDA(cudaMemcpy(outBuffer, outDeviceMem, (sizeof(Complex) * numBuffElements), cudaMemcpyDeviceToHost));

		clearComplex(buffer, numBuffElements);

		leftToProcess -= recsToGet;
		recProcessed += recsToGet;
		std::cout << " \tProcessed " << recProcessed << " out of " << args->records << std::endl;
	}


	std::cout << "Done"	<< std::endl;

	cudaFree(buffer);


	delete args;

	return 0;
}






