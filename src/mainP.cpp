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

	std::cout << "Memory Model performance Test - PINNED" << std::endl;

	//-----------------------------------------------------------------------------------
	//			Parse the arguments and compute some variables, then print
	//-----------------------------------------------------------------------------------
	auto args = ArgParser::parseArgs(argc, argv);
	ArgParser::printArgs(args);

    omp_set_num_threads(args->threads);
	std::cout << "Num of CPUs     " << omp_get_num_procs() << std::endl;
    std::cout << "Max threads     " << omp_get_max_threads() << std::endl;

	long numDataElements =	args->buffElements; 		// this is just for readability

	// grab the cuda device
	init_cuda(args->device);		// init will exit on error

	//-----------------------------------------------------------------------------------
	//			Allocate Host Memory
	//-----------------------------------------------------------------------------------

	// the data buffer - what data is read into
	std::cout << "Allocating System Memory" << std::endl;
	Complex	*inBuffer;
	Complex	*outBuffer;

	CHECK_CUDA(cudaMallocHost((void**)&inBuffer, (sizeof(Complex) * numDataElements) ));
	CHECK_CUDA(cudaMallocHost((void**)&outBuffer, (sizeof(Complex) * numDataElements) ));


	//-----------------------------------------------------------------------------------
	//			Initialize CUDA and Allocate memory
	//-----------------------------------------------------------------------------------
	cufftHandle		plan;
	Complex			*inDeviceMem;
	Complex			*outDeviceMem;

	std::cout << "Allocating GPU Memory" << std::endl;
	CHECK_CUDA(cudaMalloc ((void**)&inDeviceMem, (sizeof(Complex) * numDataElements)));
	CHECK_CUDA(cudaMalloc ((void**)&outDeviceMem, (sizeof(Complex) * numDataElements)));

	// create the FFT plan
	CHECK_FFT_STATUS( cufftPlan1d(&plan, args->fftSize, FFT_TYPE, BATCH_SIZE) );


	//-----------------------------------------------------------------------------------
	//			Process
	//-----------------------------------------------------------------------------------
	int iter = args->iter;
	long recProcessed = 0;
	for ( int i = 0; i < iter; i++) {

		readComplex(inBuffer, numDataElements, numDataElements);

		CHECK_CUDA(cudaMemcpy(inDeviceMem, inBuffer, (sizeof(Complex) * numDataElements), cudaMemcpyHostToDevice));

		CHECK_FFT_STATUS( cufftExecC2C(plan, (cufftComplex *)inDeviceMem, (cufftComplex *)outDeviceMem, CUFFT_FORWARD) );

		CHECK_CUDA(cudaMemcpy(outBuffer, outDeviceMem, (sizeof(Complex) * numDataElements), cudaMemcpyDeviceToHost));

		//printValues(inBuffer, outBuffer, 10);

		saveComplex(outBuffer, numDataElements);

		recProcessed += numDataElements;
		std::cout << " \tProcessed " << recProcessed << " out of " << args->records << std::endl;
	}

	if (recProcessed < args->records) {
		long leftToProcess = args->records - recProcessed;
		std::cout << " \tProcessed the last " << leftToProcess << " out of " << args->records << std::endl;

		readComplex(inBuffer, numDataElements, leftToProcess);

		CHECK_CUDA(cudaMemcpy(inDeviceMem, inBuffer, (sizeof(Complex) * numDataElements), cudaMemcpyHostToDevice));

		CHECK_FFT_STATUS( cufftExecC2C(plan, (cufftComplex *)inDeviceMem, (cufftComplex *)outDeviceMem, CUFFT_FORWARD) );

		CHECK_CUDA(cudaMemcpy(outBuffer, outDeviceMem, (sizeof(Complex) * numDataElements), cudaMemcpyDeviceToHost));

		saveComplex(outBuffer, numDataElements);
	}



	std::cout << "Done"	<< std::endl;

	cudaFree(inDeviceMem);
	cudaFree(outDeviceMem);
	free(inBuffer);
	free(outBuffer);

	delete args;

	return 0;
}






