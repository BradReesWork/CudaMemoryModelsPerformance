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

#include "ArgParser.hpp"



//-----------------------------------------------------------------
/**
 * Main entry point
 */
int main(int argc, char * argv[]) {

	//-----------------------------------------------------------------------------------
	//			Parse the arguments and compute some variables, then print
	//-----------------------------------------------------------------------------------
	auto args = ArgParser::parseArgs(argc, argv);
	//ArgParser::printArgs(args);

    omp_set_num_threads(args->threads);
	long numBuffElements =	args->buffElements; 		// this is just for readability


	//-----------------------------------------------------------------------------------
	//			Allocate Host Memory
	//-----------------------------------------------------------------------------------

	// the data buffer - what data is read into
	Complex	*inBuffer;

	inBuffer  = (Complex *)malloc (sizeof(Complex) * numBuffElements);


	//-----------------------------------------------------------------------------------
	//			Process
	//-----------------------------------------------------------------------------------
	long recProcessed 	= 0;
	long recsToGet		= 0;
	long leftToProcess 	= args->records;

	std::cout << "Data" << std::endl;

	while (leftToProcess > 0) {

		recsToGet = (leftToProcess > numBuffElements) ?  numBuffElements : leftToProcess;

		getComplex(inBuffer, numBuffElements, recsToGet);

		for ( int z = 0; z <  recsToGet; z++)
			std::cout << (inBuffer[z].x * 200) << std::endl;

		leftToProcess -= recsToGet;
		recProcessed += recsToGet;
	}

	free(inBuffer);
	delete args;

	return 0;
}






