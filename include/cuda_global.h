/*
 * Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
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

#ifndef __CUDA_GLOBAL_H__
#define __CUDA_GLOBAL_H__
//-----------------------------------------------

#include <cstdio>
#include <cuda_runtime.h>
#include <cufft.h>

#ifdef __cplusplus
#define LINKAGE "C"
#else
#define LINKAGE
#endif

#define CHECK_CUDA(call) {                                                   \
    cudaError err = call;                                                    \
    if( cudaSuccess != err) {                                                \
        fprintf(stderr, "Cuda error in file '%s' in line %i : %s.\n",        \
                __FILE__, __LINE__, cudaGetErrorString( err) );              \
        exit(EXIT_FAILURE);                                                  \
    } }

#define CHECK_ERROR(errorMessage) {                                          \
    cudaError_t err = cudaGetLastError();                                    \
    if( cudaSuccess != err) {                                                \
        fprintf(stderr, "Cuda error: %s in file '%s' in line %i : %s.\n",    \
                errorMessage, __FILE__, __LINE__, cudaGetErrorString( err) );\
        exit(EXIT_FAILURE);                                                  \
    }                                                                        \
    }

#define CHECK_FFT_STATUS(call) {                                             \
    cufftResult err = call;                                                  \
    if( err != CUFFT_SUCCESS) {                                             \
        fprintf(stderr, "Cuda FFT error\n" );              					\
        exit(EXIT_FAILURE);                                                  \
    } }



void init_cuda(int dev)
{
	// get a CUDA device
    cudaDeviceProp deviceProp;

	CHECK_CUDA(cudaSetDevice(dev));
    CHECK_CUDA(cudaGetDeviceProperties(&deviceProp, dev));

	std::cout << "using device " << dev 				<< std::endl;
	std::cout << "\tName       " << deviceProp.name		<< std::endl;
	std::cout << "\tCompute    " << deviceProp.major << "." << deviceProp.minor << std::endl;
}




//-----------------------------------------------
#endif
