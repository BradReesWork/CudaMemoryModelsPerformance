

#ifndef GLOBAL_H_
#define GLOBAL_H_

#include <stdio.h>
#include <stdlib.h>

#include <iostream>
#include <cmath>
#include <random>
#include <cuda_runtime.h>
#include <cufft.h>

#include <omp.h>


#define	GB		long(1073741824)

typedef float2 Complex;

#define		FFT_TYPE	cufftType(CUFFT_C2C)
#define		BATCH_SIZE	int(1)


// prototypes
void getComplex( Complex *data, const long arraySize, const long numToGet);
void clearComplex(Complex data[], long numToSave);
void printValues(Complex *inData, Complex *outData, int num);

void getComplex(Complex *data, const long arraySize, const long numToGet)
{
	long x;
	float r;
	const long p = arraySize / omp_get_max_threads();

	std::random_device rd{};
	std::mt19937 eng{rd()};
	std::normal_distribution<float> dist;


	if ( numToGet > arraySize ) {
		std::cout << "ERROR !!!   numToGet is larger than arraySize " << numToGet << "  > " << arraySize << std::endl;
		exit(-1);
	}

#pragma omp parallel for private(x,r, rd, eng, dist) schedule(static, p)
	for (x = 0 ; x < numToGet; x++) {
		r = dist(eng);
		data[x].x = r;
		data[x].y = r * 3;
	}

	if (numToGet < arraySize) {
		for (long x = numToGet ; x < arraySize; x++) {
			data[x].x = 0;
			data[x].y = 0;
		}
	}
}


void clearComplex(Complex *data, long numToSave)
{
	long idx;
	const long p = numToSave / omp_get_max_threads();

	#pragma omp parallel for private(idx) schedule(static, p)
	for (idx = 0; idx < numToSave; idx++) {
		data[idx].x = 0;
		data[idx].y = 0;
	}
}


void printValues(Complex *inData, Complex *outData, int num) {

	for (int x = 0; x < num; x++) {
		std::cout << "FFT(" << inData[x].x << " , " << inData[x].y << ")\t\t==> (";
		std::cout << outData[x].x << " , " << outData[x].y << ")" << std::endl;
	}
}







#endif /* GLOBAL_H_ */
