

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
void readComplex( Complex *data, const long arraySize, const long numToGet);
void saveComplex(Complex data[], long numToSave);
void printValues(Complex *inData, Complex *outData, int num);

// global
std::random_device rd{};
std::mt19937 eng{rd()};
std::normal_distribution<float> dist;

void readComplex( Complex *data, const long arraySize, const long numToGet)
{
	long x;
	float a, b;

	if ( numToGet > arraySize ) {
		std::cout << "ERROR !!!   numToGet is larger than arraySize " << numToGet << "  > " << arraySize << std::endl;
		exit(-1);
	}

	for (x = 0 ; x < numToGet; x++) {

		a = dist(eng);
		b = dist(eng);
		data[x].x = a;
		data[x].y = b;
	}

	if (numToGet < arraySize) {
		for (long x = numToGet ; x < arraySize; x++) {
			data[x].x = 0;
			data[x].y = 0;
		}
	}
}


void saveComplex(Complex *data, long numToSave)
{
	for (int idx = 0; idx < numToSave; idx++) {
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
