/*
 * GetData.hpp
 *
 *  Created on: Feb 11, 2018
 *      Author: brad
 */

#ifndef GETDATA_HPP_
#define GETDATA_HPP_

#include "cmp_global.h"
#include "CpuTimer.hpp"

#include <random>
#include <omp.h>

using namespace std;

class GetData
{
private:
	CpuTimer * timer	= NULL;


public:
	GetData()	{timer = new CpuTimer(); };
	~GetData()	{ delete timer;}

	float getDurationSec(){return timer->getDurationSec(); }


	/**
	 * Single thread (main thread)
	 */
	void readComplexOMP( Complex *data, long arraySize, long numToGet, int numThreads)
	{
		timer->start();

		if ( numToGet > arraySize ) {
			cout << "ERROR !!!   numToGet is larger than arraySize " << numToGet << "  vs " << arraySize << endl;
			exit(-1);
		}


		#pragma omp parallel for
		for (long x = 0 ; x < numToGet; x++) {
			if ( x == 0)
				cout << "using " << omp_get_num_threads() << " threads" << endl;
			data[x].x = rand();
			data[x].y = rand();
		}


		if (numToGet < arraySize) {
			for (long x = numToGet ; x < arraySize; x++) {
				data[x].x = 0;
				data[x].y = 0;
			}
		}

		timer->stop();
	}


	/**
	 * Single thread (main thread)
	 */
	void readComplex( Complex *data, long arraySize, long numToGet)
	{
		long x = 0;

		timer->start();

		if ( arraySize <= numToGet) {
			for ( ; x < numToGet; x++) {
				data[x].x = rand();
				data[x].y = rand();
			}
		}

		if (numToGet < arraySize) {
			for ( ; x < arraySize; x++) {
				data[x].x = 0;
				data[x].y = 0;
			}
		}

		timer->stop();
	}


};


#endif /* GETDATA_HPP_ */
