/*
 * get_data.h
 *
 *  Created on: Feb 6, 2018
 *      Author: brad
 */

#ifndef GET_DATA_H_
#define GET_DATA_H_

#include <stdio.h>
#include <stdlib.h>
#include <random>

using namespace std;

/**
 *
 */
long GetData( float2 data[], long arraySize, long numToGet, bool zeroPad)
{
	long	count = 0;

	if ( arraySize <= numToGet) {
		int x = 0;

		for ( ; x < numToGet; x++) {
			data[x] = rand();
		}

		count += x;
	}

}



#endif /* GET_DATA_H_ */
