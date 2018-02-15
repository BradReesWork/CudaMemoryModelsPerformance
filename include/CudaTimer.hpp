/*
 * TimerC.h
 *
 *  Created on: Aug 6, 2017
 *      Author: brad
 */

#ifndef __CUDA_TIMERC_H_
#define __CUDA_TIMERC_H_

#include "cuda_global.h"

class CudaTimer {

public:

	CudaTimer(){
		CHECK_CUDA(cudaEventCreate(&startEvent));
		CHECK_CUDA(cudaEventCreate(&stopEvent));
	}

	virtual ~CudaTimer() {
		cudaEventDestroy(startEvent);
		cudaEventDestroy(stopEvent);
	}

	void start(){
		cudaEventRecord(startEvent);
	}


	void stop(){
		cudaEventRecord(stopEvent);
		cudaEventSynchronize(stopEvent);
		cudaEventElapsedTime(&milliseconds, startEvent, stopEvent);

		duration =  duration + milliseconds;
	}
	void reset(){
		duration = 0;
	}

	float getDuration(){
		return duration;
	}

	float getDurationSec(){
		return (duration / 1000.0);
	}


protected:
	cudaEvent_t startEvent, stopEvent;
	float 	milliseconds = 0;
	float	duration	 = 0;


};

#endif
