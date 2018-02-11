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
 *
 */

#ifndef __CPU_TIMERC_H_
#define __CPU_TIMERC_H_

#include <chrono>


class CpuTimer {
	typedef std::chrono::milliseconds 				milliseconds;

public:

	CpuTimer(){
		duration = (milliseconds)0;
	}

	virtual ~CpuTimer() {;}

	void start(){
		t1 = clock::now();
	}


	void stop(){
		auto t2 = clock::now();
		duration =  duration + _diff(t1, t2);
	}
	void reset(){
		duration =  (milliseconds)0;
	}

	milliseconds getDuration(){
		return duration;
	}

	double getDurationSec(){
		double s =  ((double)duration.count()) / 1000;
		return s;
	}


protected:
	using clock 	 	= std::chrono::high_resolution_clock;
	using point_t		= std::chrono::high_resolution_clock::time_point;

	point_t 			t1;
	milliseconds	duration;


	milliseconds _diff(const point_t& start, const point_t& end) {
		return std::chrono::duration_cast<milliseconds>(end - start);
	}

};

#endif
