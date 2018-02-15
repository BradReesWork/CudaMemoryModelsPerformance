/*
 * SaveData.hpp
 *
 *  Created on: Feb 11, 2018
 *      Author: brad
 */

#ifndef SAVEDATA_HPP_
#define SAVEDATA_HPP_

#include "cmp_global.h"
#include "CpuTimer.hpp"


class SaveData{
private:
	CpuTimer * timer	= NULL;

public:
	SaveData() {timer = new CpuTimer();}
	~SaveData(){delete timer;}

	float getDurationSec(){return timer->getDurationSec(); }

	void saveComplex(Complex data[], long numToSave) {
		timer->start();

		for (int idx = 0; idx < numToSave; idx++) {
			data[idx].x = 0;
			data[idx].y = 0;
		}
		timer->stop();
	}

	void saveCheckComplex(Complex data[], long numToSave, float value) {
		timer->start();

		for (int idx = 0; idx < numToSave; idx++)
		{
			if (data[idx].x != value) {
				std::cout << "!!! X value is wrong = " << data[idx].x << " should be " << value << std::endl;
			}
			data[idx].x = 0;

			if (data[idx].y != value) {
				std::cout << "!!! Y value is wrong = " << data[idx].y << " should be " << value << std::endl;
			}
			data[idx].y = 0;
		}

		timer->stop();
	}

};

#endif /* SAVEDATA_HPP_ */
