/*
 * arg_parse.hpp
 *
 *  Created on: Feb 11, 2018
 *      Author: brad
 */

#ifndef ARG_PARSE_HPP_
#define ARG_PARSE_HPP_

#include "cmp_global.h"
#include <string.h>

typedef struct {
	int		dataSizeGB 		= 0;		// size of the input data in GB
	int		bufferSizeGB 	= 4;		// size of the buffers in GB, default is 4GB
	int 	device 			= 0;		// default device number,  override with the -d argument
	int 	threads 		= 1;		// number of threads, 0 is the same as 1
	int		fftSize			= 128;

	//----- Computer fields
	long	dataSize		= 0;		// size in bytes
	long	buffSize		= 0;		// size in bytes
	long 	buffElements	= 0;		// number of elements in te buffer
	long	records			= 0;		// number of records
	long	iter			= 0;		// number of iterations
} arguments_t;



class ArgParser {

public:

	static void print_usage() {
		std::cout << "cmpX " << std::endl;
		std::cout << "\t-data xxx     :  size of data in GB" 			<< std::endl;
		std::cout << "\t-buff xxx     :  buffer size in GB"				<< std::endl;
		std::cout << "\t-fft xxx      :  FFT size"						<< std::endl;
		std::cout << "\t-device x     :  CUDA device" 					<< std::endl;
		std::cout << "\t-threads x    :  Number of threads"				<< std::endl;

		exit(0);
	}


	static arguments_t * parseArgs(int argc, char * argv[]) {

		arguments_t *a = new arguments_t;

		// parse the arguments
		int i = 1;
		while(i<argc) {

			//--- Data Set Size
			if(!strcmp(argv[i],"-data")) {
				++i;
				a->dataSizeGB = atoi(argv[i]);
			}

			//--- Buffer Size
			else if(!strcmp(argv[i],"-buff")) {
				++i;
				a->bufferSizeGB = atoi(argv[i]);
			}

			else if(!strcmp(argv[i],"-fft")) {
				++i;
				a->fftSize = atoi(argv[i]);
			}

			else if(!strcmp(argv[i],"-threads")) {
				++i;
				a->threads = atoi(argv[i]);
			}

			//=================================
			// 		GPU Info
			//=================================
			//--- CUDA Device
			else if(!strcmp(argv[i],"-device")) {
				++i;
				a->device = atoi(argv[i]);
			}

			//=================================
			// 		Help or Error
			//=================================
			else if(!strcmp(argv[i],"-help")) {
				print_usage();
			}
			else{
				std::cout << "Unknown argument flag " << argv[i] << std::endl;
				print_usage();
			}

			++i;
		}

		if ( a->dataSizeGB == 0) 	print_usage();
		if ( a->bufferSizeGB == 0) 	print_usage();

		a->dataSize = a->dataSizeGB * GB;
		a->buffSize = a->bufferSizeGB * GB;

		// Since the data is in two parts, the number of elements in the buffer is half the specified size
		// Additionally, the data is a float, which is 4 Bytes
		// so divide the buffer by (2 * 4) or 8
		a->buffElements = a->buffSize / 8;

		if ( (a->buffSize % 8) != 0 ) {
			std::cout << "Buffer size is not a multiple of 8 bytes " << std::endl;
			print_usage();
		}


		// the number of iterations over the data is based on the number of records in the data
		// element == float == 4bytes,   but there are two elements per record, so divide by 8
		a->records	= a->dataSize / 8;
		a->iter 	= (long)std::ceil(a->records / a->buffElements);

		if ( (a->records % 8) != 0 ) {
			std::cout << "The number of records  is not a multiple of 8 bytes "<< std::endl;
			print_usage();
		}


		return a;
	}


	static void printArgs(arguments_t * args) {
		std::cout << "Data Set Size   " << args->dataSizeGB 	<< " GB" << " or\t\t" 	<< args->dataSize << " bytes " << std::endl;
		std::cout << "Buffer Size     " << args->bufferSizeGB 	<< " GB" <<	" or\t\t" 	<< args->buffSize << " bytes " << std::endl;
		std::cout << "FFT Size        " << args->fftSize 		<< std::endl;
		std::cout << "Buffer Elements " << args->buffElements 	<< std::endl;
		std::cout << "Threads         " << args->threads 		<< std::endl;
		std::cout << "Iterations      " << args->iter 			<< std::endl;
		std::cout << "Records         " << args->records		<< std::endl;
	}


};

#endif /* ARG_PARSE_HPP_ */
