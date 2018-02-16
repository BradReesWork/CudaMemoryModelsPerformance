# Copyright (c) 2016, NVIDIA CORPORATION. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

CUDA_HOME=/usr/local/cuda
MKL_INC=/opt/intel/mkl/include

CC=gcc
CPP=g++ 
LD=g++

# -pedantic

CUDACC=$(CUDA_HOME)/bin/nvcc -Xcompiler
CUDA_ARCH=-arch=sm_70
CUDACFLAGS=-m64 -c -O3 --ptxas-options=-v 

CFLAGS=-std=c++11 -fopenmp -fPIC -DADD_ -fmax-errors=3 -W -Wall -c -O3 -I./include -I$(CUDA_HOME)/include

LDFLAGS = \
	-lm -fopenmp\
	-L$(CUDA_HOME)/lib64 -lcudart -lcufft
	
	
OBJB= src/mainB.o 
OBJP= src/mainP.o
OBJU= src/mainU.o

H=  \
	include/ArgParser.hpp 


all: cmpB cmpP cmpU cmpO

cmpB: ${OBJB} Makefile ${H} 
	${LD} -o cmpB ${OBJB} ${LDFLAGS}


cmpP: ${OBJP} Makefile ${H}
	${LD} -o cmpP ${OBJP} ${LDFLAGS}


cmpU: ${OBJU} Makefile ${H}
	${LD} -o cmpU ${OBJU} ${LDFLAGS}

			
%.o: %.cpp Makefile
	$(CPP) $(CFLAGS) $< -o $@

#%.o: %.cu Makefile
#	$(CUDACC) $(CUDACFLAGS) $(CUDA_ARCH) $<

clean:
	rm -rf src/*.o cmp*
