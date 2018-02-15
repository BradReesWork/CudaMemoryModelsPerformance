################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
../src/mainB.cpp \
../src/mainP.cpp 

O_SRCS += \
../src/mainB.o 

OBJS += \
./src/mainB.o \
./src/mainP.o 

CPP_DEPS += \
./src/mainB.d \
./src/mainP.d 


# Each subdirectory must supply rules for building sources it contributes
src/%.o: ../src/%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	/usr/local/cuda-9.1/bin/nvcc -G -g -O0 -gencode arch=compute_60,code=sm_60 -gencode arch=compute_70,code=sm_70  -odir "src" -M -o "$(@:%.o=%.d)" "$<"
	/usr/local/cuda-9.1/bin/nvcc -G -g -O0 --compile  -x c++ -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


