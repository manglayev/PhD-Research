#include <cuComplex.h>
#ifndef __CUDACC__
	#define __CUDACC__
#endif
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <device_functions.h>
#include <stdio.h>
#include <ctime>
#include <time.h>
#include <cstdlib>
#include "Header.cuh"

__device__ cuDoubleComplex decodeQAM4(cuDoubleComplex signal)
{
	float real = cuCreal(signal);
	float imag = cuCimag(signal);

	//printf("REAL %.2f, IMAG %.2f\n", real, imag);

	if (real>0)
		real = 1;
	else
		real = -1;
	if (imag>0)
		imag = 1;
	else
		imag = -1;
	signal = make_cuDoubleComplex(real, imag);
	return signal;
}

__device__ cuDoubleComplex decodeQAM16(cuDoubleComplex signal)
{
	float real = cuCreal(signal);
	float imag = cuCimag(signal);

	//printf("REAL %.2f, IMAG %.2f\n", real, imag);

	if (real >= 0  && real < 2)
		real = 1;
	else if (real >= 2)
		real = 3;
	else if (real < 0  && real > -2)
		real = -1;
	else if (real <= -2)
		real = -3;

if (imag >= 0 && imag < 2)
	imag = 1;
else if (imag >= 2)
imag = 3;
else if (imag < 0 && imag > -2)
imag = -1;
else if (imag <= -2)
imag = -3;

signal = make_cuDoubleComplex(real, imag);
return signal;
}

__global__ void PIC(cuDoubleComplex * channel, cuDoubleComplex * signal)
{
	int i = threadIdx.x + blockIdx.x*blockDim.x;
	int index = threadIdx.x;
	__shared__ cuDoubleComplex decodedSignal[UEsPerCluster];
	__shared__ cuDoubleComplex channelWithPower[UEsPerCluster];
	__shared__ cuDoubleComplex cache1[UEsPerCluster];
	__shared__ cuDoubleComplex cache2[UEsPerCluster];
	//__shared__ cuDoubleComplex decodedMessage[UEsPerCluster];
	if (i < numberOfUEs)
	{
		//printf("undecoded signal: %d %.2f, %.2f\n", index, cuCreal(signal[index]), cuCimag(signal[index]));
		float power = (float)1 / (float)numberOfUEs;
		channelWithPower[index] = cuCmul(channel[i], make_cuDoubleComplex(power, 0));
		//printf("%d channelWithPower %.2f, %.2f\n", index, cuCreal(channelWithPower[index]), cuCimag(channelWithPower[index]));
		if (modulation == 4)
			decodedSignal[index] = decodeQAM4(cuCdiv(*signal, channelWithPower[index]));
		if (modulation == 16)
			decodedSignal[index] = decodeQAM16(cuCdiv(*signal, channelWithPower[index]));
		//printf("%d decodedSignal %.2f, %.2f\n", index, cuCreal(decodedSignal[index]), cuCimag(decodedSignal[index]));
	}
	__syncthreads();
	cuDoubleComplex temp1 = make_cuDoubleComplex(0, 0);
	cuDoubleComplex temp2 = make_cuDoubleComplex(0, 0);
	while (index < UEsPerCluster)
	{
		temp1 = cuCadd(temp1, decodedSignal[index]);
		temp2 = cuCadd(temp2, channelWithPower[index]);
		index++;
	}
	index = threadIdx.x;
	cache1[index] = temp1;
	cache2[index] = temp2;
	/*
	__syncthreads();
	if (index == 0)
	{
	printf("sumDecodedSignal %.2f, %.2f\n", cuCreal(cache1[0]), cuCimag(cache1[0]));
	printf("sumChannelWithPower %.2f, %.2f\n", cuCreal(cache2[0]), cuCimag(cache2[0]));
	}
	*/
	__syncthreads();

	if (i < numberOfUEs)
	{
		if (modulation == 4)
			decodeQAM4(cuCadd((cuCdiv(*signal, cache2[0]), cache1[0]), decodedSignal[index]));
		//decodedMessage[i] = decodeQAM4(cuCadd((cuCdiv(*signal, cache2[0]), cache1[0]), decodedSignal[i]));
		if (modulation == 16)
			decodeQAM16(cuCadd((cuCdiv(*signal, cache2[0]), cache1[0]), decodedSignal[index]));
		//decodedMessage[i] = decodeQAM16(cuCadd((cuCdiv(*signal, cache2[0]), cache1[0]), decodedSignal[i]));
		//printf("%d decodedMessage %.2f, %.2f\n", i, cuCreal(decodedMessage[i]), cuCimag(decodedMessage[i]));
	}
}

__global__ void SIC(cuDoubleComplex * channel, cuDoubleComplex * signal)
{
	int i = threadIdx.x + blockIdx.x*blockDim.x;
	int index = threadIdx.x;
	__shared__ cuDoubleComplex decodedSignal[UEsPerCluster];
	__shared__ cuDoubleComplex channelWithPower[UEsPerCluster];

	if (i < numberOfUEs)
	{
		//printf("undecoded signal: %d %.2f, %.2f\n", index, cuCreal(signal[index]), cuCimag(signal[index]));
		float power = (float)1 / (float)numberOfUEs;
		for (int i = UEsPerCluster; i >= 0; i--)
		{
			decodedSignal[index] = cuCdiv(cuCsub(*signal, channelWithPower[index]), cuCmul(channel[i], make_cuDoubleComplex(power, 0)));
			if (modulation == 4)
				decodedSignal[index] = decodeQAM4(decodedSignal[index]);
			if (modulation == 16)
				decodedSignal[index] = decodeQAM16(decodedSignal[index]);
			if (i != UEsPerCluster)
			{
				channelWithPower[index] = cuCadd(channelWithPower[index], cuCmul(channel[i], cuCmul(decodedSignal[index], make_cuDoubleComplex(power, 0))));
			}
		}
		//printf("decoded SIC signal: %d %.2f, %.2f\n", index, cuCreal(decodedSignal[index]), cuCimag(decodedSignal[index]));
	}
}

int main()
{
	cuDoubleComplex * channel = getChannel();
	cuDoubleComplex signal = getSignal(channel);
	cuDoubleComplex signalArray[numberOfUEs];
	for (int a = 0; a < numberOfUEs; a++)
	{
		if (ic == 'S')
			signalArray[a] = getSignal(channel);
		if (ic == 'P')
			signalArray[a] = signal;
		//printf("generated signal: %d %.2f, %.2f\n", a, cuCreal(signalArray[a]), cuCimag(signalArray[a]));
	}

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

	cuDoubleComplex *dev_channel;
	cuDoubleComplex *dev_signal;

	cudaMalloc((void**)&dev_channel, numberOfUEs*sizeof(cuDoubleComplex));
	cudaMalloc((void**)&dev_signal,  numberOfUEs*sizeof(cuDoubleComplex));

	cudaMemcpy(dev_channel, channel, numberOfUEs*sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_signal,  signalArray, numberOfUEs*sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);
	
	if (ic == 'P')
		PIC << < clusters, UEsPerCluster >> >(dev_channel, dev_signal);
	if (ic == 'S')
		SIC << < clusters, UEsPerCluster >> >(dev_channel, dev_signal);

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	float elapsedTime;
	cudaEventElapsedTime(&elapsedTime, start, stop);
	printf("cancellation: %cIC; modulation: %d; UEs per cluster: %d; all UEs: %d; elapsed time: %.5f ms; \n", ic, modulation, UEsPerCluster, numberOfUEs, elapsedTime);
	cudaFree(dev_channel);
	cudaFree(dev_signal);

	return 0;
}
