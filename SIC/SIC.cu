#include "SIC_header.cuh"
#include "SIC_Generate.cpp"
#include "SIC_OPA.cpp"

	__device__ double QPSK(double signal)
	{
		//printf("B signal: %.5f\n", signal);
		if (signal>0)
			signal = 1;
		else
			signal = -1;
		//printf("A signal: %.5f\n", signal);
		return signal;
	}

	__device__ double QAM16(double signal)
	{
		//printf("B signal: %.5f\n", signal);
		if (signal < 0.25)
			signal = 1;
		else if (signal >= 0.25 && signal < 0.5)
			signal = -1;
		else if (signal >= 0.5 && signal < 0.75)
			signal = 3;
		else if (signal >= 0.75)
			signal = -3;
		//printf("A signal: %.5f\n", signal);
		return signal;
	}

	__device__ double QAM64(double signal)
	{
		//printf("B signal: %.5f\n", signal);
		if (signal < 0.03125)
			signal = 7;
		else if (signal >= 0.03125 && signal < 0.0625)
			signal = 5;
		else if (signal >= 0.0625 && signal < 0.75)
			signal = 3;
		else if (signal >= 0.75 && signal < 0.09375)
			signal = 1;
		else if (signal >= 0.09375 && signal < 0.125)
			signal = -1;
		else if (signal >= 0.125 && signal < 0.15625)
			signal = -3;
		else if (signal >= 0.15625 && signal < 0.1875)
			signal = -5;
		if (signal >= 0.1875)
			signal = -7;
		//printf("A signal: %.5f\n", signal);
		return signal;
	}

	__global__ void SIC(float *powerCoefficients, double *Rayleigh, double *receivedSignal)
	{
		int index = threadIdx.x + blockIdx.x*blockDim.x;
		//int index = threadIdx.x;
		double signal[2];
		double sumSignalPowerCoefficientChannel[2];
		//index of the thread is the id of user, which is being decoded
		if (index < cellSize)
		{
			//printf("lucky I am");
			//int order = index % numberOfUEs;
			int order = 1;
			for (int i = numberOfUEs; i >= order; i--)
			{
				//int order = index % i;
				//printf("oder = %d \n", order);
				signal[0] = (receivedSignal[index] - sumSignalPowerCoefficientChannel[0]) / (Rayleigh[index]*powerCoefficients[i]);
				signal[1] = (receivedSignal[index + cellSize] - sumSignalPowerCoefficientChannel[1]) / (Rayleigh[index + cellSize] * powerCoefficients[i]);
				switch (modulation)
				{
					case 4:
					{
						 signal[0] = QPSK(signal[0]);
						 signal[1] = QPSK(signal[1]);
						 break;
					}
					case 16:
					{
						 signal[0] = QAM16(signal[0]);
						 signal[1] = QAM16(signal[1]);
						 break;
					}
					case 64:
					{
						 signal[0] = QAM64(signal[0]);
						 signal[1] = QAM64(signal[1]);
						 break;
					}
				}
				if (i != order)
				{
					sumSignalPowerCoefficientChannel[0] = sumSignalPowerCoefficientChannel[0] + (Rayleigh[index] * signal[0] * powerCoefficients[i]);
					sumSignalPowerCoefficientChannel[1] = sumSignalPowerCoefficientChannel[1] + (Rayleigh[index + cellSize] * signal[1] * powerCoefficients[i]);
				}
			}
			//printf("RayleighReal = %.3f; receivedSignalReal = %.3f;\n", Rayleigh[index], receivedSignal[index]);
			//printf("RayleighImag = %.3f; receivedSignalImag = %.3f;\n", Rayleigh[index + numberOfUEs], receivedSignal[index + numberOfUEs]);
			receivedSignal[index] = signal[0];
			receivedSignal[index + cellSize] = signal[1];
		}
	}

	int main(void)
	{
		int totalPower = 1;
		int order = cellSize % numberOfUEs;
		int i = 0;
		srand((signed)time(NULL));
		int *generatedSignal;
		SIC_Generate generate;
		switch (modulation)
		{
			case 4:
			{
				generatedSignal = generate.getGeneratedQPSKSignal();
				printf("4\n");
				break;
			}
			case 16:
			{
				generatedSignal = generate.getGeneratedQAM16Signal();
				//printf("16\n");
				break;
			}
			case 64:
			{
				generatedSignal = generate.getGeneratedQAM64Signal();
				//printf("64\n");
				break;
			}
		}
		float *powerCoefficientMatrix;
		SIC_OPA opa;
		powerCoefficientMatrix = opa.getOPA();

		double *rayleighChannel;
		rayleighChannel = getGeneratedRayleighChannel(powerCoefficientMatrix);
		float signalWithPowerCoefficient[cellSize * 2];

		for (i = 0; i<cellSize; i++)
		{
			signalWithPowerCoefficient[i] = generatedSignal[i] * sqrt(powerCoefficientMatrix[order] * totalPower);
			signalWithPowerCoefficient[i + cellSize] = generatedSignal[i + cellSize] * sqrt(powerCoefficientMatrix[order] * totalPower);
		}

		float superSignalReal = 0;
		float superSignalImag = 0;

		for (i = 0; i < cellSize; i++)
		{
			superSignalReal = superSignalReal + signalWithPowerCoefficient[i];
			superSignalImag = superSignalImag + signalWithPowerCoefficient[i + cellSize];
		}

		double receivedSignal[cellSize * 2];
		for (i = 0; i < cellSize; i++)
		{
			//noise is considered as zero
			receivedSignal[i] = superSignalReal*rayleighChannel[i];
			receivedSignal[i + cellSize] = superSignalImag*rayleighChannel[i + cellSize];
		}

		float  *dev_PowerCoefficientMatrix;
		double  *dev_RayleighChannel;
		double  *dev_ReceivedSignal;

		cudaMalloc((void**)&dev_PowerCoefficientMatrix, numberOfUEs * sizeof(float));
		cudaMemcpy(dev_PowerCoefficientMatrix, powerCoefficientMatrix, numberOfUEs * sizeof(float), cudaMemcpyHostToDevice);

		cudaMalloc((void**)&dev_RayleighChannel, cellSize * 2 * sizeof(double));
		cudaMemcpy(dev_RayleighChannel, rayleighChannel, cellSize * 2 * sizeof(double), cudaMemcpyHostToDevice);

		cudaMalloc((void**)&dev_ReceivedSignal, cellSize * 2 * sizeof(double));
		cudaMemcpy(dev_ReceivedSignal, receivedSignal, cellSize * 2 * sizeof(double), cudaMemcpyHostToDevice);

		cudaEvent_t start, stop;
		float elapsedTime;
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
		cudaEventRecord(start, 0);
		//CALLING CUDA START  ****************************************************************************************
		SIC << <cellCoefficient, groupSize >> > (dev_PowerCoefficientMatrix, dev_RayleighChannel, dev_ReceivedSignal);
		//CALLING CUDA END    ****************************************************************************************
		cudaEventRecord(stop, 0);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&elapsedTime, start, stop);

		cudaMemcpy(&receivedSignal, dev_ReceivedSignal, cellSize * 2 * sizeof(double), cudaMemcpyDeviceToHost);

		cudaFree(dev_PowerCoefficientMatrix);
		cudaFree(dev_RayleighChannel);
		cudaFree(dev_ReceivedSignal);
		/*
		for(i=0; i<cellSize; i++)
		{
			printf("%.2f %.2f\n",receivedSignal[i], receivedSignal[i+cellSize] );
		}
		*/
		//write results into file
		FILE *fp3;
		char fileLocation_2[128] = "/home/talgat/github/SIC/results.txt";
		fp3 = fopen(fileLocation_2, "w");
		printf("Time to generate: %.3f ms", elapsedTime);
		fprintf(fp3, "%.3f", elapsedTime);
		fclose(fp3);
		printf("\n");
		return 0;
	}
