#include <stdio.h>
#include <time.h>
#include <complex>
#include <cuComplex.h>
#include <iostream>
#include "Header.cuh"

double getGeneratedRandom()
{
	int randomNumberInt = rand() % ((100 - (-100)) + 1) - 100;
	double randomNumber = (double)randomNumberInt / 100;
	return randomNumber;
}

cuDoubleComplex * getChannel()
{
	double power = (double) 1 / (double)numberOfUEs;
	static cuDoubleComplex channel[numberOfUEs];
	for (int i = 0; i < numberOfUEs; i++)
	{
		channel[i] = make_cuDoubleComplex(sqrt(power)*getGeneratedRandom(), sqrt(power)*getGeneratedRandom());
		//printf("0 channel %.2f, %.2f\n", cuCreal(channel[i]), cuCimag(channel[i]));
	}
	return channel;
}

cuDoubleComplex getSignal(cuDoubleComplex * channel)
{
	srand((unsigned)time(0));
	static cuDoubleComplex noise[numberOfUEs];
	cuDoubleComplex message;
	static cuDoubleComplex signal = make_cuDoubleComplex(0, 0);
	static cuDoubleComplex signalArray[numberOfUEs];
	for (int i = 0; i < numberOfUEs; i++)
	{
		//printf("channel %.2f, %.2f\n", cuCreal(channel[i]), cuCimag(channel[i]));
		noise[i] = make_cuDoubleComplex(sqrt(0.5)*getGeneratedRandom(), sqrt(0.5)*getGeneratedRandom());
		//printf("noise %.2f, %.2f\n", cuCreal(noise[i]), cuCimag(noise[i]));
		if (modulation == 4)
			message = getQAM4();
		if (modulation == 16)
			message = getQAM16();
		//printf("message %.2f, %.2f\n", cuCreal(message), cuCimag(message));
		message = cuCmul(message, channel[i]);
		//printf("message plus channel%.2f, %.2f\n", cuCreal(message), cuCimag(message));
		signalArray[i] = cuCadd(message, noise[i]);
		//printf("signalArray %.2f, %.2f\n", cuCreal(signalArray[i]), cuCimag(signalArray[i]));
		signal = cuCadd(signal, signalArray[i]);
	}
	//printf("signal: %.3f;\n", cuCreal(signal), cuCimag(signal));
	return signal;
}

cuDoubleComplex getQAM4()
{
	static cuDoubleComplex QAM4;
	double RealPart = getGeneratedRandom();
	double ImagPart = getGeneratedRandom();

	if (RealPart>0)
		RealPart = 1;
	else
		RealPart = -1;

	if (ImagPart>0)
		ImagPart = 1;
	else
		ImagPart = -1;

	QAM4 = make_cuDoubleComplex(RealPart, ImagPart);

	return QAM4;
}

cuDoubleComplex getQAM16()
{
	static cuDoubleComplex QAM16;
	double RealPart = getGeneratedRandom();
	double ImagPart = getGeneratedRandom();

	if (RealPart < 0.25)
		RealPart = 1;
	else if (RealPart >= 0.25 && RealPart < 0.5)
		RealPart = -1;
	else if (RealPart >= 0.5 && RealPart < 0.75)
		RealPart = 3;
	else if (RealPart >= 0.75)
		RealPart = -3;

	if (ImagPart < 0.25)
		ImagPart = 1;
	else if (ImagPart >= 0.25 && ImagPart < 0.5)
		ImagPart = -1;
	else if (ImagPart >= 0.5 && ImagPart < 0.75)
		ImagPart = 3;
	else if (ImagPart >= 0.75)
		ImagPart = -3;
	QAM16 = make_cuDoubleComplex(RealPart, ImagPart);
	return QAM16;
}
