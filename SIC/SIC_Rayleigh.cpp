#include "SIC_header.cuh"
double * getGeneratedRayleighChannel(float *powerCoefficientMatrix) {
	static double rayleighChannel[cellSize * 2];
	int real = 0;
	int imag = 0;
	int order = cellSize % numberOfUEs;
	for (int i = 0; i < cellSize; i++)
	{
		real = rand() % 19 + (-9);
		imag = rand() % 19 + (-9);

		if (real < 0)
			real = (-1)*real;

		if (imag < 0)
			imag = (-1)*imag;

		//printf("%d %d \n", real, imag);

		rayleighChannel[i] = powerCoefficientMatrix[order] * 10 * sqrt(0.5)*real;
		rayleighChannel[i + cellSize] = powerCoefficientMatrix[order] * 10 * sqrt(0.5)*imag;

		//char op = rayleighChannel[i + cellSize] < 0 ? '-' : '+';
		//printf("%d %f %c %fi\n", i, rayleighChannel[i], op, fabsf(rayleighChannel[i + cellSize]));
	}

	return rayleighChannel;
}
