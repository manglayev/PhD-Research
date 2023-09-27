#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <cstring>
#include <time.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

//number of clusters is a group size
//one thread is assigned per cluster
#define numberOfUEs 10
#define groupSize 32
//this is cellCoefficients is for the number of blocks initiated
#define cellCoefficient 6
#define cellSize 1920
// always check cell size:
// cell size = numberOfUEs x groupSize x cell coefficient
#define modulation 16
int * getGeneratedQPSKSignal(void);
int * getGeneratedQAM16Signal(void);
int * getGeneratedQAM64Signal(void);
float getRandomFloat(void);
float uniform0to1Random(void);
float * getOPA(void);
double * getGeneratedRayleighChannel(float *powerCoefficientMatrix);
int lengthOfLineFunction(FILE *fp2, char *fileLocation, char *line, int lengthOfLine);
